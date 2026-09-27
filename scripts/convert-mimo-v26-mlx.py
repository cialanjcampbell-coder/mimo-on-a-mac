#!/usr/bin/env python3
"""Convert XiaomiMiMo/MiMo-V2.6-Flash-RL (HF, FP8 + MXFP4 experts) to MLX safetensors.

  convert-mimo-v26-mlx.py <hf_snapshot_dir> <out_dir> [--workers 4] [--only experts|dense|mtp|dflash|meta]

Layout (mlx-lm compatible, `quantization: mxfp4 gs32` + per-tensor presence of `.scales`):
  experts-LNN.safetensors  one per MoE layer (1..47): switch_mlp.{gate,up,down}_proj.{weight,scales}
                           stacked [256, out, in/8] uint32 / [256, out, in/32] uint8. The bytes are
                           copied verbatim from the source (MLX mxfp4 = OCP MXFP4, low nibble first,
                           e8m0 scales), so experts are bit-exact. The data section starts on a 16 KiB
                           boundary and every tensor size is a multiple of 16 KiB, so expert e of any
                           projection sits at an aligned offset: base + e * stride (for SSD streaming).
  dense.safetensors        everything else in the text model, FP8 block-scaled -> BF16 (dequantised
                           exactly as the reference: fp8 * scale_inv per 128x128 block); o_proj, router,
                           embeddings, lm_head, norms, sinks are copied as stored (BF16/F32).
  mtp.safetensors          the 3 MTP layers (model.mtp.layers.N.*), same treatment.
  dflash/                  DFlash drafter: config + BF16 weights + mask_embedding.safetensors
                           (extracted from the torch pickle's raw storage without unpickling).
Vision / audio towers are not converted (text-only).
"""
import argparse, json, os, shutil, struct, time, zipfile, fcntl
from concurrent.futures import ThreadPoolExecutor

ALIGN = 16384
BS = 128  # FP8 block size


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h


class SafeIndex:
    """name -> (file, absolute offset, nbytes, dtype, shape) over all source shards."""

    def __init__(self, src, files):
        self.t = {}
        for fn in files:
            p = os.path.join(src, fn)
            base, h = read_header(p)
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.t[k] = (p, base + a, b - a, v["dtype"], v["shape"])
        self.fds = {}

    def raw(self, name):
        p, off, n, _, _ = self.t[name]
        fd = self.fds.get(p)
        if fd is None:
            fd = os.open(p, os.O_RDONLY)
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            self.fds[p] = fd
        buf = bytearray(n)
        mv, got = memoryview(buf), 0
        while got < n:
            r = os.preadv(fd, [mv[got:]], off + got)
            if r <= 0:
                raise IOError(f"short read {name}")
            got += r
        return buf


def write_safetensors(path, tensors, metadata=None):
    """tensors: list of (name, dtype_str, shape, nbytes, producer) where producer() -> bytes-like.
    Header is space-padded so the data section starts at a 16 KiB boundary."""
    hdr, off = {}, 0
    for name, dt, shape, nbytes, _ in tensors:
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    if metadata:
        hdr["__metadata__"] = metadata
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (-(8 + len(hj))) % ALIGN
    hj += b" " * pad
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for name, _, _, nbytes, prod in tensors:
            b = prod()
            if len(b) != nbytes:
                raise ValueError(f"{name}: produced {len(b)} bytes, expected {nbytes}")
            f.write(b)
    os.replace(tmp, path)
    return 8 + len(hj)


# ---------------------------------------------------------------- experts (verbatim bytes)
PROJ = {  # out, in  (source U8 weight is [out, in/2], scale U8 [out, in/32])
    "gate_proj": (2048, 4096),
    "up_proj": (2048, 4096),
    "down_proj": (4096, 2048),
}


def convert_expert_layer(idx, out, L, n_exp):
    path = os.path.join(out, f"experts-L{L:02d}.safetensors")
    if os.path.exists(path):
        return path, "exists"
    tensors = []
    for proj, (o, i) in PROJ.items():
        pre = f"model.layers.{L}.mlp.experts"
        w_n, s_n = o * i // 2, o * i // 32

        def wprod(proj=proj, w_n=w_n):
            buf = bytearray()
            for e in range(n_exp):
                b = idx.raw(f"{pre}.{e}.{proj}.weight")
                assert len(b) == w_n
                buf += b
            return buf

        def sprod(proj=proj, s_n=s_n):
            buf = bytearray()
            for e in range(n_exp):
                b = idx.raw(f"{pre}.{e}.{proj}.weight_scale")
                assert len(b) == s_n
                buf += b
            return buf

        dst = f"model.layers.{L}.mlp.switch_mlp.{proj}"
        tensors.append((f"{dst}.weight", "U32", (n_exp, o, i // 8), n_exp * w_n, wprod))
        tensors.append((f"{dst}.scales", "U8", (n_exp, o, i // 32), n_exp * s_n, sprod))
    for _, _, _, nb, _ in tensors:
        assert nb % ALIGN == 0
    write_safetensors(path, tensors, {"format": "mlx"})
    return path, "written"


# ---------------------------------------------------------------- dense / mtp (FP8 -> BF16)
def dense_convert(idx, cfg, out, which):
    import mlx.core as mx
    import numpy as np

    bf16 = mx.bfloat16

    def load(name):
        _, _, _, dt, shape = idx.t[name]
        b = bytes(idx.raw(name))
        np_dt = {"F8_E4M3": np.uint8, "BF16": np.uint16, "F32": np.float32, "U8": np.uint8}[dt]
        a = mx.array(np.frombuffer(b, dtype=np_dt).reshape(shape))
        if dt == "BF16":
            a = a.view(bf16)
        return a, dt

    def dequant(w8, s):
        w = mx.from_fp8(w8, dtype=mx.float32)
        m, n = w.shape
        pb, pr = (-m) % BS, (-n) % BS
        w = mx.pad(w, ((0, pb), (0, pr))).reshape((m + pb) // BS, BS, (n + pr) // BS, BS)
        w = (w * s.astype(mx.float32)[:, None, :, None]).reshape(m + pb, n + pr)[:m, :n]
        return w.astype(bf16)

    def split_qkv(w8, s, n_h, n_kv, hd, vhd, TP=4):
        # fused qkv stored as TP=4 slices [q_t | k_t | v_t], FP8 scale blocks padded per slice
        q_pr, k_pr, v_pr = n_h // TP * hd, n_kv // TP * hd, n_kv // TP * vhd
        a_pr = q_pr + k_pr + v_pr
        p_pr = -(-a_pr // BS) * BS
        n = w8.shape[1]
        assert w8.shape[0] == TP * a_pr and s.shape[0] * BS == TP * p_pr, (w8.shape, s.shape)
        w = mx.from_fp8(w8, dtype=mx.float32).reshape(TP, a_pr, n)
        w = mx.pad(w, ((0, 0), (0, p_pr - a_pr), (0, (-n) % BS)))
        nc = s.shape[1]
        w = (w.reshape(TP * p_pr // BS, BS, nc, BS) * s.astype(mx.float32)[:, None, :, None])
        w = w.reshape(TP, p_pr, nc * BS)[:, :a_pr, :n]
        q = w[:, :q_pr].reshape(-1, n).astype(bf16)
        k = w[:, q_pr:q_pr + k_pr].reshape(-1, n).astype(bf16)
        v = w[:, q_pr + k_pr:].reshape(-1, n).astype(bf16)
        return q, k, v

    pat = cfg["hybrid_layer_pattern"]
    if which == "dense":
        names = [k for k in idx.t if k.startswith(("model.layers.", "model.embed_tokens", "model.norm", "lm_head"))
                 and ".mlp.experts." not in k]
        fname = "dense.safetensors"
    else:
        names = [k for k in idx.t if k.startswith("model.mtp.")]
        fname = "mtp.safetensors"

    outs = []  # (name, mx array)
    done = set()
    for k in sorted(names):
        if k in done or k.endswith("_scale_inv"):
            continue
        if k.endswith("self_attn.qkv_proj.weight"):
            pre = k[: -len(".qkv_proj.weight")]
            is_swa = which == "mtp" or bool(pat[int(k.split(".")[2])])
            if is_swa:
                dims = (cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"], cfg["swa_head_dim"], cfg["swa_v_head_dim"])
            else:
                dims = (cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["v_head_dim"])
            w8, _ = load(k)
            s, _ = load(k + "_scale_inv")
            q, kk, v = split_qkv(w8, s, *dims)
            outs += [(f"{pre}.q_proj.weight", q), (f"{pre}.k_proj.weight", kk), (f"{pre}.v_proj.weight", v)]
            done |= {k, k + "_scale_inv"}
            continue
        a, dt = load(k)
        if dt == "F8_E4M3":
            s, _ = load(k + "_scale_inv")
            a = dequant(a, s)
        outs.append((k, a))
        done.add(k)

    tensors = []
    for name, a in outs:
        mx.eval(a)
        dtn = {mx.bfloat16: "BF16", mx.float32: "F32"}[a.dtype]
        raw = np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()
        tensors.append((name, dtn, a.shape, len(raw), (lambda raw=raw: raw)))
    write_safetensors(os.path.join(out, fname), tensors, {"format": "mlx"})
    return fname, len(tensors)


def dflash_convert(src, out):
    d_src, d_out = os.path.join(src, "dflash"), os.path.join(out, "dflash")
    os.makedirs(d_out, exist_ok=True)
    shutil.copyfile(os.path.join(d_src, "config.json"), os.path.join(d_out, "config.json"))
    shutil.copyfile(os.path.join(d_src, "dflash_draft_model.safetensors"), os.path.join(d_out, "model.safetensors"))
    # mask_embedding.pt: torch zip; pickle says BFloat16Storage, 4096 elems, stride 1, little-endian.
    # Read the raw storage bytes directly -- never unpickle.
    z = zipfile.ZipFile(os.path.join(d_src, "mask_embedding.pt"))
    assert z.read("mask_embedding/byteorder") == b"little"
    raw = z.read("mask_embedding/data/0")
    assert len(raw) == 4096 * 2
    write_safetensors(os.path.join(d_out, "mask_embedding.safetensors"),
                      [("mask_embedding", "BF16", (4096,), len(raw), lambda: raw)],
                      {"mask_token_id": "151675"})
    shutil.copyfile(os.path.join(d_src, "dflash.py"), os.path.join(d_out, "dflash_reference.py"))


def write_meta(src, out, cfg):
    c = dict(cfg)
    c["source_quantization_config"] = c.pop("quantization_config", None)
    c["quantization"] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    for k in ("vision_config", "audio_config", "processor_config", "auto_map"):
        c.pop(k, None)
    c["mlx_conversion"] = {"source": "XiaomiMiMo/MiMo-V2.6-Flash-RL", "experts": "verbatim MXFP4",
                           "dense": "FP8 block -> BF16", "mtp_file": "mtp.safetensors", "dflash_dir": "dflash",
                           "align": ALIGN}
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=2)
    wm = {}
    for fn in sorted(os.listdir(out)):
        if fn.endswith(".safetensors") and fn != "mtp.safetensors":
            _, h = read_header(os.path.join(out, fn))
            for k in h:
                if k != "__metadata__":
                    wm[k] = fn
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "generation_config.json", "README.md"):
        if os.path.exists(os.path.join(src, fn)):
            shutil.copyfile(os.path.join(src, fn), os.path.join(out, fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", choices=["experts", "dense", "mtp", "dflash", "meta"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    files = sorted(set(json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"].values()))
    idx = SafeIndex(a.src, files)
    t0 = time.time()
    steps = [a.only] if a.only else ["dflash", "mtp", "dense", "experts", "meta"]
    for step in steps:
        if step == "experts":
            layers = [L for L, moe in enumerate(cfg["moe_layer_freq"]) if moe]
            with ThreadPoolExecutor(a.workers) as ex:
                for p, st in ex.map(lambda L: convert_expert_layer(idx, a.out, L, cfg["n_routed_experts"]), layers):
                    print(f"[{time.time()-t0:7.0f}s] {os.path.basename(p)} {st}", flush=True)
        elif step in ("dense", "mtp"):
            res = dense_convert(idx, cfg, a.out, step)
            print(f"[{time.time()-t0:7.0f}s] {step}: {res}", flush=True)
        elif step == "dflash":
            dflash_convert(a.src, a.out)
            print(f"[{time.time()-t0:7.0f}s] dflash done", flush=True)
        elif step == "meta":
            write_meta(a.src, a.out, cfg)
            print(f"[{time.time()-t0:7.0f}s] meta done", flush=True)


if __name__ == "__main__":
    main()
