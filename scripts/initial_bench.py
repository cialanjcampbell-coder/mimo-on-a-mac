#!/usr/bin/env python3
"""Feasibility micro-benchmarks for MiMo SSD streaming on MLX.

  initial_bench.py <model_dir> [--only contention|gather|sync|zerocopy]

contention  GPU read bandwidth alone vs while 8 threads pread expert files (F_NOCACHE) at full speed,
            and SSD throughput alone vs during GPU load.
gather      stock mx.gather_qmm(mode="mxfp4") cost for MiMo expert shapes (gate/up 2048x4096,
            down 4096x2048) over a 128-slot pool, M = 1..8,16 token rows x top-8 random slots.
sync        cost of a GPU->CPU sync per MoE layer: 47 x (small op + eval of [M,8] ids) vs one eval.
zerocopy    can os.preadv write straight into an evaluated mx.array's memory (via memoryview /
            numpy view), and does a later GPU op see the new bytes?
"""
import argparse, fcntl, glob, os, random, threading, time, json
import numpy as np
import mlx.core as mx

F_NOCACHE, F_RDAHEAD = 48, 45


def open_nocache(p):
    fd = os.open(p, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    fcntl.fcntl(fd, F_RDAHEAD, 0)
    return fd


def gpu_bw(seconds, n=2**29):
    a = mx.random.uniform(shape=(n,))
    mx.eval(a)
    mx.eval(a.sum())
    t0, it = time.time(), 0
    while time.time() - t0 < seconds:
        mx.eval(a.sum())
        it += 1
    return it * n * 4 / (time.time() - t0) / 1e9


def ssd_reader(files, stop, out, block=4 << 20, threads=8):
    fds = [(open_nocache(f), os.path.getsize(f)) for f in files]
    tot = [0] * threads

    def work(i):
        buf = bytearray(block)
        mv = memoryview(buf)
        rnd = random.Random(i)
        while not stop.is_set():
            fd, sz = rnd.choice(fds)
            off = rnd.randrange(0, sz - block) // 16384 * 16384
            tot[i] += os.preadv(fd, [mv], off)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    t0 = time.time()
    for t in ts:
        t.start()
    stop.wait()
    for t in ts:
        t.join()
    out.append(sum(tot) / (time.time() - t0) / 1e9)


def contention(model):
    files = sorted(glob.glob(f"{model}/experts-L*.safetensors"))
    alone_gpu = gpu_bw(6)
    stop, out = threading.Event(), []
    th = threading.Thread(target=ssd_reader, args=(files, stop, out))
    th.start()
    time.sleep(0.5)
    both_gpu = gpu_bw(6)
    stop.set()
    th.join()
    stop2, out2 = threading.Event(), []
    th = threading.Thread(target=ssd_reader, args=(files, stop2, out2))
    th.start()
    time.sleep(6)
    stop2.set()
    th.join()
    r = {"gpu_alone_GBs": alone_gpu, "gpu_with_ssd_GBs": both_gpu, "ssd_with_gpu_GBs": out[0], "ssd_alone_GBs": out2[0]}
    print(json.dumps(r, indent=1))
    return r


def gather(model, S=128, reps=30):
    def pool(out, inn):
        w = mx.random.normal((S, out, inn)).astype(mx.bfloat16)
        q = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
        mx.eval(*q)
        return q

    g, u, d = pool(2048, 4096), pool(2048, 4096), pool(4096, 2048)
    res = {}
    for M in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        x = mx.random.normal((M, 1, 1, 4096)).astype(mx.bfloat16)
        idx = mx.array(np.stack([np.random.choice(S, 8, replace=False) for _ in range(M)]).astype(np.uint32)).reshape(M, 1, 8)

        def f():
            a = mx.gather_qmm(x, g[0], g[1], None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4")
            b = mx.gather_qmm(x, u[0], u[1], None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4")
            h = (a * mx.sigmoid(a) * b)
            return mx.gather_qmm(h, d[0], d[1], None, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4")

        for _ in range(3):
            mx.eval(f())
        t0 = time.time()
        for _ in range(reps):
            mx.eval(f())
        ms = (time.time() - t0) / reps * 1e3
        uniq = len(np.unique(np.array(idx)))
        res[M] = {"ms": ms, "distinct_experts": uniq, "ms_per_distinct_expert": ms / uniq}
    base = res[1]["ms"]
    for M, r in res.items():
        print(f"M={M:2d}  {r['ms']:6.3f} ms  x{r['ms']/base:4.2f}  distinct={r['distinct_experts']:3d}  "
              f"{r['ms_per_distinct_expert']*1e3:6.1f} us/expert  -> {r['distinct_experts']*13.37e6/ (r['ms']/1e3)/1e9:5.0f} GB/s effective")
    return res


def sync_cost(layers=47, reps=20):
    x = mx.random.normal((1, 4096)).astype(mx.bfloat16)
    w = mx.random.normal((256, 4096)).astype(mx.bfloat16)
    mx.eval(x, w)

    def run(sync_each):
        h = x
        for _ in range(layers):
            s = (h @ w.T).astype(mx.float32)
            ids = mx.argpartition(-s, kth=7, axis=-1)[..., :8]
            if sync_each:
                np.array(ids)  # host readback, like the router sync
            h = h + 0.001 * s[..., :1].astype(h.dtype)
        mx.eval(h)

    out = {}
    for label, se in (("one_eval_per_token", False), ("sync_every_layer", True)):
        run(se)
        t0 = time.time()
        for _ in range(reps):
            run(se)
        out[label] = (time.time() - t0) / reps * 1e3
    out["per_layer_sync_us"] = (out["sync_every_layer"] - out["one_eval_per_token"]) / layers * 1e3
    print(json.dumps(out, indent=1))
    return out


def zerocopy(model):
    f = sorted(glob.glob(f"{model}/experts-L*.safetensors"))[0]
    with open(f, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
    base = 8 + n
    t = hdr["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    stride = (t["data_offsets"][1] - t["data_offsets"][0]) // 256
    pool = mx.zeros((4, 2048, 512), dtype=mx.uint32)
    mx.eval(pool)
    res = {}
    try:
        mv = memoryview(pool)
        res["memoryview_readonly"] = mv.readonly
        view = np.asarray(pool)
        res["numpy_view_writeable"] = bool(view.flags.writeable)
        res["numpy_shares_memory"] = True
        addr = view.__array_interface__["data"][0]
        res["dest_addr_mod_16k"] = addr % 16384
        fd = open_nocache(f)
        e = 7
        dst = memoryview(view[1].reshape(-1).view(np.uint8)) if view.flags.writeable else None
        if dst is None:
            view.setflags(write=True)
            dst = memoryview(view[1].reshape(-1).view(np.uint8))
        t0 = time.time()
        got = os.preadv(fd, [dst], base + t["data_offsets"][0] + e * stride)
        res["pread_bytes"] = got
        res["pread_ms"] = (time.time() - t0) * 1e3
        ref = np.frombuffer(os.pread(fd, stride, base + t["data_offsets"][0] + e * stride), dtype=np.uint32)
        gpu_sum = int(mx.sum(pool[1].astype(mx.uint64)).item()) if hasattr(mx, "uint64") else None
        res["gpu_sees_new_bytes"] = bool(np.array_equal(np.array(pool[1]).reshape(-1), ref))
        res["gpu_sum_matches"] = gpu_sum == int(ref.astype(np.uint64).sum()) if gpu_sum is not None else None
    except Exception as ex:  # report, don't crash
        res["error"] = repr(ex)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--only")
    ap.add_argument("--out")
    a = ap.parse_args()
    results = {}
    for name, fn in (("zerocopy", lambda: zerocopy(a.model)), ("sync", sync_cost), ("gather", lambda: gather(a.model)),
                     ("contention", lambda: contention(a.model))):
        if a.only and a.only != name:
            continue
        print(f"== {name}", flush=True)
        results[name] = fn()
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1, default=str)
