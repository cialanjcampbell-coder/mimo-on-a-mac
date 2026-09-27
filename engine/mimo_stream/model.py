"""MiMo-V2.6-Flash with SSD-streamed experts on MLX (Phase 1 prototype).

Dense weights (attention, norms, layer-0 MLP, routers, embeddings, lm_head) are resident in BF16.
Each MoE layer keeps S expert slots resident in pool arrays; the rest stay in the per-layer safetensors
files and are pread on demand straight into GPU-visible memory.

Per MoE layer: attention -> FP32 router (sigmoid, noaux bias for selection, top-8, normalised weights)
-> host sync of the routes ->
  * decode / verify (T <= VERIFY_MAX rows): load missing experts into pool slots (decayed-LFU eviction),
    then stock gather_qmm (T == 1) or the distinct-expert kernels (T >= 2);
  * prefill chunks (T > VERIFY_MAX): resident experts from the pool, the rest read into a separate
    staging area that does not disturb the decode cache; sorted gather_qmm over (token, expert) pairs.
"""
import json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from . import mimo_v2
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache

from .cache import SlotCache
from .kernels import group_routes, moe_experts
from .store import ExpertStore, IOPool

E, TOPK = 256, 8
VERIFY_MAX = 8
SHAPES = {"gate": (2048, 4096), "up": (2048, 4096), "down": (4096, 2048)}  # (out, in)
EXPERT_BYTES = sum(o * i // 2 + o * i // 32 for o, i in SHAPES.values())  # 13,369,344


def alloc_bank(S):
    """Slot bank: {key_w/key_s: mx array} + {(key, kind): numpy view} for pread destinations."""
    arrs, views = {}, {}
    for k, (o, i) in SHAPES.items():
        arrs[f"{k}_w"] = mx.zeros((S, o, i // 8), dtype=mx.uint32)
        arrs[f"{k}_s"] = mx.zeros((S, o, i // 32), dtype=mx.uint8)
    mx.eval(list(arrs.values()))
    for k in SHAPES:
        for kind in ("w", "s"):
            v = np.asarray(arrs[f"{k}_{kind}"])
            if not v.flags.writeable:
                v.setflags(write=True)
            views[(k, kind)] = v
    return arrs, views


def slot_dst(views, s):
    return {key: memoryview(v[s].reshape(-1).view(np.uint8)) for key, v in views.items()}


class Router(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = mx.zeros((E, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((E,))

    def __call__(self, r):
        logits = r.astype(mx.float32) @ self.weight.astype(mx.float32).T   # FP32 like the reference
        scores = mx.sigmoid(logits)
        sel = scores + self.e_score_correction_bias.astype(mx.float32)
        inds = mx.argpartition(-sel, kth=TOPK - 1, axis=-1)[..., :TOPK]
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, w


class Layer(nn.Module):
    def __init__(self, args, L):
        super().__init__()
        self.is_moe = bool(args.moe_layer_freq[L])
        self.is_swa = bool(args.hybrid_layer_pattern[L])
        self.self_attn = mimo_v2.Attention(args, self.is_swa)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        if self.is_moe:
            self.gate = Router(args)
        else:
            self.mlp = mimo_v2.MLP(args)


class Engine:
    def __init__(self, model_dir, slots_per_layer, io_threads=8, profile=None, log=print, nocache=True, prefetch_k=8,
                 dense_bits=16, cache_limit_gb=4):
        self.dir = model_dir
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.args = args = mimo_v2.ModelArgs.from_dict(cfg)
        self.window = args.sliding_window_size
        self.log = log
        t0 = time.time()
        # Wire GPU memory up to the recommended working set so macOS never compresses/swaps the expert pools
        # (unwired, rarely-touched pools got compressed under pressure: 70 GB of a 103 GB footprint), and cap
        # MLX's buffer cache so freed temporaries don't accumulate.
        dev = mx.device_info()
        mx.set_wired_limit(int(dev["max_recommended_working_set_size"]))
        mx.set_cache_limit(cache_limit_gb << 30)
        dense = mx.load(os.path.join(model_dir, "dense.safetensors"))
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.layers = [Layer(args, L) for L in range(args.num_hidden_layers)]

        def take(pre):
            """Pop (and so release) this prefix's source tensors from the lazily-loaded dict."""
            keys = [k for k in dense if k.startswith(pre)]
            return [(k[len(pre):], dense.pop(k)) for k in keys]

        def finish(module):
            # quantise + evaluate one module at a time so BF16 sources and temporaries never pile up
            if dense_bits in (4, 8):
                nn.quantize(module, group_size=64, bits=dense_bits, class_predicate=lambda p, m: isinstance(m, nn.Linear))
            mx.eval(module.parameters())

        self.embed.load_weights(take("model.embed_tokens."))
        mx.eval(self.embed.parameters())
        self.norm.load_weights(take("model.norm."))
        mx.eval(self.norm.parameters())
        self.lm_head.load_weights(take("lm_head."))
        finish(self.lm_head)
        for L, layer in enumerate(self.layers):
            w = take(f"model.layers.{L}.")
            if layer.is_moe:  # router lives at mlp.gate.* in the checkpoint
                w = [(k.replace("mlp.gate.", "gate."), v) for k, v in w]
            layer.load_weights(w, strict=True)
            del w
            finish(layer)
        dense.clear()
        mx.clear_cache()
        self.moe_layers = [L for L, l in enumerate(self.layers) if l.is_moe]
        self.store = ExpertStore(model_dir, self.moe_layers, nocache=nocache)
        self.prefetch_k = prefetch_k
        self.io = IOPool(io_threads)
        self.S = slots_per_layer
        self.banks, self.bank_views, self.slots = {}, {}, {}
        self.slot_mv = {}
        for L in self.moe_layers:
            self.banks[L], self.bank_views[L] = alloc_bank(self.S)
            self.slots[L] = SlotCache(self.S)
            self.slot_mv[L] = [slot_dst(self.bank_views[L], s) for s in range(self.S)]
        # two staging banks (double buffer): prefill layer L computes from one while layer L+1 streams into the other
        self.stages = []
        for _ in range(2):
            arrs, views = alloc_bank(E - self.S) if self.S < E else (None, None)
            self.stages.append((arrs, [slot_dst(views, s) for s in range(E - self.S)] if views else []))
        self.stage_pref = {}   # L -> (bank index, stage_of[256], futures) for a whole-layer prefetch in flight
        self.stage_turn = 0
        self.prefill_prefetch_min = 512   # chunks at least this long prefetch the next layer's non-resident experts
        self.pending = {L: ([], set()) for L in self.moe_layers}   # in-flight prefetch (futures, slots) per layer
        self.next_moe = {L: (self.moe_layers[i + 1] if i + 1 < len(self.moe_layers) else None)
                         for i, L in enumerate(self.moe_layers)}
        self.prefetch = True
        self.log(f"[engine] dense loaded + {len(self.moe_layers)}x{self.S} slots "
                 f"({len(self.moe_layers) * self.S * EXPERT_BYTES / 1e9:.1f} GB) in {time.time() - t0:.1f}s")
        self.stats = {"syncs": 0, "decode_tokens": 0}
        self.warm(profile)

    # ---- residency
    def warm(self, profile=None):
        """Fill every slot: by profile frequency ({L: counts[256]}) if given, else experts 0..S-1."""
        t0 = time.time()
        jobs = []
        for L in self.moe_layers:
            order = np.argsort(-profile[L]) if profile is not None else np.arange(E)
            for e, s in self.slots[L].preload(order[: self.S]):
                jobs += self.store.jobs(L, int(e), self.slot_mv[L][s])
        n = self.io.run(jobs)
        self.log(f"[engine] warmed {n / 1e9:.1f} GB in {time.time() - t0:.1f}s ({n / 1e9 / (time.time() - t0):.1f} GB/s)")

    def make_cache(self):
        return [RotatingKVCache(max_size=self.window) if l.is_swa else KVCache() for l in self.layers]

    # ---- MoE
    def _gather_m1(self, bank, r2, slots, w):
        """Single-row expert sum over the given slots with stock gather_qmm (fast at M=1)."""
        k = len(slots)
        idx = mx.array(slots.astype(np.uint32)).reshape(1, 1, k)
        x4 = r2.reshape(1, 1, 1, -1)
        q = dict(transpose=True, group_size=32, bits=4, mode="mxfp4")
        a = mx.gather_qmm(x4, bank["gate_w"], bank["gate_s"], None, rhs_indices=idx, **q)
        b = mx.gather_qmm(x4, bank["up_w"], bank["up_s"], None, rhs_indices=idx, **q)
        y = mx.gather_qmm(nn.silu(a) * b, bank["down_w"], bank["down_s"], None, rhs_indices=idx, **q)
        return (y.reshape(k, -1).astype(mx.float32) * mx.array(w.astype(np.float32))[:, None]).sum(axis=0)

    def _moe_small(self, L, r2, inds, w):
        """T <= VERIFY_MAX rows: make all needed experts resident, compute from the pool.
        T == 1 overlaps SSD with GPU: the resident experts' part is queued (async_eval) before waiting for
        the missing ones; the CPU only writes victim slots, the GPU only reads hit slots."""
        sc = self.slots[L]
        pend_futs, pend_slots = self.pending[L]
        self.pending[L] = ([], set())
        loads = sc.resolve(np.unique(inds))
        futs = list(pend_futs)
        if loads:
            jobs = []
            for e, s in loads:
                jobs += self.store.jobs(L, e, self.slot_mv[L][s])
            futs += self.io.submit(jobs)
        slot_inds = sc.slot_of[inds]
        bank = self.banks[L]
        T = r2.shape[0]
        if T == 1:
            not_ready = pend_slots | {s for _, s in loads}
            sl, ww = slot_inds[0], w[0]
            ready = np.array([s not in not_ready for s in sl]) if futs else np.ones(len(sl), bool)
            y = None
            if ready.any():
                y = self._gather_m1(bank, r2, sl[ready], ww[ready])
                if futs:
                    mx.async_eval(y)          # GPU starts on resident experts while the SSD reads the rest
            if futs:
                self.io.wait(futs)
            if (~ready).any():
                ym = self._gather_m1(bank, r2, sl[~ready], ww[~ready])
                y = ym if y is None else y + ym
            return y.reshape(1, -1).astype(r2.dtype)
        if futs:
            self.io.wait(futs)
        uids, urows, ucnt, uw, gidx = group_routes(slot_inds, w, R=VERIFY_MAX)
        return moe_experts(r2, bank, uids.astype(np.int32), urows, ucnt, uw, R=VERIFY_MAX, gather_idx=mx.array(gidx))

    def _pair_outputs(self, bank, r2, tok, slot):
        """Expert outputs for (token, slot) pairs already sorted by slot: [P, H] (bf16)."""
        xp = r2[mx.array(tok.astype(np.uint32))][:, None, :]
        idx = mx.array(slot.astype(np.uint32))
        q = dict(transpose=True, group_size=32, bits=4, mode="mxfp4", sorted_indices=True)
        a = mx.gather_qmm(xp, bank["gate_w"], bank["gate_s"], None, rhs_indices=idx, **q)
        b = mx.gather_qmm(xp, bank["up_w"], bank["up_s"], None, rhs_indices=idx, **q)
        return mx.gather_qmm(nn.silu(a) * b, bank["down_w"], bank["down_s"], None, rhs_indices=idx, **q)[:, 0, :]

    def _moe_prefill(self, L, r2, inds, w):
        """Pairs (token, k) are split into pool-resident and staged, each sorted by slot for gather_qmm, then
        put back in (token, k) order and summed over k in fixed order (deterministic; no atomics)."""
        sc = self.slots[L]
        T = r2.shape[0]
        ex = inds.reshape(-1)
        tok = np.repeat(np.arange(T), TOPK)
        slot = sc.slot_of[ex]
        res = slot >= 0
        parts, positions = [], []
        pos_all = np.arange(T * TOPK)
        if res.any():
            p = pos_all[res][np.argsort(slot[res], kind="stable")]
            parts.append(self._pair_outputs(self.banks[L], r2, tok[p], slot[p]))
            positions.append(p)
        if (~res).any():
            pref = self.stage_pref.pop(L, None)
            if pref is not None:   # whole layer already streaming into a staging bank
                bi, stage_of, futs = pref
                need = np.unique(ex[~res])
                assert (stage_of[need] >= 0).all()
                self.io.wait(futs)
            else:
                bi = self.stage_turn
                self.stage_turn ^= 1
                need = np.unique(ex[~res])
                stage_of = np.full(E, -1, np.int32)
                stage_of[need] = np.arange(len(need))
                jobs = []
                for e, s in zip(need, range(len(need))):
                    jobs += self.store.jobs(L, int(e), self.stages[bi][1][s])
                self.io.run(jobs)
            st = stage_of[ex]
            p = pos_all[~res][np.argsort(st[~res], kind="stable")]
            parts.append(self._pair_outputs(self.stages[bi][0], r2, tok[p], st[p]))
            positions.append(p)
        order = np.concatenate(positions)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        Y = mx.concatenate(parts, axis=0)[mx.array(inv.astype(np.uint32))].reshape(T, TOPK, -1)
        y = (Y.astype(mx.float32) * mx.array(w.astype(np.float32))[..., None]).sum(axis=1)
        return y.astype(r2.dtype)

    def _start_stage_prefetch(self, L):
        """Stream every non-resident expert of MoE layer L into the next staging bank (prefill overlap)."""
        if L is None or L in self.stage_pref or self.S >= E:
            return
        bi = self.stage_turn
        self.stage_turn ^= 1
        nonres = np.nonzero(self.slots[L].slot_of < 0)[0]
        stage_of = np.full(E, -1, np.int32)
        stage_of[nonres] = np.arange(len(nonres))
        jobs = []
        for e, s in zip(nonres, range(len(nonres))):
            jobs += self.store.jobs(L, int(e), self.stages[bi][1][s])
        self.stage_pref[L] = (bi, stage_of, self.io.submit(jobs))

    # ---- forward
    def forward(self, tokens, cache, all_logits=False):
        """tokens: list/np of ids. Returns logits [T, vocab] (all_logits) or [vocab] for the last position."""
        h = self.embed(mx.array(np.asarray(tokens, np.int32))[None])
        T = h.shape[1]
        big = T >= self.prefill_prefetch_min
        if big:
            self._start_stage_prefetch(self.moe_layers[0])
        for L, layer in enumerate(self.layers):
            c = cache[L]
            mask = create_attention_mask(h, c, window_size=self.window if layer.is_swa else None)
            x = h + layer.self_attn(layer.input_layernorm(h), mask, c)
            r = layer.post_attention_layernorm(x)
            if not layer.is_moe:
                h = x + layer.mlp(r)
                continue
            inds, w = layer.gate(r)
            nL = self.next_moe[L]
            pred = None
            if self.prefetch and nL is not None and T <= VERIFY_MAX:
                nxt = self.layers[nL]
                pi, pw = nxt.gate(nxt.post_attention_layernorm(x))   # layer L+1 router on layer L's residual
                if self.prefetch_k < TOPK:   # keep only the most confident predicted experts
                    top = mx.argsort(-pw[0, -1])[: self.prefetch_k]
                    pred = pi[0, -1][top][None]
                else:
                    pred = pi[0]
            _t = time.perf_counter()
            mx.eval(inds, w, *([pred] if pred is not None else []))   # the per-layer host sync
            self.stats["t_eval"] = self.stats.get("t_eval", 0.0) + time.perf_counter() - _t
            self.stats["syncs"] += 1
            # convert the evaluated arrays whole: slicing first (inds[0]) makes a new lazy op = another GPU sync
            inds_np = np.array(inds, copy=False)[0].astype(np.int64)
            if getattr(self, "record", None) is not None:
                self.record.append((L, inds_np.copy(), np.array(r[0].astype(mx.float32)), np.array(w[0])))
            w_np = np.array(w, copy=False)[0]
            r2 = r[0]
            if T <= VERIFY_MAX:
                y = self._moe_small(L, r2, inds_np, w_np)
                if pred is not None:   # start reading predicted misses of the next MoE layer; overlaps GPU work
                    loads = self.slots[nL].resolve(np.unique(np.array(pred, copy=False)), count=False, stats=False)
                    if loads:
                        jobs = []
                        for e, s_ in loads:
                            jobs += self.store.jobs(nL, e, self.slot_mv[nL][s_])
                        pf, ps = self.pending[nL]
                        self.pending[nL] = (pf + self.io.submit(jobs), ps | {s_ for _, s_ in loads})
                        self.stats["prefetched"] = self.stats.get("prefetched", 0) + len(loads)
            else:
                y = self._moe_prefill(L, r2, inds_np, w_np)
                if big:   # queue this layer's GPU work, then stream the next layer while it runs
                    mx.async_eval(y)
                    self._start_stage_prefetch(nL)
            h = x + y[None]
        out = self.norm(h[0] if all_logits else h[0, -1:])
        logits = self.lm_head(out).astype(mx.float32)
        _t = time.perf_counter()
        mx.eval(logits)
        self.stats["t_eval"] = self.stats.get("t_eval", 0.0) + time.perf_counter() - _t
        return logits if all_logits else logits[0]

    def decode_step(self, token, cache):
        logits = self.forward([token], cache)
        self.stats["decode_tokens"] += 1
        if self.stats["decode_tokens"] % 16 == 0:
            for L in self.moe_layers:
                self.slots[L].decay()
        return logits

    def generate(self, prompt_ids, max_new, cache=None, chunk=4096, eos=None):
        cache = cache or self.make_cache()
        t0 = time.time()
        logits = None
        for i in range(0, len(prompt_ids), chunk):
            logits = self.forward(prompt_ids[i:i + chunk], cache)
        t_prefill = time.time() - t0
        out = []
        t1 = time.time()
        for _ in range(max_new):
            tok = int(mx.argmax(logits).item())
            out.append(tok)
            if eos is not None and tok in eos:
                break
            logits = self.decode_step(tok, cache)
        t_dec = time.time() - t1
        return out, {"prefill_s": t_prefill, "prefill_tok_s": len(prompt_ids) / t_prefill,
                     "decode_s": t_dec, "decode_tok_s": len(out) / max(t_dec, 1e-9)}

    def hit_rate(self):
        h = sum(s.hits for s in self.slots.values())
        m = sum(s.misses for s in self.slots.values())
        return h / max(h + m, 1), m
