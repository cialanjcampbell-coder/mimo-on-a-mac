#!/usr/bin/env python3
"""Expert-cache simulator over MiMo routing traces (from scripts/mimo_route_trace.py).

  sim_expert_cache.py <routes_dir> [--fractions 0.3,0.4,0.49,0.55,0.6,0.7] [--out results.json]

Train split (<routes_dir>/train) sets static hot sets and per-layer budgets; results are reported on the
held-out test split (<routes_dir>/test), scoring only model-generated tokens (role==1). Context tokens
(role==0) are prefill: they go through a separate staging area and don't touch the decode cache
(the plan's design), except in the `*_pf` variants where they update the dynamic policy too.
The cache persists across test sessions in file order (a long-running server).

Policies (per layer, S slots):
  static      top-S experts by train frequency, never changes
  lru         least recently used
  lfu_decay   ds4-style: +1 per selection, halve all counters every 16 decode tokens, evict min (ties: LRU)
  hybrid      static region (60% of S) + lfu_decay dynamic region
  belady      offline optimal on the decode stream (upper bound)
Per-layer budgets: uniform S, or skew (greedy by marginal train frequency, same total).
Also reports: route concentration, verify-union sizes for M rows, cross-layer prediction accuracy (P, Q),
prefetch-adjusted hit rate, temporal reuse.
"""
import argparse, glob, json, os
import numpy as np

E, K = 256, 8


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        layers = sorted(k for k in z.files if k.startswith("L"))
        out.append({"name": os.path.basename(f), "role": z["role"], "L": {k: z[k] for k in layers},
                    "P": {k: z["P" + k[1:]] for k in layers if "P" + k[1:] in z.files},
                    "Q": {k: z["Q" + k[1:]] for k in layers if "Q" + k[1:] in z.files}})
    return out


def freqs(sessions, layers, decode_only=True):
    F = {l: np.zeros(E, np.int64) for l in layers}
    for s in sessions:
        m = s["role"] == 1 if decode_only else slice(None)
        for l in layers:
            np.add.at(F[l], s["L"][l][m].ravel(), 1)
    return F


def budgets(F, layers, total, mode):
    if mode == "uniform":
        return {l: total // len(layers) for l in layers}
    # greedy: give slots to the layer whose next-most-frequent expert has the highest count
    order = {l: np.sort(F[l])[::-1] for l in layers}
    S = {l: 0 for l in layers}
    import heapq
    h = [(-order[l][0], l) for l in layers]
    heapq.heapify(h)
    for _ in range(total):
        c, l = heapq.heappop(h)
        S[l] += 1
        if S[l] < E:
            heapq.heappush(h, (-order[l][S[l]], l))
    return S


class Layer:
    def __init__(self, S, policy, static_set=None):
        self.S, self.policy = S, policy
        self.res = np.zeros(E, bool)
        self.last = np.zeros(E, np.int64)
        self.cnt = np.zeros(E, np.float64)
        self.pinned = np.zeros(E, bool)
        if policy in ("static", "hybrid") and static_set is not None:
            self.pinned[static_set] = True
            self.res[static_set] = True

    def access(self, ex, t, score=True):
        """ex: 8 expert ids of one token. Returns number of misses."""
        need = np.zeros(E, bool)
        need[ex] = True
        miss = need & ~self.res
        n = int(miss.sum())
        self.last[ex] = t
        self.cnt[ex] += 1
        if self.policy == "static" or n == 0:
            return n
        for e in np.nonzero(miss)[0]:
            if self.res.sum() >= self.S:
                cand = self.res & ~self.pinned & ~need
                if not cand.any():
                    continue  # cannot admit (budget full of pinned/needed) -> served from staging
                idx = np.nonzero(cand)[0]
                if self.policy == "lru":
                    v = idx[np.argmin(self.last[idx])]
                else:  # lfu_decay / hybrid dynamic region
                    key = self.cnt[idx] * 1e12 + self.last[idx]
                    v = idx[np.argmin(key)]
                self.res[v] = False
            self.res[e] = True
        return n

    def decay(self):
        self.cnt *= 0.5


def simulate_verify(test, layers, S, M):
    """lfu_decay cache, decode tokens grouped into verify windows of M rows: each layer loads the union
    of the window's experts once. Returns (misses per window, distinct experts per window, both per layer)."""
    L = {l: Layer(S[l], "lfu_decay") for l in layers}
    t = nwin = ndec = 0
    miss = uni = 0
    for s in test:
        idx = np.nonzero(s["role"] == 1)[0]
        for j in range(0, len(idx) - M + 1, M):
            w = idx[j:j + M]
            t += 1
            nwin += 1
            for l in layers:
                u = np.unique(s["L"][l][w].ravel())
                uni += len(u)
                miss += L[l].access(u, t)
            ndec += M
            if ndec % 16 < M:
                for l in layers:
                    L[l].decay()
    return miss / nwin, uni / nwin


def simulate(test, layers, S, policy, static_sets, prefill_updates=False):
    tot_lookups = tot_miss = 0
    L = {l: Layer(S[l], policy, static_sets.get(l) if static_sets else None) for l in layers}
    if policy == "hybrid":
        for l in layers:  # static region = 60% of S
            k = int(0.6 * S[l])
            L[l].pinned[:] = False
            L[l].res[:] = False
            L[l].pinned[static_sets[l][:k]] = True
            L[l].res[static_sets[l][:k]] = True
    t = 0
    ndec = 0
    for s in test:
        role = s["role"]
        for i in range(len(role)):
            dec = role[i] == 1
            if not dec and not prefill_updates:
                continue
            t += 1
            for l in layers:
                m = L[l].access(s["L"][l][i], t)
                if dec:
                    tot_miss += m
                    tot_lookups += K
            if dec:
                ndec += 1
                if policy in ("lfu_decay", "hybrid") and ndec % 16 == 0:
                    for l in layers:
                        L[l].decay()
    return 1 - tot_miss / max(tot_lookups, 1), tot_miss / max(ndec, 1)


def belady(test, layers, S):
    """Offline optimal per layer on the concatenated decode stream."""
    tot_lookups = tot_miss = 0
    for l in layers:
        seq = np.concatenate([s["L"][l][s["role"] == 1] for s in test])  # [N,8]
        N = len(seq)
        # next use index per (position, expert)
        nxt = np.full((N, E), N + 1, np.int64)
        last = np.full(E, N + 1, np.int64)
        for i in range(N - 1, -1, -1):
            nxt[i] = last
            last[seq[i]] = i
        # nxt[i][e] = next position > i where e used
        res = np.zeros(E, bool)
        cap = S[l]
        for i in range(N):
            need = np.zeros(E, bool)
            need[seq[i]] = True
            miss = need & ~res
            tot_miss += int(miss.sum())
            tot_lookups += K
            for e in np.nonzero(miss)[0]:
                if res.sum() >= cap:
                    cand = np.nonzero(res & ~need)[0]
                    if len(cand) == 0:
                        continue
                    v = cand[np.argmax(nxt[i][cand])]
                    if nxt[i][v] <= nxt[i][e]:  # bypass: new expert is needed later than all residents
                        continue
                    res[v] = False
                res[e] = True
    return 1 - tot_miss / tot_lookups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("routes")
    ap.add_argument("--fractions", default="0.3,0.4,0.49,0.55,0.6,0.7")
    ap.add_argument("--out")
    ap.add_argument("--no-belady", action="store_true")
    a = ap.parse_args()
    train, test = load(os.path.join(a.routes, "train")), load(os.path.join(a.routes, "test"))
    layers = sorted(train[0]["L"].keys())
    Ftr = freqs(train, layers)
    Fall_tr = freqs(train, layers, decode_only=False)
    res = {"n_train_decode": int(sum((s["role"] == 1).sum() for s in train)),
           "n_test_decode": int(sum((s["role"] == 1).sum() for s in test)), "layers": len(layers)}
    print(f"train decode tokens {res['n_train_decode']}, test decode tokens {res['n_test_decode']}, layers {len(layers)}")

    # 1. concentration (train, decode tokens) and train->test stability
    conc = {}
    Fte = freqs(test, layers)
    for k in (8, 16, 32, 64, 128):
        tr = np.mean([np.sort(Ftr[l])[::-1][:k].sum() / Ftr[l].sum() for l in layers])
        # share of TEST routes captured by TRAIN's top-k
        te = np.mean([Fte[l][np.argsort(-Ftr[l])[:k]].sum() / Fte[l].sum() for l in layers])
        conc[k] = {"train_self": tr, "test_by_train_topk": te}
        print(f"top-{k:3d} experts/layer: {tr*100:5.1f}% of train routes, train's top-{k} cover {te*100:5.1f}% of test routes")
    res["concentration"] = conc
    per_layer_top32 = [float(np.sort(Ftr[l])[::-1][:32].sum() / Ftr[l].sum()) for l in layers]
    res["per_layer_top32_share"] = per_layer_top32

    # 2. hit rates
    static_sets = {l: np.argsort(-Fall_tr[l] * 0 - Ftr[l]) for l in layers}  # by decode frequency
    hits = {}
    for f in [float(x) for x in a.fractions.split(",")]:
        total = int(round(f * E * len(layers)))
        for alloc in ("uniform", "skew"):
            S = budgets(Ftr, layers, total, alloc)
            row = {}
            for pol in ("static", "lru", "lfu_decay", "hybrid"):
                ss = {l: static_sets[l][:S[l]] for l in layers} if pol == "static" else static_sets
                h, mpt = simulate(test, layers, S, pol, ss)
                row[pol] = {"hit": h, "miss_per_token": mpt}
            h, mpt = simulate(test, layers, S, "hybrid", static_sets, prefill_updates=True)
            row["hybrid_pf"] = {"hit": h, "miss_per_token": mpt}
            if not a.no_belady and alloc == "skew":
                row["belady"] = {"hit": belady(test, layers, S)}
            hits[f"{f}_{alloc}"] = row
            print(f"f={f:.2f} {alloc:7s} " + "  ".join(f"{p} {v['hit']*100:5.1f}%" for p, v in row.items())
                  + f"   (hybrid misses/token {row['hybrid']['miss_per_token']:.1f})", flush=True)
    res["hit_rates"] = hits

    # 3. verify-union: distinct experts per layer across M consecutive decode tokens, and distinct misses
    #    against a fixed resident set = skew-budget static top-S at f=0.49 (approximation of steady state)
    S49 = budgets(Ftr, layers, int(round(0.49 * E * len(layers))), "skew")
    resident = {l: set(static_sets[l][:S49[l]].tolist()) for l in layers}
    union = {}
    for M in (1, 2, 3, 4, 6, 8):
        u, mi, n = 0.0, 0.0, 0
        for s in test:
            idx = np.nonzero(s["role"] == 1)[0]
            for j in range(0, len(idx) - M + 1, M):
                w = idx[j:j + M]
                for l in layers:
                    st = set(s["L"][l][w].ravel().tolist())
                    u += len(st)
                    mi += len(st - resident[l])
                n += 1
        union[M] = {"distinct_per_layer": u / n / len(layers), "distinct_static_misses_per_layer": mi / n / len(layers)}
        print(f"M={M}: distinct experts/layer {union[M]['distinct_per_layer']:5.1f}  "
              f"distinct misses/layer vs static-49% {union[M]['distinct_static_misses_per_layer']:5.2f}")
    res["verify_union"] = union
    vdyn = {}
    for M in (1, 2, 3, 4, 6, 8):
        mw, uw = simulate_verify(test, layers, S49, M)
        vdyn[M] = {"miss_per_window": mw, "distinct_per_window": uw}
        print(f"M={M}: lfu_decay@49% misses per verify window {mw:6.1f} (per row {mw/M:5.1f}), "
              f"distinct experts per window {uw:6.0f}", flush=True)
    res["verify_dynamic"] = vdyn

    # 4. cross-layer prediction and temporal reuse (decode tokens of test)
    for key in ("P", "Q"):
        acc = []
        for s in test:
            m = s["role"] == 1
            for l in layers:
                if l in s[key]:
                    a_, b_ = s["L"][l][m], s[key][l][m]
                    acc.append(np.mean([len(set(x) & set(y)) / K for x, y in zip(a_, b_)]))
        if acc:
            res[f"pred_{key}"] = float(np.mean(acc))
            print(f"prediction {key}: {np.mean(acc)*100:.1f}% of true top-8 predicted")
    reuse = []
    for s in test:
        m = np.nonzero(s["role"] == 1)[0]
        for l in layers:
            r = s["L"][l]
            reuse += [len(set(r[i]) & set(r[i - 1])) for i in m[1:] if i - 1 >= 0]
    res["temporal_reuse_mean_of_8"] = float(np.mean(reuse))
    print(f"temporal reuse: {np.mean(reuse):.2f} of 8 experts shared with previous token (random ~0.25)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
