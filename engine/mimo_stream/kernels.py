"""MXFP4 MoE expert kernels for MiMo-V2.6-Flash, organised by distinct expert.

Adapted from mlx-serve's fused gather-QMV kernels (MIT, github.com/ddalcu/mlx-serve, src/transformer.zig
`gatherQmvGateUpRowsSource` / `gatherQmvDownReduceRowsSource` and the e2m1 bit-shift decode), changed to:
  * MXFP4 (OCP): e2m1 nibbles, 8 per uint32 (low nibble first), one e8m0 scale per 32 values;
  * work is keyed by *distinct expert* rather than (row, expert): each expert's weights are decoded once and
    dotted with every token row routed to it (up to R rows), so verify widths M>1 don't re-read weights;
  * experts are addressed by slot index into a resident pool [S, out, in/8], so the same kernels serve the
    SSD-streaming slot pools.

Layout per MoE layer (MiMo): gate/up [S, 2048, 512] u32 + [S, 2048, 128] u8; down [S, 4096, 256] u32 +
[S, 4096, 64] u8. Activations bf16.

mlx-serve: Copyright (c) 2026 David Dalcu, MIT License (full text in THIRD_PARTY_NOTICES.md).
"""
import mlx.core as mx
import numpy as np

HEADER = """
// e2m1 nibble -> half by shifting bits into the half exponent/mantissa (value * 2^-14); exact for all codes.
inline float mxs_e2m1(uint c) {
  return float(as_type<half>(ushort(((c & 0x7u) << 9) | ((c & 0x8u) << 12))));
}
// e8m0 byte -> 2^(s-127) as float (s=0 -> 2^-127 subnormal; s=255 -> NaN per OCP).
inline float mxs_e8m0(uint s) {
  if (s == 0u) return as_type<float>(0x00400000u);
  if (s == 255u) return NAN;
  return as_type<float>(s << 23);
}
"""

# ---- Kernel A: h[u, r, n] = silu(gate_u[n] . x[row]) * (up_u[n] . x[row]) for rows r routed to expert u.
_GATEUP_SRC = """
  // One simdgroup per (expert u, output n). Each lane handles whole 32-value scale groups:
  // one uint4 of packed weights (4 words = 32 e2m1) + one e8m0 scale per group, x read as 4 x bf16x8.
  uint lane = thread_index_in_simdgroup;
  uint n = thread_position_in_grid.y;
  uint u = thread_position_in_grid.z;
  const int K = KDIM;
  const int N = NDIM;
  const int KG = K / 32;                     // groups per row (= uint4 chunks per row)
  constexpr int IT = (KDIM / 32 + 31) / 32;  // groups per lane
  int cnt = ucnt[u];
  uint slot = uint(uslot[u]);
  const device uint4* wg4 = (const device uint4*)(wg) + ((size_t)slot * N + n) * KG;
  const device uint4* wu4 = (const device uint4*)(wu) + ((size_t)slot * N + n) * KG;
  size_t sbase = ((size_t)slot * N + n) * KG;
  float ag[R], au[R];
  for (int r = 0; r < R; ++r) { ag[r] = 0.0f; au[r] = 0.0f; }
  for (int i = 0; i < IT; ++i) {
    int grp = int(lane) + 32 * i;
    if (grp >= KG) break;
    uint4 qg = wg4[grp];
    uint4 qu = wu4[grp];
    float scg = mxs_e8m0(uint(sg[sbase + grp]));
    float scu = mxs_e8m0(uint(su[sbase + grp]));
    for (int r = 0; r < R; ++r) {
      if (r >= cnt) break;
      const device vec<bfloat16_t, 8>* xv8 = (const device vec<bfloat16_t, 8>*)(x + (size_t)urows[u * R + r] * K + grp * 32);
      float dg = 0.0f, du = 0.0f;
      for (int w = 0; w < 4; ++w) {
        vec<bfloat16_t, 8> xv = xv8[w];
        uint a = qg[w], b = qu[w];
        for (int j = 0; j < 8; ++j) {
          float xf = float(xv[j]);
          dg += xf * mxs_e2m1((a >> (4 * j)) & 0xFu);
          du += xf * mxs_e2m1((b >> (4 * j)) & 0xFu);
        }
      }
      ag[r] += dg * scg;
      au[r] += du * scu;
    }
  }
  for (int r = 0; r < R; ++r) {
    if (r >= cnt) break;
    float gs = simd_sum(ag[r]) * 16384.0f;   // undo the 2^-14 of the e2m1 decode
    float us = simd_sum(au[r]) * 16384.0f;
    if (lane == 0) {
      bfloat16_t gt = bfloat16_t(gs);
      bfloat16_t ut = bfloat16_t(us);
      bfloat16_t sig = sigtab[as_type<ushort>(gt)];
      h[((size_t)u * R + r) * N + n] = (gt * sig) * ut;
    }
  }
"""

# ---- Kernel B: part[u, r, m] = w[u, r] * (down_u[m] . h[u, r])   (float32; scatter-added per row afterwards)
_DOWN_SRC = """
  uint lane = thread_index_in_simdgroup;
  uint m = thread_position_in_grid.y;
  uint u = thread_position_in_grid.z;
  const int K = KDIM;            // = intermediate (2048)
  const int H = HDIM;            // = hidden (4096)
  const int KG = K / 32;
  constexpr int IT = (KDIM / 32 + 31) / 32;
  int cnt = ucnt[u];
  uint slot = uint(uslot[u]);
  const device uint4* wd4 = (const device uint4*)(wd) + ((size_t)slot * H + m) * KG;
  size_t sbase = ((size_t)slot * H + m) * KG;
  float a[R];
  for (int r = 0; r < R; ++r) a[r] = 0.0f;
  for (int i = 0; i < IT; ++i) {
    int grp = int(lane) + 32 * i;
    if (grp >= KG) break;
    uint4 q = wd4[grp];
    float sc = mxs_e8m0(uint(sd[sbase + grp]));
    for (int r = 0; r < R; ++r) {
      if (r >= cnt) break;
      const device vec<bfloat16_t, 8>* hv8 = (const device vec<bfloat16_t, 8>*)(h + ((size_t)u * R + r) * K + grp * 32);
      float s = 0.0f;
      for (int w = 0; w < 4; ++w) {
        vec<bfloat16_t, 8> hv = hv8[w];
        uint b = q[w];
        for (int j = 0; j < 8; ++j) s += float(hv[j]) * mxs_e2m1((b >> (4 * j)) & 0xFu);
      }
      a[r] += s * sc;
    }
  }
  for (int r = 0; r < R; ++r) {
    float t = (r < cnt) ? simd_sum(a[r]) * 16384.0f * float(uw[u * R + r]) : 0.0f;
    if (lane == 0) part[((size_t)u * R + r) * H + m] = t;
  }
"""

_cache = {}


def _kernel(name, src, inputs, outputs):
    if name not in _cache:
        _cache[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                            source=src, header=HEADER)
    return _cache[name]


_sigtab = None


def sigmoid_table():
    """sigmoid for every bf16 bit pattern (mlx-serve's SwiGLU table trick)."""
    global _sigtab
    if _sigtab is None:
        bits = np.arange(1 << 16, dtype=np.uint32) << 16
        vals = mx.array(bits.view(np.float32)).astype(mx.bfloat16)
        _sigtab = mx.sigmoid(vals)
        mx.eval(_sigtab)
    return _sigtab


def group_routes(inds, scores, R=8):
    """CPU grouping of routes by distinct expert.
    inds: np.int [M, 8] expert (or slot) ids, scores: np.float [M, 8].
    Returns uids [U], urows [U, R] (row index, padded with 0), ucnt [U], uw [U, R] (router weight, 0 pad),
    and gather_idx [M*k] = flat (u*R + r) position of each (row, k) route, for a deterministic reduction."""
    M, k = inds.shape
    flat = inds.reshape(-1)
    uids, inv = np.unique(flat, return_inverse=True)
    U = len(uids)
    urows = np.zeros((U, R), np.int32)
    uw = np.zeros((U, R), np.float32)
    ucnt = np.zeros(U, np.int32)
    rows = np.repeat(np.arange(M), k)
    sc = scores.reshape(-1)
    gidx = np.zeros(flat.size, np.uint32)
    for t in range(flat.size):
        u = inv[t]
        c = ucnt[u]
        if c >= R:
            raise ValueError("more rows than R routed to one expert")
        urows[u, c] = rows[t]
        uw[u, c] = sc[t]
        gidx[t] = u * R + c
        ucnt[u] = c + 1
    return uids, urows, ucnt, uw, gidx


def moe_experts(x, pools, uslot, urows, ucnt, uw, R=8, gather_idx=None):
    """Routed expert output for M token rows.
    x: bf16 [M, 4096]; pools: dict with gate_w, gate_s, up_w, up_s, down_w, down_s (slot pools);
    uslot [U] int32 slot per distinct expert; urows/ucnt/uw from group_routes (as mx arrays or numpy).
    Returns bf16 [M, 4096] = sum over each row's experts of score * expert(x_row)."""
    M, K = x.shape
    N = pools["gate_w"].shape[1]
    H = pools["down_w"].shape[1]
    uslot, urows, ucnt, uw = (mx.array(a) if isinstance(a, np.ndarray) else a for a in (uslot, urows, ucnt, uw))
    U = uslot.shape[0]
    ka = _kernel(f"mxs_gateup_R{R}", _GATEUP_SRC, ["x", "wg", "sg", "wu", "su", "uslot", "urows", "ucnt", "sigtab"], ["h"])
    (h,) = ka(inputs=[x, pools["gate_w"], pools["gate_s"], pools["up_w"], pools["up_s"], uslot, urows, ucnt, sigmoid_table()],
              template=[("R", R), ("KDIM", K), ("NDIM", N)],
              grid=(32, N, U), threadgroup=(32, 8, 1), output_shapes=[(U, R, N)], output_dtypes=[mx.bfloat16])
    kb = _kernel(f"mxs_down_R{R}", _DOWN_SRC, ["h", "wd", "sd", "uslot", "ucnt", "uw"], ["part"])
    (part,) = kb(inputs=[h, pools["down_w"], pools["down_s"], uslot, ucnt, uw],
                 template=[("R", R), ("KDIM", N), ("HDIM", H)],
                 grid=(32, H, U), threadgroup=(32, 8, 1), output_shapes=[(U, R, H)], output_dtypes=[mx.float32])
    # Deterministic reduction: gather each row's 8 partials back in route order and sum in fixed order
    # (a scatter-add with float atomics is order-nondeterministic, which flips near-tied routes downstream).
    if gather_idx is None:
        return mx.zeros((M, H), dtype=mx.float32).at[urows.reshape(-1)].add(part.reshape(-1, H)).astype(mx.bfloat16)
    Y = part.reshape(-1, H)[gather_idx].reshape(M, -1, H)
    return Y.sum(axis=1).astype(mx.bfloat16)
