"""Expert weights on SSD: offsets from the aligned per-layer safetensors, and a pread thread pool.

Every expert slice in experts-LNN.safetensors starts on a 16 KiB boundary (see scripts/convert-mimo-v26-mlx.py),
and pool/staging arrays are page-aligned MLX allocations, so reads are direct (F_NOCACHE, read-ahead off)
straight into GPU-visible memory. os.preadv releases the GIL, so a plain thread pool reaches SSD speed.
"""
import fcntl, json, os, struct, time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE, F_RDAHEAD = 48, 45
PROJ = ("gate_proj", "up_proj", "down_proj")
KEYS = ("gate", "up", "down")


class ExpertStore:
    def __init__(self, model_dir, moe_layers, n_experts=256, nocache=True):
        self.n = n_experts
        self.loc = {}  # (L, key, "w"|"s") -> (fd, base_offset, stride)
        self.fds = []
        for L in moe_layers:
            p = os.path.join(model_dir, f"experts-L{L:02d}.safetensors")
            with open(p, "rb") as f:
                hn = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hn))
            fd = os.open(p, os.O_RDONLY)
            # nocache=False lets the macOS page cache (RAM outside the GPU pools) act as a second expert tier
            fcntl.fcntl(fd, F_NOCACHE, 1 if nocache else 0)
            fcntl.fcntl(fd, F_RDAHEAD, 0)
            self.fds.append(fd)
            for proj, key in zip(PROJ, KEYS):
                for kind, suffix in (("w", "weight"), ("s", "scales")):
                    a, b = hdr[f"model.layers.{L}.mlp.switch_mlp.{proj}.{suffix}"]["data_offsets"]
                    self.loc[(L, key, kind)] = (fd, 8 + hn + a, (b - a) // n_experts)

    def jobs(self, L, expert, dst):
        """dst: {(key, kind): writable memoryview of one slot}. Returns list of (fd, offset, memoryview)."""
        out = []
        for key in KEYS:
            for kind in ("w", "s"):
                fd, base, stride = self.loc[(L, key, kind)]
                mv = dst[(key, kind)]
                assert mv.nbytes == stride, (L, key, kind, mv.nbytes, stride)
                out.append((fd, base + expert * stride, mv))
        return out

    def close(self):
        for fd in self.fds:
            os.close(fd)


def _read(job):
    fd, off, mv = job
    got = 0
    n = mv.nbytes
    while got < n:
        r = os.preadv(fd, [mv[got:]], off + got)
        if r <= 0:
            raise IOError(f"short read at {off + got}")
        got += r
    return n


class IOPool:
    def __init__(self, threads=8):
        self.ex = ThreadPoolExecutor(threads, thread_name_prefix="mxs-io")
        self.bytes = 0
        self.seconds = 0.0

    def submit(self, jobs):
        """Sort by (fd, offset) for locality; returns futures."""
        jobs = sorted(jobs, key=lambda j: (j[0], j[1]))
        return [self.ex.submit(_read, j) for j in jobs]

    def wait(self, futs):
        t0 = time.perf_counter()
        n = sum(f.result() for f in futs)
        self.seconds += time.perf_counter() - t0
        self.bytes += n
        return n

    def run(self, jobs):
        return self.wait(self.submit(jobs))
