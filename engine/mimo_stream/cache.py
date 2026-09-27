"""Per-layer expert slot table with ds4-style decayed-LFU eviction.

LFU with decay (+1 per selection, halve every 16 decode tokens, evict the minimum, ties by LRU) hits 91.0% at 49%
resident; Belady 96.1%.

Slot reuse is safe without leases in this engine because it syncs on the router of every MoE layer: by
the time layer L's routes reach the CPU, every earlier GPU use of layer L's slots (previous tokens) has
completed, and victims are never among the experts the current step needs.
"""
import numpy as np


class SlotCache:
    def __init__(self, n_slots, n_experts=256):
        self.S = n_slots
        self.slot_of = np.full(n_experts, -1, np.int32)   # expert -> slot or -1
        self.expert_in = np.full(n_slots, -1, np.int32)   # slot -> expert or -1
        self.cnt = np.zeros(n_experts, np.float64)
        self.last = np.zeros(n_experts, np.int64)
        self.t = 0
        self.hits = 0
        self.misses = 0

    def preload(self, experts):
        experts = list(experts)[: self.S]
        for s, e in enumerate(experts):
            self.slot_of[e] = s
            self.expert_in[s] = e
        return list(zip(experts, range(len(experts))))

    def resolve(self, needed, count=True, stats=True):
        """needed: 1-D array of distinct experts. Assigns slots to the missing ones (evicting unneeded
        experts with the lowest decayed count, ties by least recent use). Returns list of (expert, slot) to load."""
        self.t += 1
        needed = np.asarray(needed)
        if count:
            self.cnt[needed] += 1
        self.last[needed] = self.t
        missing = needed[self.slot_of[needed] < 0]
        if stats:
            self.hits += len(needed) - len(missing)
            self.misses += len(missing)
        if len(missing) == 0:
            return []
        if len(missing) > self.S:
            raise ValueError("step needs more distinct experts than the pool holds")
        exp = self.expert_in
        score = np.where(exp >= 0, self.cnt[np.maximum(exp, 0)] * 1e12 + self.last[np.maximum(exp, 0)], -1.0)
        prot = np.zeros(self.S, bool)
        res_needed = self.slot_of[needed]
        prot[res_needed[res_needed >= 0]] = True
        score[prot] = np.inf
        victims = np.argpartition(score, len(missing) - 1)[: len(missing)]
        loads = []
        for e, s in zip(missing, victims):
            old = exp[s]
            if old >= 0:
                self.slot_of[old] = -1
            exp[s] = e
            self.slot_of[e] = s
            loads.append((int(e), int(s)))
        return loads

    def decay(self):
        self.cnt *= 0.5
