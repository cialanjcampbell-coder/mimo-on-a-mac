"""Trial recorder helpers: parse server logs and macmon samples, build a report.

Standard library only. Pure functions; scripts/trial-record is the CLI.
"""
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

HOME = Path.home()
SERVER_LOGS = {
    "mimo": HOME / ".local/state/mimo-server/server.log",
    "mlxserve": HOME / ".local/state/mlx-serve-auto/server.log",
}

_NUM = r"(?:[\d.]+(?:e[+-]?\d+)?|inf)"
MIMO_RE = re.compile(
    rf"\[req\] (?P<mode>\w+) reuse=(?P<reuse>\d+) new=(?P<new>\d+) "
    rf"prefill (?P<prefill_s>{_NUM})s \((?P<prefill_tps>{_NUM}) tok/s\) \| "
    rf"decode (?P<gen>\d+) tok (?P<decode_tps>{_NUM}) tok/s \| "
    rf"misses/tok (?P<misses>{_NUM}) \| (?P<finish>\w+) \| ctx (?P<ctx>\d+)")
MLXSERVE_RE = re.compile(
    rf"<- (?P<prompt>\d+)\+(?P<gen>\d+) tokens (?:streamed|\(\d+ms\)) "
    rf"\[prefill: (?P<prefill_tps>{_NUM}) tok/s(?: \((?P<cached>\d+) cached / \d+ total\))?, "
    rf"decode: (?P<decode_tps>{_NUM}) tok/s\] \[(?P<finish>\w+)\]")


def parse_line(line):
    """One server-log line -> request dict, or None if it isn't a request summary."""
    m = MIMO_RE.search(line)
    if m:
        return {"server": "mimo", "mode": m["mode"], "reused": int(m["reuse"]),
                "new": int(m["new"]), "prefill_s": float(m["prefill_s"]),
                "gen": int(m["gen"]), "decode_tps": float(m["decode_tps"]),
                "misses": float(m["misses"]), "finish": m["finish"], "ctx": int(m["ctx"])}
    m = MLXSERVE_RE.search(line)
    if m:
        prompt, cached, gen = int(m["prompt"]), int(m["cached"] or 0), int(m["gen"])
        tps, new = float(m["prefill_tps"]), prompt - cached
        return {"server": "mlxserve", "mode": "cached" if cached else "fresh",
                "reused": cached, "new": new,
                "prefill_s": round(new / tps, 2) if tps > 0 else 0.0,
                "gen": gen, "decode_tps": float(m["decode_tps"]),
                "misses": None, "finish": m["finish"], "ctx": prompt + gen}
    return None


def _looks_like_request(line):
    s = line.strip()
    return (s.startswith("<- ") or "[req] " in s) and "client disconnected" not in s


def parse_log_text(text):
    """All request lines in a log chunk -> (requests, skipped request-like lines)."""
    reqs, skipped = [], 0
    for line in text.splitlines():
        r = parse_line(line)
        if r:
            reqs.append(r)
        elif _looks_like_request(line):
            skipped += 1
    return reqs, skipped


def log_size(path):
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def read_since(path, offset):
    """Text appended to path since byte offset (the whole file if it shrank, e.g. rotated)."""
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        size = f.seek(0, 2)
        f.seek(offset if offset <= size else 0)
        return f.read().decode("utf-8", errors="replace")


def summarize_requests(reqs):
    gen = sum(r["gen"] for r in reqs)
    timed = [r for r in reqs if r["decode_tps"] > 0 and r["gen"] > 0]
    dec_time = sum(r["gen"] / r["decode_tps"] for r in timed)
    pf = [r["prefill_s"] for r in reqs]
    return {"requests": len(reqs), "generated_tokens": gen,
            "decode_tps": round(sum(r["gen"] for r in timed) / dec_time, 2) if dec_time else None,
            "prefill_s_median": statistics.median(pf) if pf else None,
            "prefill_s_max": max(pf) if pf else None,
            "servers": sorted({r["server"] for r in reqs})}


def load_samples(path):
    """macmon `pipe` JSON lines -> list of dicts (bad or partial lines skipped)."""
    p = Path(path)
    out = []
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _stats(values):
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return None
    return {"mean": round(statistics.fmean(v), 2), "peak": round(max(v), 2)}


GLITCH_W = 400  # macmon occasionally reports one absurd power sample (seen: 1905 W); this machine draws <150 W


def _is_glitch(s):
    return any((_get(s, k) or 0) > GLITCH_W for k in ("sys_power", "all_power", "cpu_power", "gpu_power"))


def summarize_samples(samples, interval_s=1.0):
    total = len(samples)
    samples = [s for s in samples if not _is_glitch(s)]
    n = len(samples)
    if n == 0:
        return {"samples": 0}
    sys_p = [_get(s, "sys_power") for s in samples]
    good = [x for x in sys_p if isinstance(x, (int, float))]
    energy = statistics.fmean(good) * total * interval_s if good else 0.0  # scaled to the whole trial
    fans = [100.0 * f["rpm"] / f["max_rpm"] for s in samples for f in (s.get("fans") or [])
            if isinstance(f, dict) and f.get("max_rpm")]
    busy = [s["gpu_freq_mhz"] for s in samples
            if (_get(s, "gpu_active_ratio") or 0) > 0.5 and _get(s, "gpu_freq_mhz")]
    gpu = {"busy_samples": len(busy)}
    if busy:
        peak = max(busy)
        gpu.update(median_freq_mhz=statistics.median(busy), peak_freq_mhz=peak,
                   throttled_share=round(sum(f < 0.9 * peak for f in busy) / len(busy), 3))
    ram = [x for x in (_get(s, "memory", "ram_usage") for s in samples) if x is not None]
    swap = [x for x in (_get(s, "memory", "swap_usage") for s in samples) if x is not None]
    return {
        "samples": n, "glitches": total - n, "seconds": round(total * interval_s, 1),
        "sys_power_w": _stats(sys_p),
        "all_power_w": _stats([_get(s, "all_power") for s in samples]),
        "gpu_power_w": _stats([_get(s, "gpu_power") for s in samples]),
        "energy_j": round(energy, 1), "energy_wh": round(energy / 3600, 3),
        "gpu_temp_c": _stats([_get(s, "temp", "gpu_temp_avg") for s in samples]),
        "cpu_temp_c": _stats([_get(s, "temp", "cpu_temp_avg") for s in samples]),
        "fan_peak_pct": round(max(fans), 1) if fans else 0.0,
        "gpu": gpu,
        "ram_peak_gb": round(max(ram) / 1e9, 1) if ram else None,
        "swap_delta_gb": round((swap[-1] - swap[0]) / 1e9, 3) if swap else None,
    }


def build_summary(meta, end_iso, reqs, skipped, samples, macmon_exited_early):
    start = datetime.fromisoformat(meta["start"])
    end = datetime.fromisoformat(end_iso)
    totals = summarize_requests(reqs)
    power = summarize_samples(samples, meta.get("interval_s", 1.0))
    gen = totals["generated_tokens"]
    return {"label": meta["label"], "start": meta["start"], "end": end_iso,
            "duration_s": (end - start).total_seconds(), "requests": reqs, "skipped": skipped,
            "totals": totals, "power": power,
            "j_per_token": round(power["energy_j"] / gen, 3) if gen and power["samples"] else None,
            "macmon_exited_early": macmon_exited_early}


def _fmt(stat, unit):
    return "n/a" if not stat else f"mean {stat['mean']} {unit}, peak {stat['peak']} {unit}"


def render_report(s):
    t, p = s["totals"], s["power"]
    lines = [f"# Trial: {s['label']}", "",
             f"- **When:** {s['start']} → {s['end']} ({s['duration_s']:.0f} s)",
             f"- **Server(s):** {', '.join(t['servers']) or 'none seen'}",
             f"- **Requests:** {t['requests']} ({s['skipped']} other/unparsed request lines)", ""]
    if s["requests"]:
        lines += ["| # | server | mode | new | reused | prefill s | gen | decode tok/s | misses/tok | finish |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(s["requests"], 1):
            miss = "" if r["misses"] is None else r["misses"]
            lines.append(f"| {i} | {r['server']} | {r['mode']} | {r['new']} | {r['reused']} | "
                         f"{r['prefill_s']} | {r['gen']} | {r['decode_tps']} | {miss} | {r['finish']} |")
        lines.append("")
    lines += ["## Totals",
              f"- Generated tokens: {t['generated_tokens']}",
              f"- Decode (token-weighted): {t['decode_tps']} tok/s",
              f"- First-token proxy (prefill s): median {t['prefill_s_median']}, max {t['prefill_s_max']}",
              "", "## Power and thermals (macmon)"]
    if p["samples"] == 0:
        lines.append("- No macmon samples.")
    else:
        g = p["gpu"]
        jpt = s["j_per_token"] if s["j_per_token"] is not None else "n/a"
        gpu_line = (f"- GPU busy samples: {g['busy_samples']}, median {g['median_freq_mhz']} MHz, "
                    f"peak {g['peak_freq_mhz']} MHz, below 90% of peak: {100 * g['throttled_share']:.0f}%"
                    if g["busy_samples"] else "- GPU never busy (>50%) in a sample.")
        glitch = f" ({p['glitches']} glitch sample(s) >{GLITCH_W} W dropped)" if p.get("glitches") else ""
        lines += [f"- Samples: {p['samples']} over {p['seconds']} s{glitch}",
                  f"- System power: {_fmt(p['sys_power_w'], 'W')}",
                  f"- CPU+GPU+ANE power: {_fmt(p['all_power_w'], 'W')}",
                  f"- GPU power: {_fmt(p['gpu_power_w'], 'W')}",
                  f"- Energy: {p['energy_j']} J ({p['energy_wh']} Wh); per generated token: {jpt} J",
                  f"- GPU temp: {_fmt(p['gpu_temp_c'], '°C')}; CPU temp: {_fmt(p['cpu_temp_c'], '°C')}",
                  f"- Fan peak: {p['fan_peak_pct']}% of max",
                  gpu_line,
                  f"- RAM peak: {p['ram_peak_gb']} GB; swap change: {p['swap_delta_gb']} GB"]
    if s.get("macmon_exited_early"):
        lines.append("- **Warning:** macmon exited before the trial was stopped; "
                     "figures cover only its samples.")
    return "\n".join(lines) + "\n"
