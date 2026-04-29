"""Aggregate per-(replay, minute) perf summary parquet from extract-perf ndjsons.

Output: data/bar_replays/perf_per_minute.parquet
Schema: one row per (replay_id, minute). Samples from all players in the replay
are pooled and aggregated together — minute is the unit, not player.

Usage:
  uv run scripts/aggregate_replay_perf_per_minute.py
  uv run scripts/aggregate_replay_perf_per_minute.py --limit 50    # smoke

Implementation notes:
  Uses numpy + msgspec directly (no polars) and bulk-decompresses each gzip
  before decoding. The hot path is ~2.5x faster than the dict-of-dicts +
  polars approach the per-player aggregator uses, which matters at 400k+ files.
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import msgspec
import numpy as np
import polars as pl

from aggregate_replay_perf import (
    _list_perf_files,
    _parse_bar_version,
    _replay_stem,
    parse_filename,
)

_JSON_DECODER = msgspec.json.Decoder()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bar_replays"
PERF_DIR = DATA_DIR / "perf"
OUT_PATH = DATA_DIR / "perf_per_minute.parquet"

# Sanity bounds — match the per-player aggregator. The upstream parser
# occasionally emits junk floats; drop them at ingest so they can't poison
# percentiles.
_FPS_MAX = 2000.0
_PING_MAX = 300_000.0
_SIM_MAX = 60_000.0
_CPU_MAX = 2.0


def _per_minute_speed_stats(
    speed_events: list[tuple[float, float]], duration_s: float
) -> dict[int, tuple[float, float, float]]:
    """Per-minute (time-weighted mean, min speed, time slowed) from speed events.

    `speed_events` is a sorted list of (t, internalSpeed) where each entry
    marks a CHANGE to the new value. Pre-first-event speed is 1.0 from t=0.
    Returns {minute_index: (speed_mean, speed_min, time_slowed_s)}.
    """
    SLOW_THRESHOLD = 0.999  # sub-1.0 modulo float noise
    out: dict[int, list[float]] = {}  # mi -> [sum_ov, sum_ov*sp, min_sp, time_slowed]

    # Build (start, end, speed) intervals covering [0, duration_s].
    cur_t = 0.0
    cur_s = 1.0
    intervals: list[tuple[float, float, float]] = []
    for t, s in speed_events:
        if t > cur_t:
            intervals.append((cur_t, t, cur_s))
        cur_t = t
        cur_s = s
    if cur_t < duration_s:
        intervals.append((cur_t, duration_s, cur_s))

    for start, end, sp in intervals:
        mi_lo = int(start // 60)
        mi_hi = int(max(start, end - 1e-9) // 60)
        for mi in range(mi_lo, mi_hi + 1):
            wstart = mi * 60.0
            wend = wstart + 60.0
            ov_start = max(start, wstart)
            ov_end = min(end, wend)
            ov = ov_end - ov_start
            if ov <= 0:
                continue
            rec = out.get(mi)
            if rec is None:
                rec = [0.0, 0.0, sp, 0.0]
                out[mi] = rec
            rec[0] += ov
            rec[1] += ov * sp
            if sp < rec[2]:
                rec[2] = sp
            if sp < SLOW_THRESHOLD:
                rec[3] += ov

    return {
        mi: (rec[1] / rec[0] if rec[0] > 0 else 1.0, rec[2], rec[3])
        for mi, rec in out.items()
    }


def aggregate_one(fp: Path) -> list[dict] | None:
    with gzip.open(fp, "rb") as fh:
        data = fh.read()

    # Per-minute pools. We accumulate values cross-player within a minute;
    # minute is the unit so player identity is intentionally erased.
    sim_pool: dict[int, list[float]] = defaultdict(list)
    cpu_pool: dict[int, list[float]] = defaultdict(list)
    ping_pool: dict[int, list[float]] = defaultdict(list)
    fps_pool: dict[int, list[float]] = defaultdict(list)
    sat_count: dict[int, int] = defaultdict(int)
    invalid_count: dict[int, int] = defaultdict(int)
    perf_count: dict[int, int] = defaultdict(int)
    players_seen: dict[int, set] = defaultdict(set)
    # Speed events are global (server-side) but the NDJSON may contain
    # duplicates if the recorder emits them per-player. Dedupe by t below.
    speed_events_raw: list[tuple[float, float]] = []
    meta: dict | None = None

    decode = _JSON_DECODER.decode
    for line in data.splitlines():
        if not line:
            continue
        try:
            r = decode(line)
        except Exception:
            continue
        k = r.get("kind")
        if k == "perf":
            t = r.get("t")
            if t is None:
                continue
            mi = int(t // 60)
            perf_count[mi] += 1
            pn = r.get("playerNum")
            if pn is not None:
                players_seen[mi].add(pn)
            q = r.get("simFrameMsQuality")
            if q == "saturated":
                sat_count[mi] += 1
            elif q == "invalid":
                invalid_count[mi] += 1
            elif q == "exact" or q == "approx":
                sim = r.get("simFrameMs")
                if sim is not None and 0 <= sim < _SIM_MAX:
                    sim_pool[mi].append(sim)
            cpu = r.get("cpuUsage")
            if cpu is not None and 0 <= cpu <= _CPU_MAX:
                cpu_pool[mi].append(cpu)
            ping = r.get("ping")
            if ping is not None and 0 <= ping < _PING_MAX:
                ping_pool[mi].append(ping)
        elif k == "fps":
            t = r.get("t")
            if t is None:
                continue
            f = r.get("fps")
            if f is not None and 0 <= f < _FPS_MAX:
                fps_pool[int(t // 60)].append(f)
        elif k == "speed":
            t = r.get("t")
            s = r.get("internalSpeed")
            if t is not None and s is not None:
                speed_events_raw.append((float(t), float(s)))
        elif k == "meta":
            meta = r

    if not meta or not perf_count:
        return None
    duration_s = meta.get("durationMs", 0) / 1000.0
    if duration_s <= 0:
        return None

    replay_start_time, _, engine_version = parse_filename(_replay_stem(fp))
    map_name = meta.get("map")
    game_id = meta.get("gameId")
    bar_version, bar_build = _parse_bar_version(meta.get("game"))
    replay_id = _replay_stem(fp)

    # Sort + dedupe consecutive identical (t, speed) pairs (multi-player
    # recordings may emit the same global speed change once per player).
    speed_events_raw.sort()
    speed_events: list[tuple[float, float]] = []
    for ev in speed_events_raw:
        if not speed_events or speed_events[-1] != ev:
            speed_events.append(ev)
    speed_minute = _per_minute_speed_stats(speed_events, duration_s)

    minutes = sorted(set(perf_count) | set(fps_pool))
    rows_out: list[dict] = []
    for mi in minutes:
        sim_arr = np.asarray(sim_pool.get(mi, ()), dtype=np.float64)
        cpu_arr = np.asarray(cpu_pool.get(mi, ()), dtype=np.float64)
        ping_arr = np.asarray(ping_pool.get(mi, ()), dtype=np.float64)
        fps_arr = np.asarray(fps_pool.get(mi, ()), dtype=np.float64)

        sim_p50 = sim_p95 = sim_p99 = sim_mean = None
        if sim_arr.size:
            qs = np.quantile(sim_arr, [0.5, 0.95, 0.99])
            sim_p50, sim_p95, sim_p99 = float(qs[0]), float(qs[1]), float(qs[2])
            sim_mean = float(sim_arr.mean())

        cpu_p50 = cpu_p95 = None
        if cpu_arr.size:
            qs = np.quantile(cpu_arr, [0.5, 0.95])
            cpu_p50, cpu_p95 = float(qs[0]), float(qs[1])

        ping_p50 = ping_p95 = None
        if ping_arr.size:
            qs = np.quantile(ping_arr, [0.5, 0.95])
            ping_p50, ping_p95 = float(qs[0]), float(qs[1])

        fps_p5 = fps_p50 = fps_p95 = fps_mean = None
        if fps_arr.size:
            qs = np.quantile(fps_arr, [0.05, 0.5, 0.95])
            fps_p5, fps_p50, fps_p95 = float(qs[0]), float(qs[1]), float(qs[2])
            fps_mean = float(fps_arr.mean())

        # Coarse draw_ms estimate: (1 - mean cpuUsage) * 1000 / median fps.
        # This is per-minute, not per-(minute, player) — the per-player
        # aggregator gives a more precise figure, but for the per-minute
        # over-time view this approximation is plenty.
        draw_ms_est = None
        if cpu_arr.size and fps_p50 and fps_p50 > 0:
            mean_cpu = float(cpu_arr.mean())
            if 0 < mean_cpu < 1:
                draw_ms_est = (1.0 - mean_cpu) * 1000.0 / fps_p50

        # Per-minute speed stats. If the minute is missing from the dict
        # (no overlap with any speed interval — possible for the last
        # partial minute), fall back to "full speed, no slowdown".
        sp_mean, sp_min, sp_slowed_s = speed_minute.get(mi, (1.0, 1.0, 0.0))
        # Cap pct in [0, 1]; the last minute can be partial so the
        # denominator is min(60, remaining duration).
        minute_span = min(60.0, max(0.0, duration_s - mi * 60.0))
        pct_time_slowed_min = (
            sp_slowed_s / minute_span if minute_span > 0 else 0.0
        )

        rows_out.append(
            {
                "replay_id": replay_id,
                "game_id": game_id,
                "replay_start_time": replay_start_time,
                "engine_version": engine_version,
                "bar_version": bar_version,
                "bar_build": bar_build,
                "map": map_name,
                "duration_s": duration_s,
                "minute": mi,
                "n_perf_samples": perf_count[mi],
                "n_valid": sim_arr.size,
                "n_saturated": sat_count.get(mi, 0),
                "n_invalid": invalid_count.get(mi, 0),
                "n_players_seen": len(players_seen.get(mi, ())),
                "n_fps_samples": fps_arr.size,
                "cpu_usage_p50": cpu_p50,
                "cpu_usage_p95": cpu_p95,
                "ping_p50": ping_p50,
                "ping_p95": ping_p95,
                "sim_ms_p50": sim_p50,
                "sim_ms_p95": sim_p95,
                "sim_ms_p99": sim_p99,
                "sim_ms_mean": sim_mean,
                "fps_p5": fps_p5,
                "fps_p50": fps_p50,
                "fps_p95": fps_p95,
                "fps_mean": fps_mean,
                "draw_ms_est": draw_ms_est,
                "speed_mean": sp_mean,
                "speed_min": sp_min,
                "pct_time_slowed_min": pct_time_slowed_min,
            }
        )
    return rows_out


def _aggregate_one_safe(fp: Path):
    try:
        return fp, aggregate_one(fp), None
    except Exception as e:  # noqa: BLE001
        return fp, None, repr(e)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    args = p.parse_args()

    files = _list_perf_files(PERF_DIR)
    if args.limit:
        files = files[: args.limit]
    print(f"Aggregating {len(files)} ndjsons → {args.out}  jobs={args.jobs}")

    all_rows: list[dict] = []
    skipped = 0
    if args.jobs <= 1:
        results = (_aggregate_one_safe(fp) for fp in files)
        pool = None
    else:
        pool = mp.Pool(args.jobs)
        results = pool.imap_unordered(_aggregate_one_safe, files, chunksize=32)

    try:
        for i, (fp, rows, err) in enumerate(results, 1):
            if err is not None:
                print(f"  [{i}/{len(files)}] ERR {fp.name}: {err}")
                skipped += 1
            elif rows is None:
                skipped += 1
            else:
                all_rows.extend(rows)
            if i % 1000 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {len(all_rows):,} rows so far")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if not all_rows:
        print("No rows produced — nothing written.")
        return

    # infer_schema_length=None scans all rows: the first batch may have
    # replay_start_time=None (older replays whose filename doesn't carry the
    # engine date), and polars would otherwise lock the column dtype to Null
    # and fail when a later row produces a real datetime.
    df = pl.from_dicts(all_rows, infer_schema_length=None)
    df.write_parquet(args.out)
    print(
        f"Wrote {df.height:,} rows × {df.width} cols → {args.out} "
        f"({args.out.stat().st_size / 1024 / 1024:.1f} MB)  skipped={skipped}"
    )


if __name__ == "__main__":
    main()
