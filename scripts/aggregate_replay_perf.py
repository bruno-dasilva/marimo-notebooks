"""Aggregate per-replay perf summary parquet from extract-perf ndjsons.

Output: data/bar_replays/perf_summary.parquet
Schema: one row per (replay_id, player_num).

Usage:
  uv run scripts/aggregate_replay_perf.py
  uv run scripts/aggregate_replay_perf.py --limit 50    # smoke
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import msgspec
import polars as pl

# msgspec.json.Decoder() is ~5x faster than stdlib json.loads on these ndjsons
# and accepts bytes directly, skipping the utf-8 decode round-trip.
_JSON_DECODER = msgspec.json.Decoder()

# polars asof-join with `by=` warns per call; one suppression at module load
# applies in workers too because each worker re-imports the module.
warnings.filterwarnings(
    "ignore",
    message="Sortedness of columns cannot be checked when 'by' groups provided",
)


def _open_ndjson(path: Path):
    """Return a binary file handle for `.ndjson` or `.ndjson.gz`.

    Binary mode lets msgspec decode bytes directly without an intermediate
    str (~25% extra speedup over text mode).
    """
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _replay_stem(path: Path) -> str:
    """Strip `.ndjson` or `.ndjson.gz` to recover the canonical replay base."""
    name = path.name
    if name.endswith(".ndjson.gz"):
        return name[: -len(".ndjson.gz")]
    if name.endswith(".ndjson"):
        return name[: -len(".ndjson")]
    return path.stem


def _list_perf_files(perf_dir: Path) -> list[Path]:
    """List perf files preferring `.ndjson.gz` when both extensions exist."""
    gz_by_stem = {p.name[: -len(".ndjson.gz")]: p for p in perf_dir.glob("*.ndjson.gz")}
    plain = [
        p
        for p in perf_dir.glob("*.ndjson")
        if p.name[: -len(".ndjson")] not in gz_by_stem
    ]
    return sorted(list(gz_by_stem.values()) + plain, key=lambda p: p.name)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bar_replays"
PERF_DIR = DATA_DIR / "perf"
OUT_PATH = DATA_DIR / "perf_summary.parquet"
MANIFEST_PATH = DATA_DIR / "perf_summary_manifest.json"

# Filename: 2025-12-29_10-50-57-487_<map>_<YYYY.MM.DD>.ndjson
FILENAME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{3})_(.+)_(\d{4}\.\d{2}\.\d{2})$"
)


def parse_filename(stem: str):
    m = FILENAME_RE.match(stem)
    if not m:
        return None, None, None
    y, mo, d, h, mi, s, ms, mp, eng = m.groups()
    dt = datetime(
        int(y), int(mo), int(d), int(h), int(mi), int(s), int(ms) * 1000, tzinfo=timezone.utc
    )
    return dt, mp, eng


_BAR_VERSION_RE = re.compile(r"test-(\d+)-([0-9a-f]+)")


def _parse_bar_version(game: str | None) -> tuple[str | None, int | None]:
    """Extract a compact BAR version + sortable build number from `meta.game`.

    The full string looks like 'Beyond All Reason test-27674-02a81bb'.
    Returns ('test-27674-02a81bb', 27674) on match, (game, None) otherwise
    so we still preserve the raw string for releases that don't follow the
    test-<build>-<sha> convention (e.g. stable releases like 'BAR v1.x').
    """
    if not game:
        return None, None
    m = _BAR_VERSION_RE.search(game)
    if m:
        return f"test-{m.group(1)}-{m.group(2)}", int(m.group(1))
    return game.strip() or None, None


def normalize_cpu(s: str | None) -> str | None:
    """Collapse vendor punctuation/clock-speed noise. Best-effort, regex-only."""
    if not s:
        return None
    out = s
    out = re.sub(r"\((R|TM|tm|r)\)", "", out)
    out = re.sub(r"\bCPU\b", "", out)
    out = re.sub(r"\bProcessor\b", "", out)
    out = re.sub(r"\b\d+-Core\b", "", out)
    out = re.sub(r"@\s*[\d.]+\s*GHz", "", out)
    out = re.sub(r"\s+", " ", out).strip(" -,")
    return out or None


def normalize_gpu(s: str | None) -> str | None:
    if not s:
        return None
    out = re.sub(r"\s+", " ", s).strip()
    return out or None


def weighted_quantile_speed(samples: list[tuple[float, float]], q: float) -> float:
    """Interval-weighted quantile of internalSpeed. samples = [(weight, speed), ...]."""
    if not samples:
        return 1.0
    samples = sorted(samples, key=lambda x: x[1])
    total = sum(w for w, _ in samples)
    if total <= 0:
        return 1.0
    cum = 0.0
    for w, v in samples:
        cum += w
        if cum / total >= q:
            return v
    return samples[-1][1]


def replay_speed_stats(speed_records: list[tuple[float, float]], duration_s: float):
    """Compute slowdown metrics from `speed` ndjson records (sorted by t).

    Each record represents the moment internalSpeed CHANGED to its new value.
    The pre-existing speed before the first record is assumed to be 1.0 from t=0.
    """
    if duration_s <= 0:
        return {
            "pct_time_slowed": 0.0,
            "min_internal_speed": 1.0,
            "internal_speed_p10": 1.0,
            "internal_speed_p25": 1.0,
            "internal_speed_p50": 1.0,
            "max_slowdown_streak_s": 0.0,
            "slowdown_recovery_count": 0,
        }

    speed_records = sorted(speed_records, key=lambda x: x[0])
    # Build (start_t, end_t, speed) intervals.
    intervals: list[tuple[float, float, float]] = []
    cur_t = 0.0
    cur_s = 1.0
    for t, s in speed_records:
        if t > cur_t:
            intervals.append((cur_t, t, cur_s))
        cur_t = t
        cur_s = s
    if cur_t < duration_s:
        intervals.append((cur_t, duration_s, cur_s))

    SLOW_THRESHOLD = 0.999  # below 1.0 modulo float noise
    slowed_total = 0.0
    max_streak = 0.0
    cur_streak = 0.0
    recovery_count = 0
    prev_slow = False
    weighted: list[tuple[float, float]] = []
    min_speed = 1.0
    for start, end, s in intervals:
        dt = end - start
        weighted.append((dt, s))
        if s < min_speed:
            min_speed = s
        slow = s < SLOW_THRESHOLD
        if slow:
            slowed_total += dt
            cur_streak += dt
            if cur_streak > max_streak:
                max_streak = cur_streak
        else:
            if prev_slow:
                recovery_count += 1
            cur_streak = 0.0
        prev_slow = slow

    return {
        "pct_time_slowed": slowed_total / duration_s,
        "min_internal_speed": float(min_speed),
        "internal_speed_p10": float(weighted_quantile_speed(weighted, 0.10)),
        "internal_speed_p25": float(weighted_quantile_speed(weighted, 0.25)),
        "internal_speed_p50": float(weighted_quantile_speed(weighted, 0.50)),
        "max_slowdown_streak_s": float(max_streak),
        "slowdown_recovery_count": recovery_count,
    }


def per_player_saturation_streak(perf_rows_for_player: list[dict]) -> float:
    """Max contiguous-saturation streak in seconds for a single player."""
    if not perf_rows_for_player:
        return 0.0
    rows = sorted(perf_rows_for_player, key=lambda r: r["t"])
    max_streak = 0.0
    streak = 0.0
    last_t = rows[0]["t"]
    last_sat = False
    for r in rows:
        is_sat = r.get("simFrameMsQuality") == "saturated"
        if is_sat and last_sat:
            streak += r["t"] - last_t
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0.0
        last_sat = is_sat
        last_t = r["t"]
    return max_streak


def aggregate_one(fp: Path) -> list[dict] | None:
    """Aggregate one ndjson into a list of per-(replay,player) dicts."""
    perf_rows: list[dict] = []
    fps_rows: list[dict] = []
    speed_rows: list[dict] = []
    user_speed_rows: list[dict] = []
    player_stats_rows: list[dict] = []
    hardware_rows: dict[int, dict] = {}
    meta: dict | None = None

    with _open_ndjson(fp) as fh:
        for line in fh:
            try:
                r = _JSON_DECODER.decode(line)
            except Exception:
                continue
            k = r.get("kind")
            if k == "perf":
                perf_rows.append(r)
            elif k == "fps":
                fps_rows.append(r)
            elif k == "speed":
                speed_rows.append(r)
            elif k == "userSpeed":
                user_speed_rows.append(r)
            elif k == "playerStats":
                player_stats_rows.append(r)
            elif k == "hardware":
                pn = r.get("playerNum")
                if pn is not None and pn not in hardware_rows:
                    hardware_rows[pn] = r
            elif k == "meta":
                meta = r  # keep last

    if not meta or not perf_rows:
        return None

    replay_start_time, _, engine_version = parse_filename(_replay_stem(fp))
    duration_s = meta.get("durationMs", 0) / 1000.0
    if duration_s <= 0:
        return None
    late_threshold = duration_s * (2 / 3)

    map_name = meta.get("map")
    game_id = meta.get("gameId")
    bar_version, bar_build = _parse_bar_version(meta.get("game"))
    spectator_set = {p.get("playerId") for p in meta.get("spectators", [])}
    player_meta_by_num = {p["playerId"]: p for p in meta.get("players", [])}

    # Replay-level slowdown stats (server-side, identical across players).
    speed_pairs = [(r["t"], r["internalSpeed"]) for r in speed_rows]
    slow_stats = replay_speed_stats(speed_pairs, duration_s)

    # Per-replay user speed votes (kept replay-level).
    n_user_speed_changes = len(user_speed_rows)
    min_user_speed = (
        min((r.get("userSpeed", 1.0) for r in user_speed_rows), default=1.0)
    )

    # ---- Per-player aggregation via polars ----
    # sim_ms percentiles include `exact` AND `approx` quality. Per the parser
    # README the approx samples are valid as a band ([cpuUsage − 2/fps,
    # cpuUsage] × 33.33 / internalSpeed); for distributional metrics that's
    # close enough. Saturated/invalid are still excluded (their simFrameMs is
    # null and would corrupt percentiles).
    #
    # Force float dtypes for fields that may appear as integer 0 in early
    # warmup rows. Without this polars infers i64 from the first 100 rows
    # and silently truncates later float values to 0.
    perf_df = pl.DataFrame(
        perf_rows,
        schema_overrides={
            "cpuUsage": pl.Float64,
            "ping": pl.Float64,
            "internalSpeed": pl.Float64,
            "userSpeed": pl.Float64,
            "simFrameMs": pl.Float64,
            "t": pl.Float64,
        },
    )
    perf_valid = perf_df.filter(
        pl.col("simFrameMsQuality").is_in(["exact", "approx"])
    )
    perf_late_valid = perf_valid.filter(pl.col("t") >= late_threshold)

    # Sanity bounds: the upstream parser occasionally emits corrupt floats
    # (e.g. fps of 3e+307, ping of 5e6 ms). Drop those before percentile/mean
    # aggregations so a single junk sample can't blow up a player's stats.
    sane_cpu = (pl.col("cpuUsage") >= 0) & (pl.col("cpuUsage") <= 2.0)
    sane_ping = (pl.col("ping") >= 0) & (pl.col("ping") < 300_000)
    sane_sim = (pl.col("simFrameMs") >= 0) & (pl.col("simFrameMs") < 60_000)

    perf_all_agg = perf_df.group_by("playerNum").agg(
        n_perf_samples=pl.len(),
        n_saturated=(pl.col("simFrameMsQuality") == "saturated").sum(),
        n_invalid=(pl.col("simFrameMsQuality") == "invalid").sum(),
        n_approx=(pl.col("simFrameMsQuality") == "approx").sum(),
        cpu_usage_p50=pl.col("cpuUsage").filter(sane_cpu).quantile(0.5),
        cpu_usage_p95=pl.col("cpuUsage").filter(sane_cpu).quantile(0.95),
        ping_p50=pl.col("ping").filter(sane_ping).quantile(0.5),
        ping_p95=pl.col("ping").filter(sane_ping).quantile(0.95),
    )
    perf_valid_agg = perf_valid.group_by("playerNum").agg(
        n_valid=pl.len(),
        sim_ms_p50=pl.col("simFrameMs").filter(sane_sim).quantile(0.5),
        sim_ms_p95=pl.col("simFrameMs").filter(sane_sim).quantile(0.95),
        sim_ms_p99=pl.col("simFrameMs").filter(sane_sim).quantile(0.99),
        sim_ms_mean=pl.col("simFrameMs").filter(sane_sim).mean(),
    )
    perf_late_agg = perf_late_valid.group_by("playerNum").agg(
        n_valid_late=pl.len(),
        sim_ms_p50_late=pl.col("simFrameMs").filter(sane_sim).quantile(0.5),
        sim_ms_p95_late=pl.col("simFrameMs").filter(sane_sim).quantile(0.95),
    )

    if fps_rows:
        fps_df = pl.DataFrame(
            fps_rows,
            schema_overrides={"fps": pl.Float64, "t": pl.Float64},
        ).filter(
            # Same defense as above: parser sometimes emits fps=3e+200..3e+307.
            # Real fps tops out in the low thousands even on high-refresh
            # monitors; anything above 2000 is junk.
            (pl.col("fps") >= 0) & (pl.col("fps") < 2000)
        )
        fps_agg = fps_df.group_by("playerNum").agg(
            fps_p5=pl.col("fps").quantile(0.05),
            fps_p50=pl.col("fps").quantile(0.5),
            fps_mean=pl.col("fps").mean(),
            n_fps_floor=(pl.col("fps") <= 2).sum(),
        )
    else:
        fps_df = None
        fps_agg = pl.DataFrame(
            schema={
                "playerNum": pl.Int64,
                "fps_p5": pl.Float64,
                "fps_p50": pl.Float64,
                "fps_mean": pl.Float64,
                "n_fps_floor": pl.UInt32,
            }
        )

    # ---- Derived draw frame time ----
    # Same accounting as sim_ms: in the `exact` quality regime, wire cpuUsage
    # equals simProcUsage (no draw blend), so the leftover wall fraction
    # `1 - cpuUsage` is an UPPER BOUND on drawProcUsage (some of it may be
    # idle/sleep/other). Per-draw-frame ms is then drawProcUsage × 1000 / fps.
    # Treat this as a proxy: useful for relative comparisons across engines
    # for the same hardware, not as absolute draw cost.
    draw_schema = {
        "playerNum": pl.Int64,
        "n_draw_samples": pl.UInt32,
        "draw_ms_p50": pl.Float64,
        "draw_ms_p95": pl.Float64,
        "draw_ms_mean": pl.Float64,
    }
    draw_late_schema = {
        "playerNum": pl.Int64,
        "draw_ms_p50_late": pl.Float64,
        "draw_ms_p95_late": pl.Float64,
    }
    if fps_df is not None:
        # Asof-join nearest fps to each perf sample (per playerNum).
        perf_for_join = perf_df.select(
            ["t", "playerNum", "cpuUsage", "simFrameMsQuality"]
        ).sort("t")
        fps_for_join = fps_df.select(["t", "playerNum", "fps"]).sort("t")
        joined = perf_for_join.join_asof(
            fps_for_join,
            on="t",
            by="playerNum",
            strategy="nearest",
            tolerance=4.0,  # samples are ~2s; allow up to 4s mismatch
        ).filter(
            (pl.col("simFrameMsQuality") == "exact")
            & (pl.col("cpuUsage") > 0)
            & (pl.col("fps").is_not_null())
            & (pl.col("fps") > 0)
        ).with_columns(
            draw_ms=(1.0 - pl.col("cpuUsage")) * 1000.0 / pl.col("fps"),
        )
        if joined.height:
            draw_agg = joined.group_by("playerNum").agg(
                n_draw_samples=pl.len(),
                draw_ms_p50=pl.col("draw_ms").quantile(0.5),
                draw_ms_p95=pl.col("draw_ms").quantile(0.95),
                draw_ms_mean=pl.col("draw_ms").mean(),
            )
            draw_late_agg = (
                joined.filter(pl.col("t") >= late_threshold)
                .group_by("playerNum")
                .agg(
                    draw_ms_p50_late=pl.col("draw_ms").quantile(0.5),
                    draw_ms_p95_late=pl.col("draw_ms").quantile(0.95),
                )
            )
        else:
            draw_agg = pl.DataFrame(schema=draw_schema)
            draw_late_agg = pl.DataFrame(schema=draw_late_schema)
    else:
        draw_agg = pl.DataFrame(schema=draw_schema)
        draw_late_agg = pl.DataFrame(schema=draw_late_schema)

    out = (
        perf_all_agg.join(perf_valid_agg, on="playerNum", how="left")
        .join(perf_late_agg, on="playerNum", how="left")
        .join(fps_agg, on="playerNum", how="left")
        .join(draw_agg, on="playerNum", how="left")
        .join(draw_late_agg, on="playerNum", how="left")
    )

    # Per-player sat streak (Python — needs ordering).
    perf_by_player: dict[int, list[dict]] = {}
    for r in perf_rows:
        perf_by_player.setdefault(r["playerNum"], []).append(r)
    sat_streak = {
        pn: per_player_saturation_streak(rows) for pn, rows in perf_by_player.items()
    }

    rows_out: list[dict] = []
    for row in out.iter_rows(named=True):
        pn = row["playerNum"]
        pmeta = player_meta_by_num.get(pn, {})
        hw = hardware_rows.get(pn, {})
        cpu_raw = hw.get("cpu")
        gpu_raw = hw.get("gpu")
        rows_out.append(
            {
                # Identity
                "replay_id": _replay_stem(fp),
                "game_id": game_id,
                "player_num": pn,
                "player_name": pmeta.get("name"),
                "team_id": pmeta.get("teamId"),
                "ally_team_id": pmeta.get("allyTeamId"),
                "is_spectator": pn in spectator_set,
                # Match-frame
                "replay_start_time": replay_start_time,
                "engine_version": engine_version,
                "bar_version": bar_version,
                "bar_build": bar_build,
                "map": map_name,
                "duration_s": duration_s,
                # Per-player perf (full game)
                "n_perf_samples": row["n_perf_samples"],
                "n_valid": row.get("n_valid"),
                "n_approx_quality": row.get("n_approx"),
                "n_saturated": row.get("n_saturated"),
                "n_invalid": row.get("n_invalid"),
                "max_saturated_streak_s": float(sat_streak.get(pn, 0.0)),
                "cpu_usage_p50": row.get("cpu_usage_p50"),
                "cpu_usage_p95": row.get("cpu_usage_p95"),
                "ping_p50": row.get("ping_p50"),
                "ping_p95": row.get("ping_p95"),
                "sim_ms_p50": row.get("sim_ms_p50"),
                "sim_ms_p95": row.get("sim_ms_p95"),
                "sim_ms_p99": row.get("sim_ms_p99"),
                "sim_ms_mean": row.get("sim_ms_mean"),
                # Per-player perf (late game = last third)
                "n_exact_late": row.get("n_exact_late"),
                "sim_ms_p50_late": row.get("sim_ms_p50_late"),
                "sim_ms_p95_late": row.get("sim_ms_p95_late"),
                # FPS (per-player, hardware-confounded)
                "fps_p5": row.get("fps_p5"),
                "fps_p50": row.get("fps_p50"),
                "fps_mean": row.get("fps_mean"),
                "n_fps_floor": row.get("n_fps_floor"),
                # Derived draw frame ms — exact-quality only, upper bound.
                "n_draw_samples": row.get("n_draw_samples"),
                "draw_ms_p50": row.get("draw_ms_p50"),
                "draw_ms_p95": row.get("draw_ms_p95"),
                "draw_ms_mean": row.get("draw_ms_mean"),
                "draw_ms_p50_late": row.get("draw_ms_p50_late"),
                "draw_ms_p95_late": row.get("draw_ms_p95_late"),
                # Replay-level slowdown (broadcast to every player row in this replay)
                "pct_time_slowed": slow_stats["pct_time_slowed"],
                "min_internal_speed": slow_stats["min_internal_speed"],
                "internal_speed_p10": slow_stats["internal_speed_p10"],
                "internal_speed_p25": slow_stats["internal_speed_p25"],
                "internal_speed_p50": slow_stats["internal_speed_p50"],
                "max_slowdown_streak_s": slow_stats["max_slowdown_streak_s"],
                "slowdown_recovery_count": slow_stats["slowdown_recovery_count"],
                "n_user_speed_changes": n_user_speed_changes,
                "min_user_speed": min_user_speed,
                # Hardware
                "cpu_raw": cpu_raw,
                "cpu": normalize_cpu(cpu_raw),
                "cpu_cores": hw.get("cpuCores"),
                "logical_cpu_cores": hw.get("logicalCpuCores"),
                "memory": hw.get("memory"),
                "gpu_raw": gpu_raw,
                "gpu": normalize_gpu(gpu_raw),
                "gpu_memory": hw.get("gpuMemory"),
                "os": hw.get("os"),
            }
        )
    return rows_out


def _aggregate_one_safe(fp: Path):
    """Pool worker wrapper: never raises; returns (fp, rows_or_None, err)."""
    try:
        return fp, aggregate_one(fp), None
    except Exception as e:  # noqa: BLE001
        return fp, None, repr(e)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Stop after N replays (smoke).")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes (default: ncpu-1; set 1 to disable pool).",
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
    else:
        # chunksize tuned for ~50ms per task: bigger chunks reduce IPC overhead
        # but hurt load-balance if some replays are huge. 16 is a good middle.
        pool = mp.Pool(args.jobs)
        results = pool.imap_unordered(_aggregate_one_safe, files, chunksize=16)

    try:
        for i, (fp, rows, err) in enumerate(results, 1):
            if err is not None:
                print(f"  [{i}/{len(files)}] ERR {fp.name}: {err}")
                skipped += 1
            elif rows is None:
                skipped += 1
            else:
                all_rows.extend(rows)
            if i % 500 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {len(all_rows):,} rows so far")
    finally:
        if args.jobs > 1:
            pool.close()
            pool.join()

    if not all_rows:
        print("No rows produced — nothing written.")
        return

    # infer_schema_length=None scans all rows: early rows can have null
    # datetimes (pre-engine-version filenames) which lock the column dtype
    # to Null, then a later real datetime fails to append. Same fix as the
    # per-minute aggregator.
    df = pl.from_dicts(all_rows, infer_schema_length=None)
    df.write_parquet(args.out)
    print(
        f"Wrote {df.height:,} rows × {df.width} cols → {args.out} "
        f"({args.out.stat().st_size / 1024 / 1024:.1f} MB)  skipped={skipped}"
    )


if __name__ == "__main__":
    main()
