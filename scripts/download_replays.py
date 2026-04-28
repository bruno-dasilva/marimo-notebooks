"""Download BAR replay (.sdfz) files for a filtered slice of matches.

Pipeline per match:
  1. GET https://api.bar-rts.com/replays/{replay_id}  (rate limited)
  2. Read `fileName` from the JSON response.
  3. GET https://storage.uk.cloud.ovh.net/v1/AUTH_10286efc0d334efd917d476d7183232e/BAR/demos/{fileName}
  4. Save to {out}/{fileName}.

Resume is via {out}/_manifest.jsonl (append-only). Re-running skips replays
already in the manifest and replays whose .sdfz already exists on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

API_URL = "https://api.bar-rts.com/replays/{replay_id}"
STORAGE_URL = (
    "https://storage.uk.cloud.ovh.net/v1/"
    "AUTH_10286efc0d334efd917d476d7183232e/BAR/demos/{file_name}"
)
USER_AGENT = "marimo-notebooks/replay-downloader (+https://github.com/anthropics/claude-code)"


def filter_matches(
    matches_path: Path,
    match_players_path: Path,
    days: int,
    min_duration_min: int,
    limit: int | None,
) -> pl.DataFrame:
    matches = pl.read_parquet(
        matches_path,
        columns=["match_id", "replay_id", "start_time", "game_type", "game_duration"],
    )
    players_per_match = (
        pl.read_parquet(match_players_path, columns=["match_id"])
        .group_by("match_id")
        .len()
        .rename({"len": "n_players"})
    )
    cutoff = matches["start_time"].max() - pl.duration(days=days).item()
    df = (
        matches.join(players_per_match, on="match_id", how="left")
        .filter(
            (pl.col("game_type") == "Large Team")
            & (pl.col("n_players") == 16)
            & (pl.col("game_duration") >= min_duration_min * 60)
            & (pl.col("start_time") >= cutoff)
        )
        .sort("start_time", descending=True)
        .select("match_id", "replay_id", "start_time", "game_duration")
    )
    if limit is not None:
        df = df.head(limit)
    return df


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    out: dict[str, dict] = {}
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("replay_id")
            if rid:
                out[rid] = rec
    return out


def append_manifest(manifest_path: Path, record: dict) -> None:
    with manifest_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self.last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last = time.monotonic()


def http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
        return resp.read()


def http_stream(url: str, dest: Path, timeout: float) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
        bytes_written = 0
        with tmp.open("wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
    tmp.rename(dest)
    return bytes_written


def extract_filename(payload: dict) -> str | None:
    name = payload.get("fileName")
    if isinstance(name, str) and name:
        return name
    for key in ("filename", "file_name", "demoFile"):
        v = payload.get(key)
        if isinstance(v, str) and v:
            return v
    for v in payload.values():
        if isinstance(v, str) and v.endswith(".sdfz"):
            return v
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap matches processed")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--min-duration", type=int, default=75, help="minutes")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "bar_replays" / "demos",
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "bar_replays"
        / "matches.parquet",
    )
    parser.add_argument(
        "--match-players",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "bar_replays"
        / "match_players.parquet",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-rate", type=float, default=1.0, help="min seconds between API calls")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "_manifest.jsonl"

    df = filter_matches(
        args.matches, args.match_players, args.days, args.min_duration, args.limit
    )
    total = df.height
    print(
        f"Filter: Large Team / 16 players / >={args.min_duration} min / last {args.days} d "
        f"=> {total} matches"
    )
    df = df.filter(pl.col("replay_id").is_not_null())
    no_replay_id = total - df.height
    if no_replay_id:
        print(f"  skipping {no_replay_id} match(es) with null replay_id")

    manifest = load_manifest(manifest_path)
    rate = RateLimiter(args.api_rate)
    counts = {
        "ok": 0,
        "skipped_existing": 0,
        "api_failed": 0,
        "download_failed": 0,
        "no_replay_id": no_replay_id,
    }

    for i, row in enumerate(df.iter_rows(named=True), start=1):
        rid = row["replay_id"]
        prefix = f"[{i}/{df.height}] {rid}"

        cached = manifest.get(rid)
        file_name = cached.get("file_name") if cached else None

        if file_name is None:
            api_url = API_URL.format(replay_id=rid)
            if args.dry_run:
                print(f"{prefix} DRY API: {api_url}")
                continue
            try:
                rate.wait()
                payload = json.loads(http_get(api_url, timeout=30))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                print(f"{prefix} api_failed: {e}", file=sys.stderr)
                append_manifest(
                    manifest_path,
                    {
                        "replay_id": rid,
                        "file_name": None,
                        "status": "api_failed",
                        "error": str(e),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                counts["api_failed"] += 1
                continue
            file_name = extract_filename(payload)
            if not file_name:
                print(f"{prefix} api_failed: no fileName in response", file=sys.stderr)
                append_manifest(
                    manifest_path,
                    {
                        "replay_id": rid,
                        "file_name": None,
                        "status": "api_failed",
                        "error": "no fileName in response",
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                counts["api_failed"] += 1
                continue

        dest = args.out / file_name
        storage_url = STORAGE_URL.format(
            file_name=urllib.parse.quote(file_name, safe="")
        )

        if args.dry_run:
            print(f"{prefix} DRY DL: {storage_url}")
            continue

        if dest.exists() and dest.stat().st_size > 0:
            print(f"{prefix} skipped_existing -> {file_name}")
            if rid not in manifest:
                append_manifest(
                    manifest_path,
                    {
                        "replay_id": rid,
                        "file_name": file_name,
                        "status": "ok",
                        "bytes": dest.stat().st_size,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            counts["skipped_existing"] += 1
            continue

        try:
            n = http_stream(storage_url, dest, timeout=60)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"{prefix} download_failed: {e}", file=sys.stderr)
            append_manifest(
                manifest_path,
                {
                    "replay_id": rid,
                    "file_name": file_name,
                    "status": "download_failed",
                    "error": str(e),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            counts["download_failed"] += 1
            continue

        print(f"{prefix} ok -> {file_name} ({n / 1024:.1f} KB)")
        append_manifest(
            manifest_path,
            {
                "replay_id": rid,
                "file_name": file_name,
                "status": "ok",
                "bytes": n,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        counts["ok"] += 1

    print()
    print("Summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
