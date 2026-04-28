"""Download BAR replay (.sdfz) files listed in replay_urls.txt.

Input is the file produced by `scripts/list_replay_urls.py` — one absolute
storage URL per line. Each URL points directly at a `.sdfz` object, so no
API call is needed: we just GET the URL and write the response to
`{out}/{basename}`.

Downloads run concurrently (default 32 workers). Resume is via on-disk
file presence plus an append-only `{out}/_manifest.jsonl`; existing files
and replays already recorded as ok/failed are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "marimo-notebooks/replay-downloader (+https://github.com/anthropics/claude-code)"


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
            name = rec.get("file_name")
            if name:
                out[name] = rec
    return out


class ManifestWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        line = json.dumps(record) + "\n"
        with self._lock:
            with self.path.open("a") as f:
                f.write(line)


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


def file_name_from_url(url: str) -> str:
    path = urllib.parse.urlsplit(url).path
    name = urllib.parse.unquote(path.rsplit("/", 1)[-1])
    if not name:
        raise ValueError(f"cannot extract filename from URL: {url!r}")
    return name


def read_urls(path: Path) -> list[str]:
    urls: list[str] = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s:
                urls.append(s)
    return urls


# Loose map normalization for joining filenames against parquet `map` strings.
# More aggressive than analysis/match_overview.py:normalize_map — also strips
# bare trailing integers (so "Failed Negotiations 1" matches "Failed Negotiations"),
# converts underscores to spaces, and lowercases. We accept the small risk of
# collapsing distinct maps ("Koom Valley 3" → "koom valley") because the URL
# downloader already disambiguates by hour+second timestamp.
_MAP_VERSION_RE = re.compile(
    r"\s+(?:v\d+(?:\.\d+)*|\d+(?:\.\d+)*)\s*$", re.IGNORECASE
)


def normalize_map(name: str) -> str:
    out = name.replace("_", " ").strip()
    prev = None
    while out != prev:
        prev = out
        out = _MAP_VERSION_RE.sub("", out).strip()
    return out.lower()


_FILENAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})-(\d{2})-\d+_(.+?)_[0-9].*\.sdfz$"
)


def parse_filename(name: str) -> tuple[str, int, str] | None:
    """Returns (minute_key, second, map_base) or None if unparseable."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    minute_key, sec_str, map_name = m.groups()
    return minute_key, int(sec_str), normalize_map(map_name)


def build_8v8_index(
    matches_path: Path,
    match_players_path: Path,
) -> dict[tuple[str, str], list[int]]:
    """{(minute_key, map_base): [second, ...]} for every 8v8 (Large Team, 16
    player) match in the parquet. Map names are normalized; minute_key is
    `YYYY-MM-DD_HH-MM` derived from start_time."""
    import polars as pl  # only required when filtering is on

    matches = pl.read_parquet(
        matches_path, columns=["match_id", "start_time", "map", "game_type"]
    )
    players = (
        pl.read_parquet(match_players_path, columns=["match_id"])
        .group_by("match_id")
        .len()
        .rename({"len": "n_players"})
    )
    df = (
        matches.join(players, on="match_id", how="left")
        .filter(
            (pl.col("game_type") == "Large Team") & (pl.col("n_players") == 16)
        )
        .select("start_time", "map")
    )
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ts, map_name in df.iter_rows():
        if map_name is None:
            continue
        minute_key = ts.strftime("%Y-%m-%d_%H-%M")
        index[(minute_key, normalize_map(map_name))].append(ts.second)
    return index


def matches_8v8(
    file_name: str, index: dict[tuple[str, str], list[int]], tol_sec: int
) -> bool:
    parsed = parse_filename(file_name)
    if parsed is None:
        return False
    minute_key, second, map_base = parsed
    secs = index.get((minute_key, map_base))
    if not secs:
        return False
    return any(abs(second - s) <= tol_sec for s in secs)


def download_one(
    url: str,
    out_dir: Path,
    timeout: float,
    manifest: ManifestWriter,
) -> tuple[str, str, int | None, str | None]:
    """Returns (file_name, status, bytes, error)."""
    file_name = file_name_from_url(url)
    dest = out_dir / file_name

    if dest.exists() and dest.stat().st_size > 0:
        return file_name, "skipped_existing", dest.stat().st_size, None

    try:
        n = http_stream(url, dest, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        manifest.append(
            {
                "file_name": file_name,
                "url": url,
                "status": "download_failed",
                "error": str(e),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return file_name, "download_failed", None, str(e)

    manifest.append(
        {
            "file_name": file_name,
            "url": url,
            "status": "ok",
            "bytes": n,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return file_name, "ok", n, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urls",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "bar_replays"
        / "replay_urls.txt",
        help="text file with one storage URL per line",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "bar_replays" / "demos",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap URLs processed")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="skip filenames lexically less than this (e.g. '2025-06-01')",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="sort newest first instead of oldest first",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="skip the 8v8 (Large Team / 16-player) filter; download every URL",
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
    parser.add_argument(
        "--time-tolerance",
        type=int,
        default=30,
        help="seconds of slack when matching filename timestamp to start_time",
    )
    args = parser.parse_args()

    if not args.urls.exists():
        print(f"urls file not found: {args.urls}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "_manifest.jsonl"
    manifest = load_manifest(manifest_path)
    writer = ManifestWriter(manifest_path)

    urls = read_urls(args.urls)
    total_input = len(urls)

    if args.no_filter:
        index = None
    else:
        print(f"loading 8v8 index from {args.matches.name} + {args.match_players.name} ...")
        index = build_8v8_index(args.matches, args.match_players)
        print(f"  {len(index):,} unique (minute, map) buckets")

    decoded: list[tuple[str, str]] = []
    skipped_filter = 0
    for url in urls:
        try:
            file_name = file_name_from_url(url)
        except ValueError as e:
            print(f"skip: {e}", file=sys.stderr)
            continue
        if index is not None and not matches_8v8(file_name, index, args.time_tolerance):
            skipped_filter += 1
            continue
        decoded.append((file_name, url))
    decoded.sort(key=lambda x: x[0], reverse=args.reverse)

    if args.start_from is not None:
        before = len(decoded)
        if args.reverse:
            decoded = [(n, u) for n, u in decoded if n <= args.start_from]
        else:
            decoded = [(n, u) for n, u in decoded if n >= args.start_from]
        skipped_start_from = before - len(decoded)
    else:
        skipped_start_from = 0

    if args.limit is not None:
        decoded = decoded[: args.limit]

    pending: list[str] = []
    skipped: list[tuple[str, str]] = []  # (file_name, status)
    for file_name, url in decoded:
        rec = manifest.get(file_name)
        if rec and rec.get("status") == "ok":
            skipped.append((file_name, "ok"))
            continue
        pending.append(url)

    order = "newest-first" if args.reverse else "oldest-first"
    filter_label = "off" if index is None else f"8v8 (±{args.time_tolerance}s)"
    print(
        f"input: {total_input} urls | filter: {filter_label} | "
        f"filter-skipped: {skipped_filter} | sort: {order} | "
        f"start-from-skipped: {skipped_start_from} | "
        f"manifest-skipped: {len(skipped)} | "
        f"to download: {len(pending)} | workers: {args.workers}"
    )
    for file_name, status in skipped:
        print(f"manifest-skipped ({status}) -> {file_name}")

    if args.dry_run:
        for url in pending[:20]:
            print(f"DRY: {url}")
        if len(pending) > 20:
            print(f"... and {len(pending) - 20} more")
        return 0

    counts = {"ok": 0, "skipped_existing": 0, "download_failed": 0}
    done = 0
    n_pending = len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, url, args.out, args.timeout, writer): url
            for url in pending
        }
        for fut in as_completed(futures):
            url = futures[fut]
            done += 1
            try:
                file_name, status, n, err = fut.result()
            except Exception as e:
                print(f"[{done}/{n_pending}] worker_crashed: {url} -> {e}", file=sys.stderr)
                counts["download_failed"] += 1
                continue
            counts[status] = counts.get(status, 0) + 1
            if status == "ok":
                print(f"[{done}/{n_pending}] ok -> {file_name} ({n / 1024:.1f} KB)")
            elif status == "skipped_existing":
                print(f"[{done}/{n_pending}] skipped_existing -> {file_name}")
            else:
                print(f"[{done}/{n_pending}] {status}: {file_name}: {err}", file=sys.stderr)

    print()
    print("Summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
