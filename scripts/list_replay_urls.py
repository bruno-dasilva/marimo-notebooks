"""Dump BAR replay object URLs from the public OVH Swift bucket.

Lists everything under `demos/` from the last N days (default 365) and writes
one absolute URL per line to an output file. Resumable: re-running continues
from the last URL already on disk (Swift `marker` is exclusive, so no dupes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

BUCKET_URL = (
    "https://storage.uk.cloud.ovh.net/v1/AUTH_10286efc0d334efd917d476d7183232e/BAR"
)
USER_AGENT = "marimo-notebooks/replay-lister (+https://github.com/anthropics/claude-code)"


def http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
        return resp.read()


def list_page(
    marker: str, end_marker: str | None, limit: int, timeout: float
) -> list[dict]:
    params = {
        "prefix": "demos/",
        "format": "json",
        "limit": str(limit),
        "marker": marker,
    }
    if end_marker:
        params["end_marker"] = end_marker
    url = f"{BUCKET_URL}/?{urllib.parse.urlencode(params)}"
    raw = http_get(url, timeout=timeout)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected Swift response shape: {type(data).__name__}")
    return data


def url_for(name: str) -> str:
    return f"{BUCKET_URL}/{urllib.parse.quote(name, safe='/')}"


def name_from_url(url: str) -> str:
    prefix = BUCKET_URL + "/"
    if not url.startswith(prefix):
        raise ValueError(f"line does not start with bucket URL: {url!r}")
    return urllib.parse.unquote(url[len(prefix):])


def last_line(path: Path) -> str | None:
    """Return the last non-empty line of `path`, or None if empty/missing."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    # Small files: read all. Replay-url file is plain text and growth is linear,
    # so this is fine; switch to seek-from-end if it ever matters.
    with path.open("rb") as f:
        data = f.read()
    for line in reversed(data.splitlines()):
        s = line.decode("utf-8").strip()
        if s:
            return s
    return None


def initial_marker(out_path: Path, days: int, start_date: str | None) -> str:
    tail = last_line(out_path)
    if tail:
        marker = name_from_url(tail)
        print(f"resume: marker = {marker}")
        return marker
    if start_date:
        cutoff = datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        cutoff = date.today() - timedelta(days=days)
    marker = f"demos/{cutoff:%Y-%m-%d}"
    print(f"start:  marker = {marker}")
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "bar_replays"
        / "replay_urls.txt",
    )
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD; overrides --days")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD; optional upper bound")
    parser.add_argument("--page-size", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--restart", action="store_true", help="ignore existing output file")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.restart and args.out.exists():
        args.out.unlink()

    marker = initial_marker(args.out, args.days, args.start_date)
    end_marker = f"demos/{args.end_date}" if args.end_date else None

    pages = 0
    added = 0
    with args.out.open("a") as f:
        while True:
            try:
                entries = list_page(marker, end_marker, args.page_size, args.timeout)
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"page fetch failed at marker={marker!r}: {e}", file=sys.stderr)
                return 1
            if not entries:
                break
            lines = []
            for entry in entries:
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    continue
                lines.append(url_for(name) + "\n")
            if lines:
                f.write("".join(lines))
                f.flush()
                os.fsync(f.fileno())
            pages += 1
            added += len(lines)
            marker = entries[-1]["name"]
            print(f"page {pages}: +{len(lines)} (total +{added}) last={marker}")

    total = sum(1 for _ in args.out.open()) if args.out.exists() else 0
    print(f"\ndone. pages={pages} added={added} total_lines={total} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
