# /// script
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bar_replays"

    # Strip trailing " v1.2" / " 1.4.4" / " V2" so re-versioned maps collapse together.
    # Bare numeric suffix requires at least one dot (so "Koom Valley 3" survives).
    _MAP_VERSION_RE = re.compile(
        r"\s+(?:v\d+(?:\.\d+)*|\d+(?:\.\d+)+)\s*$", re.IGNORECASE
    )

    def normalize_map(name: str) -> str:
        if name is None:
            return name
        prev = None
        out = name
        while out != prev:
            prev = out
            out = _MAP_VERSION_RE.sub("", out).strip()
        return out

    return DATA_DIR, alt, mo, normalize_map, pl


@app.cell
def _(DATA_DIR, normalize_map, pl):
    matches_raw = pl.read_parquet(
        DATA_DIR / "matches.parquet",
        columns=[
            "match_id",
            "start_time",
            "map",
            "game_type",
            "game_duration",
            "is_ranked",
        ],
    )

    players_per_match = (
        pl.read_parquet(DATA_DIR / "match_players.parquet", columns=["match_id"])
        .group_by("match_id")
        .len()
        .rename({"len": "n_players"})
    )

    matches_8v8 = (
        matches_raw.join(players_per_match, on="match_id", how="left")
        .filter((pl.col("game_type") == "Large Team") & (pl.col("n_players") == 16))
        .with_columns(
            duration_min=pl.col("game_duration") / 60,
            map_base=pl.col("map").map_elements(normalize_map, return_dtype=pl.String),
        )
    )
    matches_8v8.head()
    return (matches_8v8,)


@app.cell
def _(matches_8v8, mo):
    max_ts = matches_8v8["start_time"].max()
    window_dropdown = mo.ui.dropdown(
        options={"30d": 30, "90d": 90, "365d": 365, "All time": None},
        value="365d",
        label="Time window",
    )
    duration_range = mo.ui.range_slider(
        start=0,
        stop=120,
        step=1,
        value=[5, 90],
        label="Duration filter (minutes)",
        full_width=True,
    )
    ranked_only = mo.ui.checkbox(value=True, label="Ranked only")
    mo.hstack([window_dropdown, ranked_only, duration_range], justify="start", gap=2)
    return duration_range, max_ts, ranked_only, window_dropdown


@app.cell
def _(duration_range, matches_8v8, max_ts, pl, ranked_only, window_dropdown):
    days = window_dropdown.value
    cutoff = None if days is None else max_ts - pl.duration(days=days).item()

    m_f = matches_8v8
    if cutoff is not None:
        m_f = m_f.filter(pl.col("start_time") >= cutoff)
    if ranked_only.value:
        m_f = m_f.filter(pl.col("is_ranked"))
    dmin, dmax = duration_range.value
    m_f = m_f.filter(
        (pl.col("duration_min") >= dmin) & (pl.col("duration_min") <= dmax)
    )
    m_f = m_f.with_columns(week=pl.col("start_time").dt.truncate("1w"))
    return (m_f,)


@app.cell
def _(m_f, mo):
    n_matches = m_f.height
    median_duration = m_f["duration_min"].median()
    n_ranked = int(m_f["is_ranked"].sum())
    pct_ranked = (n_ranked / n_matches * 100) if n_matches else 0.0
    n_maps = m_f["map_base"].n_unique()

    headline = mo.hstack(
        [
            mo.stat(label="8v8 matches", value=f"{n_matches:,}"),
            mo.stat(
                label="Median duration",
                value=f"{median_duration:.1f} min" if median_duration is not None else "—",
            ),
            mo.stat(label="Ranked", value=f"{pct_ranked:.1f}%"),
            mo.stat(label="Distinct maps", value=f"{n_maps:,}"),
        ],
        justify="start",
        gap=2,
    )
    headline
    return


@app.cell
def _(alt, m_f, pl):
    daily_counts = (
        m_f.with_columns(day=pl.col("start_time").dt.truncate("1d"))
        .group_by("day")
        .agg(matches=pl.len())
        .sort("day")
    )
    matches_per_day = (
        alt.Chart(daily_counts)
        .mark_area(opacity=0.6, line=True)
        .encode(
            x=alt.X("day:T", title="Day"),
            y=alt.Y("matches:Q", title="8v8 matches per day"),
            tooltip=[alt.Tooltip("day:T"), alt.Tooltip("matches:Q", format=",")],
        )
        .properties(height=260, width="container", title="8v8 matches per day")
    )
    matches_per_day
    return


@app.cell
def _(alt, m_f, pl):
    duration_hist_data = (
        m_f.with_columns(bin=(pl.col("duration_min") // 1 * 1).cast(pl.Int64))
        .group_by("bin")
        .agg(matches=pl.len())
        .sort("bin")
    )
    duration_hist = (
        alt.Chart(duration_hist_data)
        .mark_bar()
        .encode(
            x=alt.X("bin:Q", title="Duration (min, 1-min bins)"),
            x2="bin_end:Q",
            y=alt.Y("matches:Q", title="Matches"),
            tooltip=[
                alt.Tooltip("bin:Q", title="≥ min"),
                alt.Tooltip("matches:Q", format=","),
            ],
        )
        .transform_calculate(bin_end="datum.bin + 1")
        .properties(height=240, width="container", title="Duration distribution")
    )
    duration_hist
    return


@app.cell
def _(alt, m_f, pl):
    weekly_dur = (
        m_f.group_by("week")
        .agg(
            p25=pl.col("duration_min").quantile(0.25),
            p50=pl.col("duration_min").quantile(0.50),
            p75=pl.col("duration_min").quantile(0.75),
            matches=pl.len(),
        )
        .filter(pl.col("matches") >= 50)
        .sort("week")
    )
    band = (
        alt.Chart(weekly_dur)
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X("week:T", title="Week"),
            y=alt.Y("p25:Q", title="Duration (min)"),
            y2="p75:Q",
        )
    )
    line = (
        alt.Chart(weekly_dur)
        .mark_line()
        .encode(
            x="week:T",
            y=alt.Y("p50:Q", title="Median duration (min)"),
            tooltip=[
                alt.Tooltip("week:T"),
                alt.Tooltip("p50:Q", format=".1f", title="Median min"),
                alt.Tooltip("p25:Q", format=".1f", title="p25"),
                alt.Tooltip("p75:Q", format=".1f", title="p75"),
                alt.Tooltip("matches:Q", format=",", title="Matches"),
            ],
        )
    )
    duration_over_time = (band + line).properties(
        height=280, width="container", title="Match duration per week (median, p25-p75 band)"
    )
    duration_over_time
    return


@app.cell
def _(alt, m_f, pl):
    top_maps = (
        m_f.group_by("map_base")
        .agg(
            matches=pl.len(),
            median_min=pl.col("duration_min").median(),
        )
        .sort("matches", descending=True)
        .head(15)
    )
    map_bar = (
        alt.Chart(top_maps)
        .mark_bar()
        .encode(
            x=alt.X("matches:Q", title="Matches"),
            y=alt.Y("map_base:N", sort="-x", title="Map"),
            color=alt.Color(
                "median_min:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title="Median min"),
            ),
            tooltip=[
                alt.Tooltip("map_base:N", title="Map"),
                alt.Tooltip("matches:Q", format=","),
                alt.Tooltip("median_min:Q", format=".1f", title="Median min"),
            ],
        )
        .properties(height=380, width="container", title="Top 15 maps (8v8)")
    )
    map_bar
    return


if __name__ == "__main__":
    app.run()
