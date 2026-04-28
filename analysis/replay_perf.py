# /// script
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bar_replays"
    PERF_DIR = DATA_DIR / "perf"
    return DATA_DIR, PERF_DIR, alt, json, mo, pl


@app.cell
def _(PERF_DIR):
    # Lazy: just enumerate filenames, don't read content. With hundreds of
    # 10+ MB ndjsons this matters — full content is loaded only for the
    # currently selected replay below.
    replay_files = sorted(PERF_DIR.glob("*.ndjson"))
    return (replay_files,)


@app.cell
def _(mo, replay_files):
    replay_options = {p.stem: i for i, p in enumerate(replay_files)}
    replay_dropdown = mo.ui.dropdown(
        options=replay_options,
        value=next(iter(replay_options)) if replay_options else None,
        label=f"Replay ({len(replay_files)} available)",
        searchable=True,
    )
    replay_dropdown
    return (replay_dropdown,)


@app.cell
def _(json, pl, replay_dropdown, replay_files):
    # Read just the selected file. extract-perf appends a `meta` row at the
    # end of each replay block; if a single file has multiple meta rows
    # (e.g. concatenated batch outputs), we take the LAST segment.
    _path = replay_files[replay_dropdown.value]
    _all_rows = [json.loads(l) for l in _path.read_text().splitlines() if l]
    _meta_indices = [i for i, r in enumerate(_all_rows) if r.get("kind") == "meta"]
    if len(_meta_indices) > 1:
        _start = _meta_indices[-2] + 1  # start of last segment
        rows = _all_rows[_start:]
    else:
        rows = _all_rows
    meta = next(r for r in rows if r["kind"] == "meta")
    selected_game_id = meta["gameId"]

    def _table(kind):
        chunk = [r for r in rows if r.get("kind") == kind]
        return pl.DataFrame(chunk) if chunk else pl.DataFrame()

    perf_df = _table("perf")
    fps_df = _table("fps")
    speed_df = _table("speed")
    user_speed_df = _table("userSpeed")

    players_df = pl.DataFrame(meta["players"]).rename(
        {"playerId": "playerNum"}
    ).with_columns(pl.col("playerNum").cast(pl.Int64))

    # `t` is in seconds for live signals (perf/fps/speed). Confirmed against
    # meta.durationMs ≈ max(t)*1000.
    duration_s = meta["durationMs"] / 1000.0
    return (
        duration_s,
        fps_df,
        meta,
        perf_df,
        players_df,
        selected_game_id,
        speed_df,
        user_speed_df,
    )


@app.cell
def _(DATA_DIR, pl, selected_game_id):
    matches = pl.read_parquet(
        DATA_DIR / "matches.parquet",
        columns=[
            "match_id",
            "replay_id",
            "start_time",
            "map",
            "game_duration",
            "is_ranked",
            "winning_team",
        ],
    ).filter(pl.col("replay_id") == selected_game_id)

    if matches.height:
        match_row = matches.row(0, named=True)
        match_id = match_row["match_id"]
        match_players = (
            pl.read_parquet(
                DATA_DIR / "match_players.parquet",
                columns=[
                    "match_id",
                    "user_id",
                    "team_id",
                    "old_skill",
                    "new_skill",
                    "faction",
                    "left_after",
                ],
            )
            .filter(pl.col("match_id") == match_id)
        )
        players_meta = pl.read_parquet(
            DATA_DIR / "players.parquet",
            columns=["user_id", "name", "country"],
        ).join(match_players, on="user_id", how="inner")
    else:
        match_row = None
        players_meta = pl.DataFrame()
    return match_row, players_meta


@app.cell
def _(duration_s, match_row, meta, mo):
    def _fmt_dur(s):
        s = int(round(s))
        return f"{s // 60}:{s % 60:02d}"

    map_name = (match_row or meta).get("map") if match_row else meta.get("map")
    cards = [
        mo.stat(label="Map", value=map_name or "—"),
        mo.stat(label="Duration", value=_fmt_dur(duration_s)),
        mo.stat(label="Engine", value=meta.get("engine", "—")),
        mo.stat(label="Players", value=str(len(meta.get("players", [])))),
    ]
    if match_row is not None:
        cards.extend(
            [
                mo.stat(
                    label="Ranked",
                    value="yes" if match_row["is_ranked"] else "no",
                ),
                mo.stat(
                    label="Winning ally",
                    value=str(match_row["winning_team"]),
                ),
                mo.stat(
                    label="Started",
                    value=match_row["start_time"].strftime("%Y-%m-%d %H:%M"),
                ),
            ]
        )
    mo.hstack(cards, justify="start", gap=2)
    return


@app.cell
def _(mo, players_df):
    quality_filter = mo.ui.multiselect(
        options=["exact", "approx", "saturated", "invalid"],
        value=["exact", "approx"],
        label="simFrameMs quality",
    )
    smoothing = mo.ui.slider(
        start=0,
        stop=60,
        step=2,
        value=10,
        label="Rolling window (s)",
        show_value=True,
    )
    name_to_pid = {
        row["name"]: int(row["playerNum"])
        for row in players_df.sort("name").iter_rows(named=True)
    }
    player_selector = mo.ui.multiselect(
        options=name_to_pid,
        value=list(name_to_pid.keys()),
        label="Players",
    )
    mo.vstack(
        [
            mo.hstack([quality_filter, smoothing], justify="start", gap=2),
            player_selector,
        ]
    )
    return player_selector, quality_filter, smoothing


@app.cell
def _(players_df):
    # Stable color per player across all charts.
    _palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    ]
    _names = sorted(players_df["name"].to_list())
    player_colors = {n: _palette[i % len(_palette)] for i, n in enumerate(_names)}
    return (player_colors,)


@app.cell
def _(alt, duration_s, fps_df, perf_df, pl, speed_df, user_speed_df):
    # --- Game Health tab ---------------------------------------------------
    def _speed_timeline(df, col):
        if df.is_empty():
            return pl.DataFrame({"t": [0.0, duration_s], col: [1.0, 1.0]})
        return (
            df.select(["t", col])
            .sort("t")
            # Anchor a t=0 sample at 1.0 so the line starts at full speed.
            .vstack(pl.DataFrame({"t": [0.0], col: [1.0]}))
            .sort("t")
            .vstack(pl.DataFrame({"t": [duration_s], col: [None]}))
            .with_columns(pl.col(col).fill_null(strategy="forward"))
        )

    internal = _speed_timeline(speed_df, "internalSpeed").with_columns(
        signal=pl.lit("internalSpeed")
    ).rename({"internalSpeed": "value"})
    user = (
        user_speed_df.group_by("t").agg(pl.col("userSpeed").median())
        if not user_speed_df.is_empty()
        else pl.DataFrame({"t": [], "userSpeed": []}, schema={"t": pl.Float64, "userSpeed": pl.Float64})
    )
    user_tl = _speed_timeline(user, "userSpeed").with_columns(
        signal=pl.lit("userSpeed")
    ).rename({"userSpeed": "value"})

    speed_long = pl.concat([internal, user_tl], how="vertical")

    health_lines = (
        alt.Chart(speed_long)
        .mark_line(interpolate="step-after", strokeWidth=1.6)
        .encode(
            x=alt.X("t:Q", title="Game time (s)"),
            y=alt.Y(
                "value:Q",
                title="Speed multiplier",
                scale=alt.Scale(domain=[0, 1.05]),
            ),
            color=alt.Color(
                "signal:N",
                title=None,
                scale=alt.Scale(
                    domain=["internalSpeed", "userSpeed"],
                    range=["#1f77b4", "#999999"],
                ),
                legend=alt.Legend(orient="right", labelLimit=140),
            ),
            tooltip=[
                alt.Tooltip("t:Q", format=".1f"),
                alt.Tooltip("signal:N"),
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
    )

    # Cliff markers: 5 s buckets that contain >=1 saturated perf sample or
    # >=1 fps≤2 sample. These flag client-side "past the cliff" windows;
    # see ~/www/demo-parser/src/bin/README.md.
    cliff_rows = []
    if not perf_df.is_empty() and "simFrameMsQuality" in perf_df.columns:
        sat = perf_df.filter(pl.col("simFrameMsQuality") == "saturated").select(
            t=(pl.col("t") // 5 * 5)
        ).with_columns(kind=pl.lit("saturated cpu"))
        cliff_rows.append(sat)
    if not fps_df.is_empty():
        floor = fps_df.filter(pl.col("fps") <= 2).select(
            t=(pl.col("t") // 5 * 5)
        ).with_columns(kind=pl.lit("fps ≤ 2"))
        cliff_rows.append(floor)
    cliffs = pl.concat(cliff_rows).unique() if cliff_rows else pl.DataFrame()

    if not cliffs.is_empty():
        cliff_marks = (
            alt.Chart(cliffs)
            .mark_rule(opacity=0.25, strokeWidth=2)
            .encode(
                x="t:Q",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(
                        domain=["saturated cpu", "fps ≤ 2"],
                        range=["#d62728", "#ff7f0e"],
                    ),
                    legend=alt.Legend(orient="right", labelLimit=140),
                ),
                tooltip=[alt.Tooltip("t:Q", format=".0f"), alt.Tooltip("kind:N")],
            )
        )
        health_chart = (cliff_marks + health_lines).properties(
            height=300,
            width="container",
            padding={"left": 5, "top": 5, "bottom": 5, "right": 73},
            title="Server speed & client cliff markers",
        )
    else:
        health_chart = health_lines.properties(
            height=300,
            width="container",
            padding={"left": 5, "top": 5, "bottom": 5, "right": 50},
            title="Server speed (no cliffs detected)",
        )
    return (health_chart,)


@app.cell
def _(
    alt,
    fps_df,
    mo,
    perf_df,
    pl,
    player_colors,
    player_selector,
    players_df,
    quality_filter,
    smoothing,
):
    # --- Per-player time series tab ---------------------------------------
    selected_pids = list(player_selector.value)
    _color_scale = alt.Scale(
        domain=list(player_colors.keys()),
        range=list(player_colors.values()),
    )

    def _smooth(df, value_col):
        # Bin to N-second buckets, average per player. Avoids painting
        # 2 s noise across the full game timeline.
        # Sort by (player, t) — Altair's mark_line connects points in
        # dataframe order; group_by output is unordered, so without this
        # the per-player lines zigzag.
        win = max(2, int(smoothing.value) or 2)
        if df.is_empty():
            return df
        return (
            df.with_columns(t_bin=(pl.col("t") // win * win))
            .group_by(["t_bin", "playerNum"])
            .agg(pl.col(value_col).mean())
            .rename({"t_bin": "t"})
            .sort(["playerNum", "t"])
        )

    perf_clean = perf_df.filter(
        pl.col("simFrameMsQuality").is_in(quality_filter.value)
        & pl.col("simFrameMs").is_not_null()
        & pl.col("playerNum").is_in(selected_pids)
    )
    fps_clean = fps_df.filter(pl.col("playerNum").is_in(selected_pids))

    perf_plot = (
        _smooth(perf_clean, "simFrameMs")
        .join(players_df.select(["playerNum", "name"]), on="playerNum")
    ) if not perf_clean.is_empty() else pl.DataFrame()
    fps_plot = (
        _smooth(fps_clean, "fps")
        .join(players_df.select(["playerNum", "name"]), on="playerNum")
    ) if not fps_clean.is_empty() else pl.DataFrame()

    sim_chart = (
        alt.Chart(perf_plot)
        .mark_line(opacity=0.75, strokeWidth=1.2)
        .encode(
            x=alt.X("t:Q", title="Game time (s)"),
            y=alt.Y("simFrameMs:Q", title="Sim frame (ms)"),
            color=alt.Color(
                "name:N",
                scale=_color_scale,
                title="Player",
                legend=alt.Legend(orient="right", labelLimit=140),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("t:Q", format=".0f", title="t (s)"),
                alt.Tooltip("simFrameMs:Q", format=".2f"),
            ],
        )
        .properties(
            height=240,
            width="container",
            padding={"left": 5, "top": 5, "bottom": 5, "right": 50},
            title=alt.TitleParams(
                "Sim frame timing per player (approx.)",
                subtitle="Lower is better — ms of CPU spent per sim frame",
                subtitleColor="#888",
            ),
        )
    )

    fps_chart = (
        alt.Chart(fps_plot)
        .mark_line(opacity=0.75, strokeWidth=1.2)
        .encode(
            x=alt.X("t:Q", title="Game time (s)"),
            y=alt.Y("fps:Q", title="Render FPS"),
            color=alt.Color(
                "name:N",
                scale=_color_scale,
                title="Player",
                legend=alt.Legend(orient="right", labelLimit=140),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("t:Q", format=".0f", title="t (s)"),
                alt.Tooltip("fps:Q", format=".0f"),
            ],
        )
        .properties(
            height=240,
            width="container",
            padding={"left": 5, "top": 5, "bottom": 5, "right": 50},
            title=alt.TitleParams(
                "Render FPS per player",
                subtitle="Higher is better — FPS ≤ 2 means the engine hit its draw-floor",
                subtitleColor="#888",
            ),
        )
    )

    # `alt.vconcat` doesn't propagate `width="container"` — render as
    # separate marimo charts stacked, so each fills the page width.
    ts_charts = mo.vstack(
        [mo.ui.altair_chart(sim_chart), mo.ui.altair_chart(fps_chart)]
    )
    return (ts_charts,)


@app.cell
def _(
    alt,
    fps_df,
    mo,
    perf_df,
    pl,
    player_colors,
    players_df,
    players_meta,
    quality_filter,
):
    # --- Whole-game summary tab -------------------------------------------
    perf_for_summary = perf_df.filter(
        pl.col("simFrameMsQuality").is_in(quality_filter.value)
        & pl.col("simFrameMs").is_not_null()
    )

    perf_agg = (
        perf_for_summary.group_by("playerNum").agg(
            mean_sim_ms=pl.col("simFrameMs").mean(),
            p95_sim_ms=pl.col("simFrameMs").quantile(0.95),
            n_perf_samples=pl.len(),
        )
        if not perf_for_summary.is_empty()
        else pl.DataFrame(schema={
            "playerNum": pl.Int64,
            "mean_sim_ms": pl.Float64,
            "p95_sim_ms": pl.Float64,
            "n_perf_samples": pl.UInt32,
        })
    )

    saturated_agg = (
        perf_df.filter(pl.col("simFrameMsQuality") == "saturated")
        .group_by("playerNum")
        .agg(n_saturated=pl.len())
    )

    fps_agg = (
        fps_df.group_by("playerNum").agg(
            mean_fps=pl.col("fps").mean(),
            p5_fps=pl.col("fps").quantile(0.05),
            n_fps_floor=(pl.col("fps") <= 2).sum(),
        )
        if not fps_df.is_empty()
        else pl.DataFrame(schema={
            "playerNum": pl.Int64,
            "mean_fps": pl.Float64,
            "p5_fps": pl.Float64,
            "n_fps_floor": pl.Int64,
        })
    )

    summary = (
        players_df.select(["playerNum", "name", "teamId", "allyTeamId"])
        .join(perf_agg, on="playerNum", how="left")
        .join(saturated_agg, on="playerNum", how="left")
        .join(fps_agg, on="playerNum", how="left")
        .with_columns(pl.col("n_saturated").fill_null(0))
    )

    if not players_meta.is_empty() and "name" in players_meta.columns:
        skill_lookup = players_meta.select(
            ["name", "old_skill", "new_skill", "faction", "country"]
        )
        summary = summary.join(skill_lookup, on="name", how="left")

    summary_sorted = summary.sort("mean_sim_ms", descending=True, nulls_last=True)

    summary_table = mo.ui.table(summary_sorted.to_pandas(), pagination=False)

    _color_scale = alt.Scale(
        domain=list(player_colors.keys()),
        range=list(player_colors.values()),
    )
    summary_bar = (
        alt.Chart(summary_sorted.drop_nulls("mean_sim_ms"))
        .mark_bar(opacity=0.85)
        .encode(
            y=alt.Y("name:N", sort="-x", title="Player"),
            x=alt.X("mean_sim_ms:Q", title="Mean sim frame (ms)"),
            color=alt.Color("name:N", scale=_color_scale, legend=None),
            tooltip=[
                alt.Tooltip("name:N", title="Player"),
                alt.Tooltip("allyTeamId:N", title="Ally team"),
                alt.Tooltip("mean_sim_ms:Q", format=".2f"),
                alt.Tooltip("p95_sim_ms:Q", format=".2f", title="p95 sim ms"),
                alt.Tooltip("mean_fps:Q", format=".1f"),
                alt.Tooltip("p5_fps:Q", format=".1f", title="p5 fps"),
                alt.Tooltip("n_saturated:Q", title="# saturated"),
                alt.Tooltip("n_fps_floor:Q", title="# fps≤2"),
            ],
        )
        .properties(height=24 * max(1, summary_sorted.height), width="container")
    )
    return summary_bar, summary_table


@app.cell
def _(health_chart, mo, summary_bar, summary_table, ts_charts):
    mo.ui.tabs(
        {
            "Time Series": mo.vstack([health_chart, ts_charts]),
            "Summary": mo.vstack([summary_table, summary_bar]),
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
