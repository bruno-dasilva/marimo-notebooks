# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "polars>=1.0",
#     "altair>=5.0",
#     "pandas>=2.0",
#     "pyarrow>=15.0",
# ]
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bar_replays"
    SUMMARY_PATH = DATA_DIR / "perf_summary.parquet"
    PER_MINUTE_PATH = DATA_DIR / "perf_per_minute.parquet"
    return PER_MINUTE_PATH, SUMMARY_PATH, alt, mo, pl


@app.cell
def _(SUMMARY_PATH, mo, pl):
    summary_raw = (
        pl.read_parquet(SUMMARY_PATH)
        .filter(~pl.col("is_spectator"))
        .filter(pl.col("engine_version").is_not_null())
        .filter(pl.col("replay_start_time").is_not_null())
    )
    duration_slider = mo.ui.range_slider(
        start=0,
        stop=180,
        step=5,
        value=[0, 180],
        label="Game length (minutes)",
        show_value=True,
    )
    return duration_slider, summary_raw


@app.cell
def _(duration_slider, mo, pl, summary_raw):
    lo_min, hi_min = duration_slider.value
    summary = summary_raw.filter(
        (pl.col("duration_s") >= lo_min * 60)
        & (pl.col("duration_s") <= hi_min * 60)
    )
    summary_md = mo.vstack(
        [
            duration_slider,
            mo.md(
                f"**{summary.height:,}** player-rows from "
                f"**{summary.select('replay_id').n_unique():,}** replays "
                f"(of {summary_raw.height:,} / "
                f"{summary_raw.select('replay_id').n_unique():,} pre-filter) · "
                f"length **{lo_min}–{hi_min} min** · "
                f"engines: {sorted(summary.select('engine_version').unique().to_series().to_list())}"
            ),
        ]
    )
    summary_md
    return (summary,)


@app.cell
def _(mo, pl, summary):
    MIN_CPU_ROWS = 30
    cpu_stats = (
        summary.filter(pl.col("cpu").is_not_null())
        .group_by("cpu")
        .agg(
            n=pl.len(),
            sim_ms_med=pl.col("sim_ms_p95_late").median(),
        )
        .filter(pl.col("n") >= MIN_CPU_ROWS)
        .filter(pl.col("sim_ms_med").is_not_null())
        .sort("sim_ms_med")
    )
    cpu_options = {"All CPUs": "__all__"} | {
        f"{row['cpu']}  ({row['sim_ms_med']:.1f}ms · n={row['n']})": row["cpu"]
        for row in cpu_stats.iter_rows(named=True)
    }
    cpu_picker = mo.ui.dropdown(
        options=cpu_options,
        value="All CPUs",
        label=f"CPU ({len(cpu_options) - 1} with ≥{MIN_CPU_ROWS} rows, fastest first)",
        searchable=True,
    )
    return (cpu_picker,)


@app.cell
def _(cpu_picker, pl, summary):
    sel_cpu = cpu_picker.value
    if sel_cpu == "__all__":
        cpu_filtered = summary
        cpu_scope = "all CPUs"
    else:
        cpu_filtered = summary.filter(pl.col("cpu") == sel_cpu)
        cpu_scope = sel_cpu
    return cpu_filtered, cpu_scope


@app.cell
def _(
    bar_end_dt,
    bar_start_dt,
    cpu_filtered,
    cpu_scope,
    granularity_picker,
    mo,
    pl,
    series_select,
):
    trunc_str_cpu, bucket_label_cpu = granularity_picker.value
    is_bar_cpu = trunc_str_cpu == "__bar_version__"
    # Apply the shared BAR-version multiselect AND the date-range filter
    # regardless of bucket mode — lets users narrow to a subset of releases
    # and a specific time window.
    _cpu_scoped = cpu_filtered.filter(
        pl.col("bar_version").is_in(list(series_select.value))
        & (pl.col("replay_start_time") >= bar_start_dt)
        & (pl.col("replay_start_time") < bar_end_dt)
    )
    if is_bar_cpu:
        # When bucketing by BAR version, the "bucket" column carries the
        # bar_version string and we keep bar_build alongside for sorting.
        cpu_with_bucket = _cpu_scoped.filter(
            pl.col("bar_version").is_not_null()
        ).with_columns(bucket=pl.col("bar_version"))
    else:
        cpu_with_bucket = _cpu_scoped.with_columns(
            bucket=pl.col("replay_start_time").dt.truncate(trunc_str_cpu),
        )
    cpu_binned = (
        cpu_with_bucket.group_by(["bucket", "engine_version"])
        .agg(
            n_player_rows=pl.len(),
            n_replays=pl.col("replay_id").n_unique(),
            bar_build=pl.col("bar_build").min(),
            sim_ms_p95_late_med=pl.col("sim_ms_p95_late").median(),
            pct_time_slowed_med=pl.col("pct_time_slowed").median(),
            internal_speed_p10_med=pl.col("internal_speed_p10").median(),
            sim_ms_p95_med=pl.col("sim_ms_p95").median(),
        )
        .sort(["engine_version", "bar_build" if is_bar_cpu else "bucket"])
    )
    cpu_scope_md = mo.md(
        f"Scope: **{cpu_scope}** · {cpu_filtered.height:,} player-rows · "
        f"{cpu_filtered.select('replay_id').n_unique():,} replays · "
        f"{cpu_binned.height} ({bucket_label_cpu} × engine) cells"
    )
    return bucket_label_cpu, cpu_binned, cpu_scope_md, is_bar_cpu


@app.cell
def _(
    alt,
    bucket_label_cpu,
    cpu_binned,
    cpu_picker,
    cpu_scope_md,
    date_range_slider,
    granularity_picker,
    is_bar_cpu,
    mo,
    series_picker,
    series_select,
):
    MIN_CELL = 2
    cpu_plot_df = cpu_binned.filter(
        cpu_binned["n_player_rows"] >= MIN_CELL
    ).to_pandas()
    if is_bar_cpu:
        cpu_plot_df = cpu_plot_df.sort_values("bar_build")
    size_range_cpu = (
        [5, 60] if (bucket_label_cpu == "day" or is_bar_cpu) else [20, 200]
    )
    # X axis is temporal for date buckets, nominal (sorted by bar_build)
    # for BAR-version buckets so release order is preserved.
    _x_type = "N" if is_bar_cpu else "T"
    _x_sort = (
        alt.SortField("bar_build", order="ascending") if is_bar_cpu else None
    )
    _x_kwargs = {"sort": _x_sort} if is_bar_cpu else {}

    def _cpu_line(metric: str, title: str, y_format: str = ".2f"):
        line = (
            alt.Chart(cpu_plot_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X(
                    f"bucket:{_x_type}",
                    title=bucket_label_cpu.capitalize(),
                    **_x_kwargs,
                ),
                y=alt.Y(f"{metric}:Q", title=title),
                color=alt.Color("engine_version:N", title="Engine"),
            )
        )
        points = (
            alt.Chart(cpu_plot_df)
            .mark_point(filled=True)
            .encode(
                x=alt.X(f"bucket:{_x_type}", **_x_kwargs),
                y=f"{metric}:Q",
                color=alt.Color("engine_version:N"),
                size=alt.Size(
                    "n_player_rows:Q",
                    title="player-rows",
                    scale=alt.Scale(range=size_range_cpu),
                ),
                tooltip=[
                    alt.Tooltip(f"bucket:{_x_type}", title=bucket_label_cpu),
                    "engine_version",
                    alt.Tooltip("bar_build:Q", title="BAR build"),
                    alt.Tooltip(f"{metric}:Q", format=y_format),
                    "n_player_rows",
                    "n_replays",
                ],
            )
        )
        return (line + points).properties(width=900, height=240, title=title)

    cpu_view = mo.vstack(
        [
            granularity_picker,
            cpu_picker,
            series_picker,
            date_range_slider,
            series_select,
            cpu_scope_md,
            mo.ui.altair_chart(
                _cpu_line("sim_ms_p95_late_med", "sim_ms p95 (late game, ms)")
            ),
            mo.ui.altair_chart(
                _cpu_line("pct_time_slowed_med", "pct of game time slowed", ".2%")
            ),
            mo.ui.altair_chart(
                _cpu_line(
                    "internal_speed_p10_med",
                    "internalSpeed p10 (depth of slowdown)",
                )
            ),
            mo.ui.altair_chart(
                _cpu_line("sim_ms_p95_med", "sim_ms p95 (full game, ms)")
            ),
        ]
    )
    return (cpu_view,)


@app.cell
def _(hw_split_checkbox, pl, summary):
    # Temporal baseline: each entity's baseline = the median of their fps
    # metrics in THEIR FIRST WEEK with data. Every subsequent replay is
    # measured relative to that fixed first-week anchor. Drift over time
    # (across weeks/engines) now flows through the relative metric instead
    # of being absorbed into a global per-entity median.
    MIN_USER_REPLAYS = 3
    # When the hw split is on, a player who upgrades CPU/GPU starts a new
    # baseline for the new rig — relative metrics then track per-rig drift
    # instead of mixing pre/post-upgrade replays into one baseline.
    baseline_key = ["player_name", "cpu", "gpu"] if hw_split_checkbox.value else ["player_name"]
    s_w = summary.with_columns(
        week=pl.col("replay_start_time").dt.truncate("1w")
    ).filter(pl.col("fps_p50").is_not_null())

    # Per-entity first-week + total replay count.
    first_weeks = (
        s_w.group_by(baseline_key)
        .agg(
            first_week=pl.col("week").min(),
            n_user_replays=pl.col("replay_id").n_unique(),
        )
        .filter(pl.col("n_user_replays") >= MIN_USER_REPLAYS)
    )

    # Baseline = median of metrics among the entity's replays in their first week.
    user_baselines = (
        s_w.join(first_weeks, on=baseline_key, how="inner")
        .filter(pl.col("week") == pl.col("first_week"))
        .group_by(baseline_key)
        .agg(
            baseline_fps_p50=pl.col("fps_p50").median(),
            baseline_fps_p5=pl.col("fps_p5").median(),
            baseline_fps_mean=pl.col("fps_mean").median(),
            baseline_draw_ms_p50=pl.col("draw_ms_p50").median(),
            baseline_draw_ms_p95=pl.col("draw_ms_p95").median(),
            baseline_sim_ms_p50=pl.col("sim_ms_p50").median(),
            baseline_sim_ms_p95=pl.col("sim_ms_p95").median(),
            baseline_sim_ms_p95_late=pl.col("sim_ms_p95_late").median(),
            baseline_sim_ms_mean=pl.col("sim_ms_mean").median(),
            first_week=pl.col("first_week").first(),
            n_user_replays=pl.col("n_user_replays").first(),
            n_baseline_replays=pl.len(),
        )
        .filter(pl.col("baseline_fps_p50") > 0)
        .filter(pl.col("baseline_fps_p5") > 0)
        .filter(pl.col("baseline_fps_mean") > 0)
    )

    # Use when().otherwise(None) instead of plain division: a zero baseline
    # produces inf/NaN, which a single user can carry into a per-user cell
    # mean and from there into the weighted-mean across users — leaving the
    # whole bucket blank in the chart. None values are filtered by `_wmean`.
    def _safe_ratio(num, denom):
        return pl.when(pl.col(denom) > 0).then(pl.col(num) / pl.col(denom)).otherwise(None)

    summary_norm = summary.join(
        user_baselines, on=baseline_key, how="inner"
    ).with_columns(
        rel_fps_p5=_safe_ratio("fps_p5", "baseline_fps_p5"),
        rel_fps_p50=_safe_ratio("fps_p50", "baseline_fps_p50"),
        rel_fps_mean=_safe_ratio("fps_mean", "baseline_fps_mean"),
        # rel_draw_ms / rel_sim_ms: < 1.0 = faster than first week (good)
        rel_draw_ms_p50=_safe_ratio("draw_ms_p50", "baseline_draw_ms_p50"),
        rel_draw_ms_p95=_safe_ratio("draw_ms_p95", "baseline_draw_ms_p95"),
        rel_sim_ms_p50=_safe_ratio("sim_ms_p50", "baseline_sim_ms_p50"),
        rel_sim_ms_p95=_safe_ratio("sim_ms_p95", "baseline_sim_ms_p95"),
        rel_sim_ms_p95_late=_safe_ratio("sim_ms_p95_late", "baseline_sim_ms_p95_late"),
        rel_sim_ms_mean=_safe_ratio("sim_ms_mean", "baseline_sim_ms_mean"),
    )
    return MIN_USER_REPLAYS, baseline_key, summary_norm, user_baselines


@app.cell
def _(
    MIN_USER_REPLAYS,
    hw_split_checkbox,
    mo,
    pl,
    summary_norm,
    user_baselines,
):
    n_solo = user_baselines.filter(pl.col("n_baseline_replays") == 1).height
    n_multi = user_baselines.height - n_solo
    entity = "player×hardware combos" if hw_split_checkbox.value else "users"
    gpu_baseline_md = mo.md(
        f"**{user_baselines.height:,}** {entity} with ≥{MIN_USER_REPLAYS} replays · "
        f"**{summary_norm.height:,}** normalized player-rows · "
        f"baseline = median of each entity's `fps_*` in their **first week** with data "
        f"({n_multi:,} had ≥2 replays in their first week, {n_solo:,} had only 1). "
        f"Headline = **rel_fps_p5** (tail frames). `rel_fps_p50` (typical frame) is hardware-bound "
        f"and the least sensitive — kept as a control."
    )
    return (gpu_baseline_md,)


@app.cell
def _(mo, pl, summary_norm):
    MIN_GPU_ROWS = 30
    gpu_stats = (
        summary_norm.filter(pl.col("gpu").is_not_null())
        .group_by("gpu")
        .agg(
            n=pl.len(),
            fps_med=pl.col("fps_p50").median(),
        )
        .filter(pl.col("n") >= MIN_GPU_ROWS)
        .filter(pl.col("fps_med").is_not_null())
        .sort("fps_med", descending=True)
    )
    gpu_options = {"All GPUs": "__all__"} | {
        f"{row['gpu']}  ({row['fps_med']:.0f}fps · n={row['n']})": row["gpu"]
        for row in gpu_stats.iter_rows(named=True)
    }
    gpu_picker = mo.ui.dropdown(
        options=gpu_options,
        value="All GPUs",
        label=f"GPU ({len(gpu_options) - 1} with ≥{MIN_GPU_ROWS} rows, fastest first)",
        searchable=True,
    )
    return (gpu_picker,)


@app.cell
def _(gpu_picker, pl, summary_norm):
    sel_gpu = gpu_picker.value
    if sel_gpu == "__all__":
        gpu_filtered = summary_norm
        gpu_scope = "all GPUs"
    else:
        gpu_filtered = summary_norm.filter(pl.col("gpu") == sel_gpu)
        gpu_scope = sel_gpu
    return gpu_filtered, gpu_scope


@app.cell
def _(
    bar_end_dt,
    bar_start_dt,
    baseline_key,
    gpu_filtered,
    gpu_scope,
    granularity_picker,
    mo,
    pl,
    series_select,
):
    # Two-step weighted aggregation:
    #   1. Per (entity × bucket × engine), take mean of relative metrics —
    #      every entity contributes ONE value per cell regardless of replay
    #      count. "entity" = baseline_key (player_name, optionally + cpu/gpu).
    #   2. Across entities in a cell, take WEIGHTED mean where weight =
    #      entity's total cohort replay count. Regulars dominate, one-offs
    #      have small influence on the cell average.
    trunc_str_gpu, bucket_label_gpu = granularity_picker.value
    is_bar_gpu = trunc_str_gpu == "__bar_version__"
    _gpu_scoped = gpu_filtered.filter(
        pl.col("bar_version").is_in(list(series_select.value))
        & (pl.col("replay_start_time") >= bar_start_dt)
        & (pl.col("replay_start_time") < bar_end_dt)
    )
    if is_bar_gpu:
        gpu_with_bucket = _gpu_scoped.filter(
            pl.col("bar_version").is_not_null()
        ).with_columns(bucket=pl.col("bar_version"))
    else:
        gpu_with_bucket = _gpu_scoped.with_columns(
            bucket=pl.col("replay_start_time").dt.truncate(trunc_str_gpu),
        )
    per_user_cell = gpu_with_bucket.group_by(
        ["bucket", "engine_version", *baseline_key]
    ).agg(
        rel_fps_p5_u=pl.col("rel_fps_p5").mean(),
        rel_fps_mean_u=pl.col("rel_fps_mean").mean(),
        rel_fps_p50_u=pl.col("rel_fps_p50").mean(),
        rel_draw_ms_p50_u=pl.col("rel_draw_ms_p50").mean(),
        rel_draw_ms_p95_u=pl.col("rel_draw_ms_p95").mean(),
        weight=pl.col("n_user_replays").first(),
        n_in_cell=pl.len(),
    )

    def _wmean(col):
        v = pl.col(col)
        w = pl.col("weight")
        valid_w = w.filter(v.is_not_null())
        valid_v = v.filter(v.is_not_null())
        return (valid_v * valid_w).sum() / valid_w.sum()

    gpu_binned = (
        per_user_cell.group_by(["bucket", "engine_version"])
        .agg(
            n_player_rows=pl.col("n_in_cell").sum(),
            n_users=pl.len(),
            total_weight=pl.col("weight").sum(),
            rel_fps_p5_avg=_wmean("rel_fps_p5_u"),
            rel_fps_mean_avg=_wmean("rel_fps_mean_u"),
            rel_fps_p50_avg=_wmean("rel_fps_p50_u"),
            rel_draw_ms_p50_avg=_wmean("rel_draw_ms_p50_u"),
            rel_draw_ms_p95_avg=_wmean("rel_draw_ms_p95_u"),
            rel_fps_p5_med=pl.col("rel_fps_p5_u").median(),
        )
        .sort(["engine_version", "bucket"])
    )
    # Side-aggregation of BAR versions per (bucket × engine_version) cell.
    # When bucket *is* bar_version this just returns 1 per cell, but the
    # min/max/list still make the tooltip self-explanatory.
    bar_per_cell = (
        gpu_with_bucket.filter(pl.col("bar_version").is_not_null())
        .group_by(["bucket", "engine_version"])
        .agg(
            n_bar_versions=pl.col("bar_version").n_unique(),
            bar_build_min=pl.col("bar_build").min(),
            bar_build_max=pl.col("bar_build").max(),
            bar_versions_str=pl.col("bar_version")
            .unique()
            .sort()
            .str.join(", "),
        )
    )
    gpu_binned = gpu_binned.join(
        bar_per_cell, on=["bucket", "engine_version"], how="left"
    )
    gpu_scope_md = mo.md(
        f"GPU scope: **{gpu_scope}** · {gpu_filtered.height:,} player-rows · "
        f"{gpu_filtered.select('player_name').n_unique():,} users · "
        f"{gpu_binned.height} ({bucket_label_gpu} × engine) cells · "
        f"weighted by per-user cohort replay count"
    )
    return bucket_label_gpu, gpu_binned, gpu_scope_md, is_bar_gpu


@app.cell
def _(
    alt,
    bucket_label_gpu,
    date_range_slider,
    gpu_baseline_md,
    gpu_binned,
    gpu_picker,
    gpu_scope_md,
    granularity_picker,
    hw_split_checkbox,
    is_bar_gpu,
    mo,
    pl,
    series_picker,
    series_select,
):
    MIN_GPU_CELL = 3
    gpu_plot_df = gpu_binned.filter(
        gpu_binned["n_player_rows"] >= MIN_GPU_CELL
    ).to_pandas()
    if is_bar_gpu:
        gpu_plot_df = gpu_plot_df.sort_values("bar_build_min")
    size_range_gpu = (
        [5, 60] if (bucket_label_gpu == "day" or is_bar_gpu) else [20, 200]
    )
    _x_type = "N" if is_bar_gpu else "T"
    _x_sort = (
        alt.SortField("bar_build_min", order="ascending")
        if is_bar_gpu
        else None
    )
    _x_kwargs = {"sort": _x_sort} if is_bar_gpu else {}

    def _gpu_chart(metric: str, title: str):
        line = (
            alt.Chart(gpu_plot_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X(
                    f"bucket:{_x_type}",
                    title=bucket_label_gpu.capitalize(),
                    **_x_kwargs,
                ),
                y=alt.Y(
                    f"{metric}:Q",
                    title=title,
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("engine_version:N", title="Engine"),
            )
        )
        points = (
            alt.Chart(gpu_plot_df)
            .mark_point(filled=True)
            .encode(
                x=alt.X(f"bucket:{_x_type}", **_x_kwargs),
                y=f"{metric}:Q",
                color=alt.Color("engine_version:N"),
                size=alt.Size(
                    "n_player_rows:Q",
                    title="player-rows",
                    scale=alt.Scale(range=size_range_gpu),
                ),
                tooltip=[
                    alt.Tooltip(f"bucket:{_x_type}", title=bucket_label_gpu),
                    "engine_version",
                    alt.Tooltip(f"{metric}:Q", format=".3f"),
                    "n_player_rows",
                    "n_users",
                    alt.Tooltip("n_bar_versions:Q", title="BAR versions (n)"),
                    alt.Tooltip("bar_build_min:Q", title="BAR build min"),
                    alt.Tooltip("bar_build_max:Q", title="BAR build max"),
                    alt.Tooltip("bar_versions_str:N", title="BAR versions"),
                ],
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [1.0]}).to_pandas())
            .mark_rule(color="gray", strokeDash=[3, 3])
            .encode(y="y:Q")
        )
        return (line + points + rule).properties(
            width=900, height=240, title=title
        )

    gpu_view = mo.vstack(
        [
            gpu_baseline_md,
            granularity_picker,
            hw_split_checkbox,
            gpu_picker,
            series_picker,
            date_range_slider,
            series_select,
            gpu_scope_md,
            mo.ui.altair_chart(
                _gpu_chart(
                    "rel_fps_p5_avg",
                    "Mean relative fps_p5 (tail-frame FPS, per-user normalized)",
                )
            ),
            mo.ui.altair_chart(
                _gpu_chart(
                    "rel_fps_mean_avg",
                    "Mean relative fps_mean (avg-frame FPS, per-user normalized)",
                )
            ),
            mo.ui.altair_chart(
                _gpu_chart(
                    "rel_fps_p50_avg",
                    "Mean relative fps_p50 (typical-frame FPS — least sensitive)",
                )
            ),
            mo.ui.altair_chart(
                _gpu_chart(
                    "rel_draw_ms_p50_avg",
                    "Mean relative draw_ms_p50 (derived per-frame draw cost — LOWER is better)",
                )
            ),
            mo.ui.altair_chart(
                _gpu_chart(
                    "rel_draw_ms_p95_avg",
                    "Mean relative draw_ms_p95 (tail draw cost — LOWER is better)",
                )
            ),
            mo.ui.altair_chart(
                _gpu_chart(
                    "rel_fps_p5_med",
                    "Median relative fps_p5 (control — collapses to ~1.0 by construction)",
                )
            ),
        ]
    )
    return (gpu_view,)


@app.cell
def _(mo):
    # Sentinel "__bar_version__" tells consumers to bucket on bar_version
    # instead of truncating replay_start_time. The x axis on the charts
    # then becomes a categorical (BAR version), sorted by bar_build so the
    # release order is preserved.
    granularity_picker = mo.ui.dropdown(
        options={
            "Daily": ("1d", "day"),
            "Weekly": ("1w", "week"),
            "BAR version": ("__bar_version__", "BAR version"),
        },
        value="Weekly",
        label="Bucket size",
    )
    return (granularity_picker,)


@app.cell
def _(mo):
    hw_split_checkbox = mo.ui.checkbox(
        value=False,
        label="Count hardware changes as new (key by player + cpu + gpu)",
    )
    return (hw_split_checkbox,)


@app.cell
def _(alt, granularity_picker, hw_split_checkbox, mo, pl, summary):
    trunc_str, bucket_label = granularity_picker.value
    # The "BAR version" granularity isn't meaningful for new-player counts
    # (we'd just be counting first-seen by release, which buries cohorts),
    # so fall back to weekly here when it's selected on the shared picker.
    if trunc_str == "__bar_version__":
        trunc_str, bucket_label = "1w", "week"
    # Identity key: with the checkbox on, a player who upgrades CPU or GPU
    # shows up again as "new" the first time we see the new rig.
    key_cols = ["player_name"]
    entity_label = "players"
    if hw_split_checkbox.value:
        key_cols = ["player_name", "cpu", "gpu"]
        entity_label = "player×hardware combos"

    first_seen = (
        summary.filter(pl.col("player_name").is_not_null())
        .with_columns(bucket=pl.col("replay_start_time").dt.truncate(trunc_str))
        .group_by(key_cols)
        .agg(first_bucket=pl.col("bucket").min())
    )
    earliest_bucket = first_seen.select(pl.col("first_bucket").min()).item()
    # Drop the first bucket — everyone there is "new" by construction and
    # would dwarf the rest of the chart.
    new_per_bucket = (
        first_seen.filter(pl.col("first_bucket") > earliest_bucket)
        .group_by("first_bucket")
        .agg(n_new=pl.len())
        .sort("first_bucket")
        .rename({"first_bucket": "bucket"})
    )

    new_players_df = new_per_bucket.to_pandas()
    new_players_chart = (
        alt.Chart(new_players_df)
        .mark_bar()
        .encode(
            x=alt.X("bucket:T", title=bucket_label.capitalize()),
            y=alt.Y(
                "n_new:Q",
                title=f"New {entity_label} (first seen this {bucket_label})",
            ),
            tooltip=[alt.Tooltip("bucket:T"), alt.Tooltip("n_new:Q", title="new")],
        )
        .properties(
            width=900,
            height=280,
            title=f"New {entity_label} per {bucket_label}",
        )
    )
    new_players_view = mo.vstack(
        [
            granularity_picker,
            hw_split_checkbox,
            mo.md(
                f"**{first_seen.height:,}** unique {entity_label} total across "
                f"**{new_per_bucket.height + 1}** {bucket_label}s. "
                f"First {bucket_label} ({earliest_bucket.date()}) excluded — every "
                f"entity there is new by construction."
            ),
            mo.ui.altair_chart(new_players_chart),
        ]
    )
    return (new_players_view,)


@app.cell
def _(cpu_picker, pl, summary_norm):
    # Reuse the CPU picker from the absolute-sim tab — sim cost is CPU-bound,
    # same scope makes sense.
    sel_cpu_norm = cpu_picker.value
    if sel_cpu_norm == "__all__":
        sim_norm_filtered = summary_norm
        sim_norm_scope = "all CPUs"
    else:
        sim_norm_filtered = summary_norm.filter(pl.col("cpu") == sel_cpu_norm)
        sim_norm_scope = sel_cpu_norm
    return sim_norm_filtered, sim_norm_scope


@app.cell
def _(
    bar_end_dt,
    bar_start_dt,
    baseline_key,
    granularity_picker,
    mo,
    pl,
    series_select,
    sim_norm_filtered,
    sim_norm_scope,
):
    # Same two-step weighted aggregation as the GPU tab — see comment there.
    trunc_str_sim, bucket_label_sim = granularity_picker.value
    is_bar_sim = trunc_str_sim == "__bar_version__"
    _sim_scoped = sim_norm_filtered.filter(
        pl.col("bar_version").is_in(list(series_select.value))
        & (pl.col("replay_start_time") >= bar_start_dt)
        & (pl.col("replay_start_time") < bar_end_dt)
    )
    if is_bar_sim:
        sim_with_bucket = _sim_scoped.filter(
            pl.col("bar_version").is_not_null()
        ).with_columns(bucket=pl.col("bar_version"))
    else:
        sim_with_bucket = _sim_scoped.with_columns(
            bucket=pl.col("replay_start_time").dt.truncate(trunc_str_sim),
        )
    sim_per_user_cell = sim_with_bucket.group_by(
        ["bucket", "engine_version", *baseline_key]
    ).agg(
        rel_sim_ms_p50_u=pl.col("rel_sim_ms_p50").mean(),
        rel_sim_ms_p95_u=pl.col("rel_sim_ms_p95").mean(),
        rel_sim_ms_p95_late_u=pl.col("rel_sim_ms_p95_late").mean(),
        rel_sim_ms_mean_u=pl.col("rel_sim_ms_mean").mean(),
        weight=pl.col("n_user_replays").first(),
        # Per-user-cell game-length weight: sum of replay durations the
        # user contributed to this cell. Used for a duration-weighted
        # variant where a long game counts more than a short one.
        duration_weight=pl.col("duration_s").sum(),
        n_in_cell=pl.len(),
    )

    def _wmean_sim(col, weight_col="weight"):
        v = pl.col(col)
        w = pl.col(weight_col)
        valid_w = w.filter(v.is_not_null())
        valid_v = v.filter(v.is_not_null())
        return (valid_v * valid_w).sum() / valid_w.sum()

    sim_binned = (
        sim_per_user_cell.group_by(["bucket", "engine_version"])
        .agg(
            n_player_rows=pl.col("n_in_cell").sum(),
            n_users=pl.len(),
            rel_sim_ms_p50_avg=_wmean_sim("rel_sim_ms_p50_u"),
            rel_sim_ms_p95_avg=_wmean_sim("rel_sim_ms_p95_u"),
            rel_sim_ms_p95_late_avg=_wmean_sim("rel_sim_ms_p95_late_u"),
            rel_sim_ms_mean_avg=_wmean_sim("rel_sim_ms_mean_u"),
            # Unweighted mean of per-user means: every user counts the same
            # regardless of how many total replays they have. The weighted
            # version above lets regulars dominate, which is good for tracking
            # the typical experience; this version is good for asking
            # "did the median user's sim get faster or slower this week?"
            rel_sim_ms_mean_mom=pl.col("rel_sim_ms_mean_u").mean(),
            # Weighted by total game duration the user contributed in this
            # cell — a 60min match counts 6× more than a 10min one.
            rel_sim_ms_mean_durwavg=_wmean_sim("rel_sim_ms_mean_u", "duration_weight"),
        )
        .sort(["engine_version", "bucket"])
    )
    # Side-aggregation of BAR versions per (bucket × engine) cell — same
    # tooltip enrichment as the GPU tab.
    bar_per_cell_sim = (
        sim_with_bucket.filter(pl.col("bar_version").is_not_null())
        .group_by(["bucket", "engine_version"])
        .agg(
            n_bar_versions=pl.col("bar_version").n_unique(),
            bar_build_min=pl.col("bar_build").min(),
            bar_build_max=pl.col("bar_build").max(),
            bar_versions_str=pl.col("bar_version")
            .unique()
            .sort()
            .str.join(", "),
        )
    )
    sim_binned = sim_binned.join(
        bar_per_cell_sim, on=["bucket", "engine_version"], how="left"
    )
    sim_scope_md = mo.md(
        f"CPU scope: **{sim_norm_scope}** · {sim_norm_filtered.height:,} player-rows · "
        f"{sim_norm_filtered.select('player_name').n_unique():,} users · "
        f"{sim_binned.height} ({bucket_label_sim} × engine) cells · "
        f"weighted by per-user cohort replay count"
    )
    return bucket_label_sim, is_bar_sim, sim_binned, sim_scope_md


@app.cell
def _(
    alt,
    bucket_label_sim,
    cpu_picker,
    date_range_slider,
    granularity_picker,
    hw_split_checkbox,
    is_bar_sim,
    mo,
    pl,
    series_picker,
    series_select,
    sim_binned,
    sim_scope_md,
):
    MIN_SIM_CELL = 3
    sim_plot_df = sim_binned.filter(
        sim_binned["n_player_rows"] >= MIN_SIM_CELL
    ).to_pandas()
    if is_bar_sim:
        sim_plot_df = sim_plot_df.sort_values("bar_build_min")
    size_range_sim = (
        [5, 60] if (bucket_label_sim == "day" or is_bar_sim) else [20, 200]
    )
    _x_type = "N" if is_bar_sim else "T"
    _x_sort = (
        alt.SortField("bar_build_min", order="ascending")
        if is_bar_sim
        else None
    )
    _x_kwargs = {"sort": _x_sort} if is_bar_sim else {}

    def _sim_chart(metric: str, title: str):
        line = (
            alt.Chart(sim_plot_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X(
                    f"bucket:{_x_type}",
                    title=bucket_label_sim.capitalize(),
                    **_x_kwargs,
                ),
                y=alt.Y(
                    f"{metric}:Q",
                    title=title,
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("engine_version:N", title="Engine"),
            )
        )
        points = (
            alt.Chart(sim_plot_df)
            .mark_point(filled=True)
            .encode(
                x=alt.X(f"bucket:{_x_type}", **_x_kwargs),
                y=f"{metric}:Q",
                color=alt.Color("engine_version:N"),
                size=alt.Size(
                    "n_player_rows:Q",
                    title="player-rows",
                    scale=alt.Scale(range=size_range_sim),
                ),
                tooltip=[
                    alt.Tooltip(f"bucket:{_x_type}", title=bucket_label_sim),
                    "engine_version",
                    alt.Tooltip(f"{metric}:Q", format=".3f"),
                    "n_player_rows",
                    "n_users",
                    alt.Tooltip("n_bar_versions:Q", title="BAR versions (n)"),
                    alt.Tooltip("bar_build_min:Q", title="BAR build min"),
                    alt.Tooltip("bar_build_max:Q", title="BAR build max"),
                    alt.Tooltip("bar_versions_str:N", title="BAR versions"),
                ],
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [1.0]}).to_pandas())
            .mark_rule(color="gray", strokeDash=[3, 3])
            .encode(y="y:Q")
        )
        return (line + points + rule).properties(
            width=900, height=240, title=title
        )

    sim_view = mo.vstack(
        [
            mo.md(
                "Per-user normalized **sim_ms** (CPU work). Each user's first-week "
                "median is the anchor; subsequent replays are measured relative to it. "
                "**LOWER is better** (< 1.0 = sim got faster than first week)."
            ),
            granularity_picker,
            hw_split_checkbox,
            cpu_picker,
            series_picker,
            date_range_slider,
            series_select,
            sim_scope_md,
            mo.ui.altair_chart(
                _sim_chart(
                    "rel_sim_ms_p95_late_avg",
                    "Mean relative sim_ms p95 (late game) — LOWER is better",
                )
            ),
            mo.ui.altair_chart(
                _sim_chart(
                    "rel_sim_ms_p95_avg",
                    "Mean relative sim_ms p95 (full game) — LOWER is better",
                )
            ),
            mo.ui.altair_chart(
                _sim_chart(
                    "rel_sim_ms_mean_avg",
                    "Mean relative sim_ms mean (avg frame cost) — LOWER is better",
                )
            ),
            mo.ui.altair_chart(
                _sim_chart(
                    "rel_sim_ms_mean_mom",
                    "Mean of means — relative sim_ms mean, unweighted across users — LOWER is better",
                )
            ),
            mo.ui.altair_chart(
                _sim_chart(
                    "rel_sim_ms_mean_durwavg",
                    "Mean relative sim_ms mean, weighted by game duration — LOWER is better",
                )
            ),
            mo.ui.altair_chart(
                _sim_chart(
                    "rel_sim_ms_p50_avg",
                    "Mean relative sim_ms p50 (typical frame) — LOWER is better",
                )
            ),
        ]
    )
    return (sim_view,)


@app.cell
def _(mo):
    series_picker = mo.ui.dropdown(
        options={
            "Last 10 BAR versions": 10,
            "Last 20 BAR versions": 20,
            "Last 50 BAR versions": 50,
            "All BAR versions": None,
        },
        value="Last 20 BAR versions",
        label="Series",
    )
    # Color schemes ordered roughly by "how easy is it to distinguish many
    # adjacent lines": turbo cycles hues so 52 weeks stay distinct; the
    # perceptually-uniform schemes (viridis/plasma/inferno/magma/cividis)
    # read time direction more clearly but blur adjacent series. sinebow
    # and rainbow are full hue cycles — busiest, but easiest to spot a
    # specific outlier line.
    color_scheme_picker = mo.ui.dropdown(
        options=[
            "turbo",
            "plasma",
            "inferno",
            "magma",
            "viridis",
            "cividis",
            "sinebow",
            "rainbow",
            "warmgreys",
        ],
        value="turbo",
        label="Color scheme",
    )
    return color_scheme_picker, series_picker


@app.cell
def _(mo, pl, summary_raw):
    # Time-range slider scopes the BAR-version multiselect: only versions
    # with a replay inside this date window appear as options. Bounds come
    # from the actual data so the slider always covers the full corpus.
    _bounds = summary_raw.select(
        min_t=pl.col("replay_start_time").min(),
        max_t=pl.col("replay_start_time").max(),
    ).row(0)
    data_min_dt, data_max_dt = _bounds
    data_min_date = data_min_dt.date()
    _total_days = max(1, (data_max_dt.date() - data_min_date).days)
    date_range_slider = mo.ui.range_slider(
        start=0,
        stop=_total_days,
        step=1,
        value=[0, _total_days],
        label=f"Date offset (days from {data_min_date.isoformat()})",
        show_value=True,
    )
    return data_min_date, date_range_slider


@app.cell
def _(data_min_date, date_range_slider):
    # Convert the date-range slider's day offsets into tz-aware UTC datetimes
    # for filtering. End is exclusive (adding 1 day to the upper offset
    # includes that whole day's replays in the window).
    from datetime import datetime, time, timedelta, timezone

    _lo_d, _hi_d = date_range_slider.value
    bar_start_dt = datetime.combine(
        data_min_date + timedelta(days=_lo_d), time.min, tzinfo=timezone.utc
    )
    bar_end_dt = datetime.combine(
        data_min_date + timedelta(days=_hi_d + 1), time.min, tzinfo=timezone.utc
    )
    return bar_end_dt, bar_start_dt


@app.cell
def _(
    bar_end_dt,
    bar_start_dt,
    duration_slider,
    mo,
    pl,
    series_picker,
    summary_raw,
):
    # Series dimension is BAR `meta.game` version (extracted into bar_version
    # / bar_build by the aggregator). bar_build is the integer build number,
    # which is monotonic and gives us a deterministic ordering for both
    # "last N versions" selection and the color ramp.
    from datetime import timedelta as _timedelta

    _lo_min_sel, _hi_min_sel = duration_slider.value
    _n_sel = series_picker.value

    _versions_df = (
        summary_raw.filter(
            (pl.col("duration_s") >= _lo_min_sel * 60)
            & (pl.col("duration_s") <= _hi_min_sel * 60)
            & (pl.col("replay_start_time") >= bar_start_dt)
            & (pl.col("replay_start_time") < bar_end_dt)
            & pl.col("bar_version").is_not_null()
        )
        .select(["bar_version", "bar_build"])
        .unique()
        .sort(
            ["bar_build", "bar_version"], descending=[True, True], nulls_last=True
        )
    )
    if _n_sel is not None:
        _versions_df = _versions_df.head(_n_sel)
    # Re-sort ascending for display + color order (oldest → newest).
    _versions_df = _versions_df.sort(
        ["bar_build", "bar_version"], nulls_last=True
    )
    available_versions = _versions_df.get_column("bar_version").to_list()
    _window_label = (
        f"{bar_start_dt.date().isoformat()} → "
        f"{(bar_end_dt - _timedelta(days=1)).date().isoformat()}"
    )
    series_select = mo.ui.multiselect(
        options=available_versions,
        value=available_versions,
        label=(
            f"Show BAR versions ({len(available_versions)} in window "
            f"{_window_label}, deselect to hide)"
        ),
    )
    return (series_select,)


@app.cell
def _(PER_MINUTE_PATH, duration_slider, mo, pl, series_select):
    # Per-(replay, minute) parquet: samples pooled across all players in a
    # replay for that minute. Used to plot frame cost vs minute-into-game,
    # with one line per selected BAR version so you can see how the
    # within-game cost curve has shifted across releases.
    per_minute_raw = pl.read_parquet(PER_MINUTE_PATH).filter(
        pl.col("bar_version").is_not_null()
    )
    lo_min_pm, hi_min_pm = duration_slider.value
    per_minute_b = per_minute_raw.filter(
        (pl.col("duration_s") >= lo_min_pm * 60)
        & (pl.col("duration_s") <= hi_min_pm * 60)
        & pl.col("bar_version").is_in(list(series_select.value))
    )

    MIN_MINUTE_REPLAYS = 30
    per_minute_binned = (
        per_minute_b.group_by(["minute", "bar_version"])
        .agg(
            n_replays=pl.col("replay_id").n_unique(),
            # Carry bar_build through so the chart can sort the color ramp
            # by release order rather than alphabetical bar_version.
            bar_build=pl.col("bar_build").first(),
            sim_ms_p50_med=pl.col("sim_ms_p50").median(),
            sim_ms_p95_med=pl.col("sim_ms_p95").median(),
            sim_ms_p99_med=pl.col("sim_ms_p99").median(),
            sim_ms_mean_med=pl.col("sim_ms_mean").median(),
            fps_p5_med=pl.col("fps_p5").median(),
            fps_p50_med=pl.col("fps_p50").median(),
            draw_ms_est_med=pl.col("draw_ms_est").median(),
        )
        .filter(pl.col("n_replays") >= MIN_MINUTE_REPLAYS)
        .sort(["bar_build", "minute"])
    )
    per_minute_scope_md = mo.md(
        f"**{per_minute_b.height:,}** (replay × minute) rows from "
        f"**{per_minute_b.select('replay_id').n_unique():,}** replays · "
        f"length **{lo_min_pm}–{hi_min_pm} min** · "
        f"**{len(series_select.value)}** BAR versions shown · "
        f"≥{MIN_MINUTE_REPLAYS} replays per (minute × version) cell"
    )
    return per_minute_binned, per_minute_scope_md


@app.cell
def _(duration_slider, mo, pl, series_select, summary_raw):
    # Per-replay speed metrics (internalSpeed quantiles, pct_time_slowed) are
    # global to a replay — duplicated across player rows in summary_raw — so
    # we dedupe to one row per replay before binning. Otherwise large lobbies
    # would weight more heavily.
    lo_min_sd, hi_min_sd = duration_slider.value
    speed_replays = (
        summary_raw.filter(
            (pl.col("duration_s") >= lo_min_sd * 60)
            & (pl.col("duration_s") <= hi_min_sd * 60)
            & pl.col("bar_version").is_not_null()
        )
        .filter(pl.col("internal_speed_p10").is_not_null())
        .unique(subset=["replay_id"], keep="first")
        .filter(pl.col("bar_version").is_in(list(series_select.value)))
    )

    SPEED_BUCKET_MIN = 5
    MIN_SPEED_REPLAYS = 5
    speed_dur_binned = (
        speed_replays.with_columns(
            duration_bucket=(pl.col("duration_s") / 60.0).floor()
            // SPEED_BUCKET_MIN
            * SPEED_BUCKET_MIN
        )
        .group_by(["duration_bucket", "engine_version"])
        .agg(
            n_replays=pl.len(),
            n_bar_versions=pl.col("bar_version").n_unique(),
            internal_speed_p10_med=pl.col("internal_speed_p10").median(),
            internal_speed_p25_med=pl.col("internal_speed_p25").median(),
            internal_speed_p50_med=pl.col("internal_speed_p50").median(),
            min_internal_speed_med=pl.col("min_internal_speed").median(),
            pct_time_slowed_med=pl.col("pct_time_slowed").median(),
        )
        .filter(pl.col("n_replays") >= MIN_SPEED_REPLAYS)
        .sort(["engine_version", "duration_bucket"])
    )
    n_engines_shown = speed_dur_binned.select("engine_version").n_unique()
    speed_dur_md = mo.md(
        f"**{speed_replays.height:,}** replays binned to "
        f"**{SPEED_BUCKET_MIN}-min** duration buckets · "
        f"**{n_engines_shown}** engine versions shown "
        f"(across **{len(series_select.value)}** selected BAR versions) · "
        f"≥{MIN_SPEED_REPLAYS} replays per (bucket × engine) cell"
    )
    return SPEED_BUCKET_MIN, speed_dur_binned, speed_dur_md


@app.cell
def _(
    SPEED_BUCKET_MIN,
    alt,
    color_scheme_picker,
    date_range_slider,
    mo,
    per_minute_binned,
    per_minute_scope_md,
    pl,
    series_picker,
    series_select,
    speed_dur_binned,
    speed_dur_md,
):
    color_scheme = color_scheme_picker.value
    # Keep color/legend in release order (oldest → newest by bar_build),
    # not alphabetical bar_version.
    _color_sort = alt.SortField("bar_build", order="ascending")

    # One chart per metric: with many BAR versions and a single metric,
    # color-only encoding stays readable. Mixing metrics in one chart with
    # that many series is a mess — strokeDash wouldn't carry against
    # the chosen scheme.
    def _minute_chart(
        metric: str, title: str, y_title: str = "Frame time (ms)"
    ):
        df = per_minute_binned.filter(pl.col(metric).is_not_null()).to_pandas()
        return (
            alt.Chart(df)
            .mark_line(strokeWidth=1.2)
            .encode(
                x=alt.X("minute:Q", title="Minute into game"),
                y=alt.Y(
                    f"{metric}:Q",
                    title=y_title,
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "bar_version:N",
                    title="BAR version",
                    sort=_color_sort,
                    scale=alt.Scale(scheme=color_scheme),
                ),
                tooltip=[
                    alt.Tooltip("minute:Q", title="minute"),
                    alt.Tooltip("bar_version:N", title="BAR version"),
                    alt.Tooltip("bar_build:Q", title="build"),
                    alt.Tooltip(f"{metric}:Q", format=".2f"),
                    "n_replays",
                ],
            )
            .properties(width=900, height=300, title=title)
        )

    def _speed_chart(metric: str, title: str, y_format: str = ".3f"):
        # Sim-speed tab is sliced by engine_version (typically 3–6 values),
        # since slowdown behavior tracks engine releases more cleanly than
        # individual BAR builds. The bar_version multiselect still narrows
        # which replays feed the aggregation.
        df = speed_dur_binned.filter(pl.col(metric).is_not_null()).to_pandas()
        return (
            alt.Chart(df)
            .mark_line(strokeWidth=1.5, point=True)
            .encode(
                x=alt.X(
                    "duration_bucket:Q",
                    title=f"Game duration (min, {SPEED_BUCKET_MIN}-min buckets)",
                ),
                y=alt.Y(
                    f"{metric}:Q",
                    title=title,
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "engine_version:N",
                    title="Engine",
                    scale=alt.Scale(scheme=color_scheme),
                ),
                tooltip=[
                    alt.Tooltip("duration_bucket:Q", title="duration (min)"),
                    alt.Tooltip("engine_version:N", title="Engine"),
                    alt.Tooltip(f"{metric}:Q", format=y_format),
                    "n_replays",
                    alt.Tooltip("n_bar_versions:Q", title="BAR versions (n)"),
                ],
            )
            .properties(width=900, height=300, title=title)
        )

    # Shared controls header — same UI elements appear in both tabs so
    # changing the series picker, date range, or color scheme on one tab
    # immediately reflects on the other.
    _controls = [
        mo.hstack([series_picker, color_scheme_picker], justify="start"),
        date_range_slider,
        series_select,
        per_minute_scope_md,
    ]
    _intro = mo.md(
        "Per-minute frame timing from `perf_per_minute.parquet`. "
        "X-axis is **minute into the game** (not total game length) — "
        "shows how cost evolves as a single match progresses. One "
        "line per selected **BAR version** (`meta.game`); color goes "
        "dark→bright with build number, so the most recent release "
        "is the brightest. The `duration_slider` at the top filters "
        "which replays are in scope; the multiselect below lets you "
        "hide individual versions to declutter."
    )

    dur_sim_view = mo.vstack(
        [
            _intro,
            *_controls,
            mo.ui.altair_chart(
                _minute_chart("sim_ms_p50_med", "Sim ms p50 vs minute into game")
            ),
            mo.ui.altair_chart(
                _minute_chart("sim_ms_p95_med", "Sim ms p95 vs minute into game")
            ),
            mo.ui.altair_chart(
                _minute_chart("sim_ms_p99_med", "Sim ms p99 vs minute into game")
            ),
            mo.ui.altair_chart(
                _minute_chart(
                    "sim_ms_mean_med", "Sim ms mean vs minute into game"
                )
            ),
        ]
    )

    dur_draw_view = mo.vstack(
        [
            _intro,
            *_controls,
            mo.ui.altair_chart(
                _minute_chart(
                    "fps_p5_med",
                    "FPS p5 vs minute into game (HIGHER is better)",
                    y_title="FPS",
                )
            ),
            mo.ui.altair_chart(
                _minute_chart(
                    "fps_p50_med",
                    "FPS p50 vs minute into game (HIGHER is better)",
                    y_title="FPS",
                )
            ),
            mo.ui.altair_chart(
                _minute_chart(
                    "draw_ms_est_med", "Estimated draw_ms vs minute into game"
                )
            ),
        ]
    )

    dur_speed_view = mo.vstack(
        [
            mo.md(
                "Per-replay **speed factor** vs total game duration, from "
                "`perf_summary.parquet`. `internalSpeed` is the engine's "
                "global speed multiplier — it drops below 1.0 when the sim "
                "can't keep up. Quantiles are interval-weighted across the "
                "replay (so brief slowdowns don't dominate). One series per "
                "selected BAR version."
            ),
            *_controls,
            speed_dur_md,
            mo.ui.altair_chart(
                _speed_chart(
                    "internal_speed_p10_med",
                    "internalSpeed p10 (worst 10% of game time) vs duration — HIGHER is better",
                )
            ),
            mo.ui.altair_chart(
                _speed_chart(
                    "internal_speed_p25_med",
                    "internalSpeed p25 vs duration — HIGHER is better",
                )
            ),
            mo.ui.altair_chart(
                _speed_chart(
                    "internal_speed_p50_med",
                    "internalSpeed p50 (median speed across the game) vs duration — HIGHER is better",
                )
            ),
            mo.ui.altair_chart(
                _speed_chart(
                    "min_internal_speed_med",
                    "min internalSpeed (worst single moment) vs duration — HIGHER is better",
                )
            ),
            mo.ui.altair_chart(
                _speed_chart(
                    "pct_time_slowed_med",
                    "pct of game time slowed (speed < 1.0) vs duration — LOWER is better",
                    y_format=".2%",
                )
            ),
        ]
    )
    return dur_draw_view, dur_sim_view, dur_speed_view


@app.cell
def _(
    cpu_view,
    dur_draw_view,
    dur_sim_view,
    dur_speed_view,
    gpu_view,
    mo,
    new_players_view,
    sim_view,
):
    mo.ui.tabs(
        {
            "CPU / Sim": cpu_view,
            "Sim — per-user normalized": sim_view,
            "GPU / FPS — per-user normalized": gpu_view,
            "Sim time vs duration": dur_sim_view,
            "Draw time vs duration": dur_draw_view,
            "Sim speed vs duration": dur_speed_view,
            "Player growth": new_players_view,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
