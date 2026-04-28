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
    return SUMMARY_PATH, alt, mo, pl


@app.cell
def _(SUMMARY_PATH, mo, pl):
    summary = (
        pl.read_parquet(SUMMARY_PATH)
        .filter(~pl.col("is_spectator"))
        .filter(pl.col("engine_version").is_not_null())
        .filter(pl.col("replay_start_time").is_not_null())
    )
    mo.md(
        f"**{summary.height:,}** player-rows from "
        f"**{summary.select('replay_id').n_unique():,}** replays · "
        f"engines: {sorted(summary.select('engine_version').unique().to_series().to_list())}"
    )
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
def _(cpu_filtered, cpu_scope, mo, pl):
    cpu_binned = (
        cpu_filtered.with_columns(
            week=pl.col("replay_start_time").dt.truncate("1w"),
        )
        .group_by(["week", "engine_version"])
        .agg(
            n_player_rows=pl.len(),
            n_replays=pl.col("replay_id").n_unique(),
            sim_ms_p95_late_med=pl.col("sim_ms_p95_late").median(),
            pct_time_slowed_med=pl.col("pct_time_slowed").median(),
            internal_speed_p10_med=pl.col("internal_speed_p10").median(),
            sim_ms_p95_med=pl.col("sim_ms_p95").median(),
        )
        .sort(["engine_version", "week"])
    )
    cpu_scope_md = mo.md(
        f"Scope: **{cpu_scope}** · {cpu_filtered.height:,} player-rows · "
        f"{cpu_filtered.select('replay_id').n_unique():,} replays · "
        f"{cpu_binned.height} (week × engine) cells"
    )
    return cpu_binned, cpu_scope_md


@app.cell
def _(alt, cpu_binned, cpu_picker, cpu_scope_md, mo):
    MIN_CELL = 2
    cpu_plot_df = cpu_binned.filter(
        cpu_binned["n_player_rows"] >= MIN_CELL
    ).to_pandas()

    def _cpu_line(metric: str, title: str, y_format: str = ".2f"):
        line = (
            alt.Chart(cpu_plot_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("week:T", title="Week"),
                y=alt.Y(f"{metric}:Q", title=title),
                color=alt.Color("engine_version:N", title="Engine"),
            )
        )
        points = (
            alt.Chart(cpu_plot_df)
            .mark_point(filled=True)
            .encode(
                x="week:T",
                y=f"{metric}:Q",
                color=alt.Color("engine_version:N"),
                size=alt.Size(
                    "n_player_rows:Q",
                    title="player-rows",
                    scale=alt.Scale(range=[20, 200]),
                ),
                tooltip=[
                    alt.Tooltip("week:T"),
                    "engine_version",
                    alt.Tooltip(f"{metric}:Q", format=y_format),
                    "n_player_rows",
                    "n_replays",
                ],
            )
        )
        return (line + points).properties(width=900, height=240, title=title)

    cpu_view = mo.vstack(
        [
            cpu_picker,
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
def _(pl, summary):
    # Temporal baseline: each user's baseline = the median of their fps
    # metrics in THEIR FIRST WEEK with data. Every subsequent replay is
    # measured relative to that fixed first-week anchor. Drift over time
    # (across weeks/engines) now flows through the relative metric instead
    # of being absorbed into a global per-user median.
    MIN_USER_REPLAYS = 3
    s_w = summary.with_columns(
        week=pl.col("replay_start_time").dt.truncate("1w")
    ).filter(pl.col("fps_p50").is_not_null())

    # Per-user first-week + total replay count.
    first_weeks = (
        s_w.group_by("player_name")
        .agg(
            first_week=pl.col("week").min(),
            n_user_replays=pl.col("replay_id").n_unique(),
        )
        .filter(pl.col("n_user_replays") >= MIN_USER_REPLAYS)
    )

    # Baseline = median of metrics among the user's replays in their first week.
    user_baselines = (
        s_w.join(first_weeks, on="player_name", how="inner")
        .filter(pl.col("week") == pl.col("first_week"))
        .group_by("player_name")
        .agg(
            baseline_fps_p50=pl.col("fps_p50").median(),
            baseline_fps_p5=pl.col("fps_p5").median(),
            baseline_fps_mean=pl.col("fps_mean").median(),
            baseline_draw_ms_p50=pl.col("draw_ms_p50").median(),
            baseline_draw_ms_p95=pl.col("draw_ms_p95").median(),
            first_week=pl.col("first_week").first(),
            n_user_replays=pl.col("n_user_replays").first(),
            n_baseline_replays=pl.len(),
        )
        .filter(pl.col("baseline_fps_p50") > 0)
        .filter(pl.col("baseline_fps_p5") > 0)
        .filter(pl.col("baseline_fps_mean") > 0)
    )

    summary_norm = summary.join(
        user_baselines, on="player_name", how="inner"
    ).with_columns(
        rel_fps_p5=pl.col("fps_p5") / pl.col("baseline_fps_p5"),
        rel_fps_p50=pl.col("fps_p50") / pl.col("baseline_fps_p50"),
        rel_fps_mean=pl.col("fps_mean") / pl.col("baseline_fps_mean"),
        # rel_draw_ms: < 1.0 = faster than first week (good), > 1.0 = slower
        rel_draw_ms_p50=pl.col("draw_ms_p50") / pl.col("baseline_draw_ms_p50"),
        rel_draw_ms_p95=pl.col("draw_ms_p95") / pl.col("baseline_draw_ms_p95"),
    )
    return MIN_USER_REPLAYS, summary_norm, user_baselines


@app.cell
def _(MIN_USER_REPLAYS, mo, pl, summary_norm, user_baselines):
    n_solo = user_baselines.filter(pl.col("n_baseline_replays") == 1).height
    n_multi = user_baselines.height - n_solo
    gpu_baseline_md = mo.md(
        f"**{user_baselines.height:,}** users with ≥{MIN_USER_REPLAYS} replays · "
        f"**{summary_norm.height:,}** normalized player-rows · "
        f"baseline = median of each user's `fps_*` in their **first week** with data "
        f"({n_multi:,} users had ≥2 replays in their first week, {n_solo:,} had only 1). "
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
def _(gpu_filtered, gpu_scope, mo, pl):
    # Two-step weighted aggregation:
    #   1. Per (user × week × engine), take mean of relative metrics — every
    #      user contributes ONE value per cell regardless of replay count.
    #   2. Across users in a cell, take WEIGHTED mean where weight =
    #      user's total cohort replay count. Regulars dominate, one-off
    #      players have small influence on the cell average.
    per_user_cell = (
        gpu_filtered.with_columns(
            week=pl.col("replay_start_time").dt.truncate("1w"),
        )
        .group_by(["week", "engine_version", "player_name"])
        .agg(
            rel_fps_p5_u=pl.col("rel_fps_p5").mean(),
            rel_fps_mean_u=pl.col("rel_fps_mean").mean(),
            rel_fps_p50_u=pl.col("rel_fps_p50").mean(),
            rel_draw_ms_p50_u=pl.col("rel_draw_ms_p50").mean(),
            rel_draw_ms_p95_u=pl.col("rel_draw_ms_p95").mean(),
            weight=pl.col("n_user_replays").first(),
            n_in_cell=pl.len(),
        )
    )

    def _wmean(col):
        v = pl.col(col)
        w = pl.col("weight")
        valid_w = w.filter(v.is_not_null())
        valid_v = v.filter(v.is_not_null())
        return (valid_v * valid_w).sum() / valid_w.sum()

    gpu_binned = (
        per_user_cell.group_by(["week", "engine_version"])
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
        .sort(["engine_version", "week"])
    )
    gpu_scope_md = mo.md(
        f"GPU scope: **{gpu_scope}** · {gpu_filtered.height:,} player-rows · "
        f"{gpu_filtered.select('player_name').n_unique():,} users · "
        f"{gpu_binned.height} (week × engine) cells · "
        f"weighted by per-user cohort replay count"
    )
    return gpu_binned, gpu_scope_md


@app.cell
def _(alt, gpu_baseline_md, gpu_binned, gpu_picker, gpu_scope_md, mo, pl):
    MIN_GPU_CELL = 3
    gpu_plot_df = gpu_binned.filter(
        gpu_binned["n_player_rows"] >= MIN_GPU_CELL
    ).to_pandas()

    def _gpu_chart(metric: str, title: str):
        line = (
            alt.Chart(gpu_plot_df)
            .mark_line(strokeWidth=1.5)
            .encode(
                x=alt.X("week:T", title="Week"),
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
                x="week:T",
                y=f"{metric}:Q",
                color=alt.Color("engine_version:N"),
                size=alt.Size(
                    "n_player_rows:Q",
                    title="player-rows",
                    scale=alt.Scale(range=[20, 200]),
                ),
                tooltip=[
                    alt.Tooltip("week:T"),
                    "engine_version",
                    alt.Tooltip(f"{metric}:Q", format=".3f"),
                    "n_player_rows",
                    "n_users",
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
            gpu_picker,
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
def _(cpu_view, gpu_view, mo):
    mo.ui.tabs(
        {
            "CPU / Sim": cpu_view,
            "GPU / FPS — per-user normalized": gpu_view,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
