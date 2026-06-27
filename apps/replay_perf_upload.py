# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
#     "polars>=1.0",
#     "altair>=5.0",
#     "pyarrow>=15.0",
# ]
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        """
        # Replay performance — upload a `.sdfz`

        Drop a Beyond All Reason demo below. It's parsed **entirely in your
        browser** (no upload to any server) to show per-player sim-frame timing,
        render FPS, server speed, a whole-game summary, and the hardware each
        player reported.
        """
    )
    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl


@app.cell
def _(pl):
    # --- Pure-Python .sdfz parser (runs identically locally and in WASM) ------
    # Ported from the perf-relevant slice of sdfz-demo-parser (TS):
    #   header  -> demo-parser.ts:parseHeader
    #   packets -> packet-parser.ts (PLAYERINFO/INTERNAL_SPEED/STARTPLAYING/
    #              GAMEOVER/LUAMSG) and the gameStartTime/actualGameTime rule
    #   sim ms  -> bin/extract-perf.ts:deriveSimFrameMs
    #   lua     -> lua-parser.ts (FPS_BROADCAST, SYSTEM_INFO)
    #   players -> script-parser.ts (player/team/allyteam from the TDF script)
    # Validated for exact perf/fps/sim parity against the TS extractor.
    import gzip
    import math
    import re
    import struct
    from datetime import datetime, timezone

    GAME_SPEED = 30
    FRAME_BUDGET_MS = 1000.0 / GAME_SPEED
    MIN_DRAW_FPS = 2
    SAT_EPS = 1e-3

    # packet ids we act on; everything else is skipped by length
    STARTPLAYING, INTERNAL_SPEED, GAMEOVER, PLAYERINFO, LUAMSG = 4, 20, 30, 38, 50

    _PLAYERINFO = struct.Struct("<Bfi")   # playerNum u8, cpuUsage f32, ping i32
    _F32 = struct.Struct("<f")
    _I32 = struct.Struct("<i")
    _FRAME = struct.Struct("<fI")         # modGameTime f32, length u32

    _SYSTEM_INFO_RE = re.compile(
        r"(?:CPU:\s+(?P<cpu>.*?)\s*)?"
        r"(?:CPU cores:\s+(?:[\d:.]+\]\s+Physical CPU Cores:\s+)?(?P<cpuCores>\d+)\s*"
        r"(?:/\s*(?:[\d:.]+\]\s+Logical CPU Cores:\s+)?(?P<logicalCpuCores>\d+))?\s*)?"
        r"(?:RAM:\s+(?P<memory>.*?)\s*)?"
        r"(?:GPU:\s+(?P<gpu>.*?)\s*)?"
        r"(?:GPU VRAM:\s+(?P<gpuMemory>.*?)\s*)?"
        r"(?:Display max:\s+(?P<maxRes>.*?)\n(?P<display>.*?)\s+(?P<windowMode>.*?)\s*)?"
        r"(?:OS:\s+(?P<os>.*?)\s*)?"
        r"(?:Engine:\s+(?P<wordSize>.*?)\s*)?"
        r"(?:Lobby:\s+(?P<lobby>.*?)\s*)?$"
    )

    _PERF_SCHEMA = {
        "playerNum": pl.Int64, "t": pl.Float64, "simFrameMs": pl.Float64,
        "simFrameMsQuality": pl.Utf8, "internalSpeed": pl.Float64,
        "cpuUsage": pl.Float64, "ping": pl.Int64,
    }
    _FPS_SCHEMA = {"playerNum": pl.Int64, "t": pl.Float64, "fps": pl.Float64}
    _PLAYERS_SCHEMA = {
        "playerNum": pl.Int64, "name": pl.Utf8, "teamId": pl.Int64,
        "allyTeamId": pl.Int64, "is_spectator": pl.Boolean, "faction": pl.Utf8,
        "country": pl.Utf8, "skill": pl.Utf8, "winning_team": pl.Int64,
        "match_id": pl.Utf8,
    }
    _HW_COLS = ["cpu", "cpuCores", "logicalCpuCores", "memory", "gpu",
                "gpuMemory", "maxRes", "display", "windowMode", "os", "lobby"]
    _HW_SCHEMA = {"playerNum": pl.Int64, "name": pl.Utf8,
                  **{c: pl.Utf8 for c in _HW_COLS}}

    def _derive_sim_frame_ms(cpu_usage, internal_speed, last_fps):
        if not math.isfinite(cpu_usage) or cpu_usage < 0 or cpu_usage > 1 + SAT_EPS:
            return None, "invalid"
        if cpu_usage >= 1 - SAT_EPS:
            return None, "saturated"
        fps = last_fps if last_fps is not None else 60
        draw_adj = min(1, MIN_DRAW_FPS / max(1, fps))
        low_threshold = min(0.65, 0.8 - draw_adj)
        sim = cpu_usage * FRAME_BUDGET_MS / max(0.01, internal_speed)
        quality = "exact" if cpu_usage <= low_threshold else "approx"
        return sim, quality

    class _Reader:
        __slots__ = ("buf", "off")

        def __init__(self, buf):
            self.buf, self.off = buf, 0

        def read(self, n):
            b = self.buf[self.off:self.off + n]
            self.off += n
            return b

        def i32(self):
            v = _I32.unpack_from(self.buf, self.off)[0]
            self.off += 4
            return v

        def i64(self):
            v = struct.unpack_from("<q", self.buf, self.off)[0]
            self.off += 8
            return v

        def string(self, n):
            return self.read(n).decode("latin-1").replace("\x00", "")

    def _parse_header(r):
        h = {}
        h["magic"] = r.string(16)
        h["version"] = r.i32()
        h["headerSize"] = r.i32()
        h["versionString"] = r.string(256)
        h["gameId"] = r.read(16).hex()
        h["startTime"] = r.i64()
        h["scriptSize"] = r.i32()
        h["demoStreamSize"] = r.i32()
        h["gameTime"] = r.i32()
        h["wallclockTime"] = r.i32()
        h["numPlayers"] = r.i32()
        h["playerStatSize"] = r.i32()
        h["playerStatElemSize"] = r.i32()
        h["numTeams"] = r.i32()
        h["teamStatSize"] = r.i32()
        h["teamStatElemSize"] = r.i32()
        h["teamStatPeriod"] = r.i32()
        h["winningAllyTeamsSize"] = r.i32()
        return h

    def _parse_tdf(text):
        n = len(text)
        pos = 0

        def skip_ws():
            nonlocal pos
            while pos < n:
                c = text[pos]
                if c in " \t\r\n":
                    pos += 1
                elif text.startswith("//", pos):
                    nl = text.find("\n", pos)
                    pos = n if nl < 0 else nl
                elif text.startswith("/*", pos):
                    e = text.find("*/", pos)
                    pos = n if e < 0 else e + 2
                else:
                    break

        def parse_block():
            nonlocal pos
            obj = {}
            while pos < n:
                skip_ws()
                if pos >= n:
                    break
                c = text[pos]
                if c == "}":
                    pos += 1
                    break
                if c == "[":
                    end = text.find("]", pos)
                    if end < 0:
                        break
                    name = text[pos + 1:end].strip().lower()
                    pos = end + 1
                    skip_ws()
                    if pos < n and text[pos] == "{":
                        pos += 1
                        obj[name] = parse_block()
                    else:
                        obj[name] = {}
                else:
                    eq = text.find("=", pos)
                    if eq < 0:
                        break
                    key = text[pos:eq].strip().lower()
                    semi = text.find(";", eq + 1)
                    if semi < 0:
                        cands = [x for x in (text.find("\n", eq + 1),
                                             text.find("}", eq + 1)) if x != -1]
                        semi = min(cands) if cands else n
                    obj[key] = text[eq + 1:semi].strip()
                    pos = semi + 1 if (semi < n and text[semi] == ";") else semi
            return obj

        return parse_block()

    def _to_int(s):
        try:
            return int(s)
        except (TypeError, ValueError):
            return None

    def _build_players(root):
        game = root.get("game", {})
        map_name = game.get("mapname")
        teams = {}
        for key, val in game.items():
            if isinstance(val, dict) and key.startswith("team") and key[4:].isdigit():
                teams[int(key[4:])] = {
                    "allyTeamId": _to_int(val.get("allyteam")),
                    "faction": val.get("side"),
                }
        players = []
        for key, val in game.items():
            if not (isinstance(val, dict) and key.startswith("player")
                    and key[6:].isdigit()):
                continue
            pid = int(key[6:])
            team_id = _to_int(val.get("team"))
            tm = teams.get(team_id, {})
            players.append({
                "playerNum": pid,
                "name": val.get("name") or f"player{pid}",
                "teamId": team_id,
                "allyTeamId": tm.get("allyTeamId"),
                "is_spectator": val.get("spectator") == "1",
                "faction": tm.get("faction"),
                "country": val.get("countrycode"),
                "skill": val.get("skill"),
            })
        return map_name, players

    def parse_demo_bytes(raw):
        sdf = gzip.decompress(raw)
        r = _Reader(sdf)
        h = _parse_header(r)
        r.off = h["headerSize"]
        script_text = r.read(h["scriptSize"]).decode("utf-8", "replace")
        stream = r.read(h["demoStreamSize"])
        map_name, players = _build_players(_parse_tdf(script_text))

        # columnar accumulators — avoid per-row tuple/object overhead
        c_pid, c_t, c_sim, c_q, c_isp, c_cpu, c_ping = [], [], [], [], [], [], []
        f_pid, f_t, f_fps = [], [], []
        hardware = {}
        last_fps = {}
        internal_speed = 1.0
        game_start = None
        duration_s = None
        winning_ally = []

        buf = stream
        n = len(buf)
        off = 0
        frame_unpack = _FRAME.unpack_from
        pi_unpack = _PLAYERINFO.unpack_from
        ap_pid, ap_t, ap_sim, ap_q, ap_isp, ap_cpu, ap_ping = (
            c_pid.append, c_t.append, c_sim.append, c_q.append,
            c_isp.append, c_cpu.append, c_ping.append)
        af_pid, af_t, af_fps = f_pid.append, f_t.append, f_fps.append

        while off + 8 <= n:
            mod_game_time, length = frame_unpack(buf, off)
            off += 8
            if length == 0 or off + length > n:
                break
            pid = buf[off]
            t = 0.0 if game_start is None else mod_game_time - game_start

            if pid == PLAYERINFO:
                player_num, cpu_usage, ping = pi_unpack(buf, off + 1)
                sim, qual = _derive_sim_frame_ms(
                    cpu_usage, internal_speed, last_fps.get(player_num))
                ap_pid(player_num); ap_t(t); ap_sim(sim); ap_q(qual)
                ap_isp(internal_speed); ap_cpu(cpu_usage); ap_ping(ping)
            elif pid == INTERNAL_SPEED:
                internal_speed = _F32.unpack_from(buf, off + 1)[0]
            elif pid == STARTPLAYING:
                if game_start is None and _I32.unpack_from(buf, off + 1)[0] == 0:
                    game_start = mod_game_time
            elif pid == GAMEOVER:
                winning_ally = list(buf[off + 3:off + length])
                duration_s = t
            elif pid == LUAMSG:
                msg = buf[off + 7:off + length]
                if msg:
                    if msg[0] == 0x40:  # '@' FPS_BROADCAST
                        try:
                            fps = float(msg.decode("latin-1")[3:])
                        except ValueError:
                            fps = None
                        if fps is not None and math.isfinite(fps):
                            pn = buf[off + 3]
                            last_fps[pn] = fps
                            af_pid(pn); af_t(t); af_fps(fps)
                    elif msg[:3] == b"$y$":  # SYSTEM_INFO
                        pn = buf[off + 3]
                        if pn not in hardware:
                            m = _SYSTEM_INFO_RE.match(msg.decode("latin-1")[5:])
                            if m:
                                hardware[pn] = m.groupdict()
            off += length

        if duration_s is None:
            # No GAMEOVER: mirror demo-parser.ts (falls back to wallclockTime).
            duration_s = float(h["wallclockTime"])
        winning_team = winning_ally[0] if winning_ally else None
        started = datetime.fromtimestamp(
            h["startTime"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

        perf_df = pl.DataFrame(
            {"playerNum": c_pid, "t": c_t, "simFrameMs": c_sim,
             "simFrameMsQuality": c_q, "internalSpeed": c_isp,
             "cpuUsage": c_cpu, "ping": c_ping},
            schema=_PERF_SCHEMA,
        )
        fps_df = pl.DataFrame(
            {"playerNum": f_pid, "t": f_t, "fps": f_fps}, schema=_FPS_SCHEMA)

        players_enriched = pl.DataFrame(
            players,
            schema={k: _PLAYERS_SCHEMA[k] for k in (
                "playerNum", "name", "teamId", "allyTeamId", "is_spectator",
                "faction", "country", "skill")},
        ).with_columns(
            winning_team=pl.lit(winning_team, dtype=pl.Int64),
            match_id=pl.lit(None, dtype=pl.Utf8),
        )
        players_all = players_enriched.select(
            "playerNum", "name", "teamId", "allyTeamId", "is_spectator")

        name_by_pid = {p["playerNum"]: p["name"] for p in players}
        hw_rows = [
            {"playerNum": pn, "name": name_by_pid.get(pn, f"player{pn}"),
             **{c: hw.get(c) for c in _HW_COLS}}
            for pn, hw in sorted(hardware.items())
        ]
        hardware_df = pl.DataFrame(hw_rows, schema=_HW_SCHEMA)

        replay_row = {
            "map": map_name, "engine_version": h["versionString"],
            "bar_build": None, "duration_s": duration_s,
            "frame_rate": GAME_SPEED, "started": started,
        }
        return {
            "replay_row": replay_row, "frame_rate": GAME_SPEED,
            "duration_s": duration_s, "perf_df": perf_df, "fps_df": fps_df,
            "players_all": players_all, "players_enriched": players_enriched,
            "hardware_df": hardware_df,
        }

    def empty_result():
        players_enriched = pl.DataFrame(schema=_PLAYERS_SCHEMA)
        return {
            "replay_row": {"map": None, "engine_version": None, "bar_build": None,
                           "duration_s": 0.0, "frame_rate": GAME_SPEED,
                           "started": None},
            "frame_rate": GAME_SPEED, "duration_s": 0.0,
            "perf_df": pl.DataFrame(schema=_PERF_SCHEMA),
            "fps_df": pl.DataFrame(schema=_FPS_SCHEMA),
            "players_all": players_enriched.select(
                "playerNum", "name", "teamId", "allyTeamId", "is_spectator"),
            "players_enriched": players_enriched,
            "hardware_df": pl.DataFrame(schema=_HW_SCHEMA),
        }

    return empty_result, parse_demo_bytes


@app.cell
def _(mo):
    uploader = mo.ui.file(
        filetypes=[".sdfz"],
        kind="area",
        max_size=500_000_000,
        label="Upload a BAR .sdfz demo",
    )
    uploader
    return (uploader,)


@app.cell
def _(empty_result, mo, parse_demo_bytes, pl, uploader):
    if uploader.value:
        _res = parse_demo_bytes(uploader.value[0].contents)
    else:
        _res = empty_result()

    replay_row = _res["replay_row"]
    frame_rate = _res["frame_rate"]
    duration_s = _res["duration_s"]
    perf_df = _res["perf_df"]
    fps_df = _res["fps_df"]
    players_all = _res["players_all"]
    players_enriched = _res["players_enriched"]
    hardware_df = _res["hardware_df"]

    # Server-side speed is the same for all players at a given frame; one
    # player's series is enough. Pick the lowest playerNum to be stable.
    if perf_df.is_empty():
        speed_df = pl.DataFrame(schema={"t": pl.Float64, "internalSpeed": pl.Float64})
    else:
        speed_df = (
            perf_df.filter(pl.col("playerNum") == perf_df["playerNum"].min())
            .select("t", "internalSpeed")
            .filter(pl.col("internalSpeed").is_not_null())
        )

    _banner = (
        mo.md("> **Upload a `.sdfz` above to begin.**")
        if not uploader.value
        else None
    )
    _banner
    return (
        duration_s,
        fps_df,
        hardware_df,
        perf_df,
        players_all,
        players_enriched,
        replay_row,
        speed_df,
    )


@app.cell
def _(mo):
    # Spectators have cpu/fps telemetry (they run the full sim in BAR) so they
    # otherwise show up in the selector, summary and charts. Default to hiding
    # them; toggle on to fold them back into every player view below.
    include_spectators = mo.ui.checkbox(value=False, label="Include spectators")
    include_spectators
    return (include_spectators,)


@app.cell
def _(include_spectators, pl, players_all):
    players_df = (
        players_all
        if include_spectators.value
        else players_all.filter(~pl.col("is_spectator"))
    )
    return (players_df,)


@app.cell
def _(duration_s, mo, players_df, players_enriched, replay_row):
    def _fmt_dur(s):
        s = int(round(s))
        return f"{s // 60}:{s % 60:02d}"

    # players_enriched carries match-level fields (winning_team) on every row;
    # one row is enough to read them.
    pe_row = (
        players_enriched.row(0, named=True) if players_enriched.height else None
    )

    cards = [
        mo.stat(label="Map", value=replay_row.get("map") or "—"),
        mo.stat(label="Duration", value=_fmt_dur(duration_s)),
        mo.stat(label="Engine", value=replay_row.get("engine_version") or "—"),
        mo.stat(label="Players", value=str(players_df.height)),
    ]
    # Demo-derived fields: winning ally (from GAMEOVER) and start time (header).
    # old/new skill, is_ranked and match_id come only from the match API and
    # are unavailable when parsing a raw demo.
    if pe_row is not None and pe_row.get("winning_team") is not None:
        cards.append(
            mo.stat(label="Winning ally", value=str(pe_row["winning_team"]))
        )
    if replay_row.get("started"):
        cards.append(mo.stat(label="Started", value=replay_row["started"]))
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
        for row in players_df.sort(
            ["allyTeamId", "teamId", "playerNum"]
        ).iter_rows(named=True)
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
    _palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    ]
    # Order by team then in-game player id (not alphabetically) so the color
    # scale domain — and therefore every chart legend — groups teammates
    # together with the first player on the first team leading. Makes it easy
    # to read who's allied with whom straight off the legend.
    _names = (
        players_df.sort(["allyTeamId", "teamId", "playerNum"])["name"].to_list()
    )
    player_colors = {n: _palette[i % len(_palette)] for i, n in enumerate(_names)}
    return (player_colors,)


@app.cell
def _(alt, duration_s, fps_df, perf_df, pl, speed_df):
    # --- Game Health tab ---------------------------------------------------
    # Server speed timeline — only `internalSpeed` (userSpeed carries no signal).
    if speed_df.is_empty():
        speed_long = pl.DataFrame(
            {"t": [0.0, duration_s], "value": [1.0, 1.0],
             "signal": ["internalSpeed", "internalSpeed"]}
        )
    else:
        # Anchor at t=0 (full speed) and t=duration so the line spans the game.
        speed_long = (
            speed_df.rename({"internalSpeed": "value"})
            .with_columns(pl.col("value").cast(pl.Float64))
            .vstack(pl.DataFrame({"t": [0.0], "value": [1.0]}))
            .vstack(pl.DataFrame({"t": [duration_s], "value": [None]}))
            .sort("t")
            .with_columns(pl.col("value").fill_null(strategy="forward"))
            .with_columns(signal=pl.lit("internalSpeed"))
        )

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
                    domain=["internalSpeed"],
                    range=["#1f77b4"],
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
    # >=1 fps<=2 sample. These flag client-side "past the cliff" windows.
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

    _per_player_chart_height = max(280, 14 * len(selected_pids) + 60)

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
            height=_per_player_chart_height,
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
            height=_per_player_chart_height,
            width="container",
            padding={"left": 5, "top": 5, "bottom": 5, "right": 50},
            title=alt.TitleParams(
                "Render FPS per player",
                subtitle="Higher is better — FPS ≤ 2 means the engine hit its draw-floor",
                subtitleColor="#888",
            ),
        )
    )

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
    players_enriched,
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

    # players_enriched carries faction/country/skill from the start script at
    # the playerNum level for this replay.
    if players_enriched.height:
        skill_lookup = players_enriched.select(
            ["playerNum", "faction", "country", "skill"]
        )
        summary = summary.join(skill_lookup, on="playerNum", how="left")

    summary_sorted = summary.sort("mean_sim_ms", descending=True, nulls_last=True)

    summary_table = mo.ui.table(summary_sorted, pagination=False)

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
def _(hardware_df, mo):
    # --- Hardware tab ------------------------------------------------------
    # Per-player system info the client broadcast once at game start
    # (SYSTEM_INFO Lua message). Not available from ClickHouse — demo-only.
    if hardware_df.is_empty():
        hardware_view = mo.md("_No hardware info in this demo (or none uploaded yet)._")
    else:
        hardware_view = mo.ui.table(
            hardware_df.rename({
                "cpuCores": "cores",
                "logicalCpuCores": "threads",
                "gpuMemory": "vram",
                "windowMode": "window",
            }),
            pagination=False,
        )
    return (hardware_view,)


@app.cell
def _(hardware_view, health_chart, mo, summary_bar, summary_table, ts_charts):
    mo.ui.tabs(
        {
            "Time Series": mo.vstack([health_chart, ts_charts]),
            "Summary": mo.vstack([summary_table, summary_bar]),
            "Hardware": hardware_view,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
