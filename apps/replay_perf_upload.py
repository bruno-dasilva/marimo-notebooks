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

__generated_with = "0.23.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""
    # Replay performance — upload a `.sdfz`

    Drop a Beyond All Reason demo below. It's parsed **entirely in your
    browser** (no upload to any server) to show per-player sim-frame timing,
    render FPS, server speed, a whole-game summary, and the hardware each
    player reported.
    """)
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
    PAUSE, STARTPLAYING, INTERNAL_SPEED, GAMEOVER, PLAYERINFO, LUAMSG = (
        13, 4, 20, 30, 38, 50)

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
    _PAUSE_SCHEMA = {"t": pl.Float64}
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
    # Real measured sim/draw frame times broadcast by the engine-side gadget
    # game_frametime_broadcast.lua ("#ft<simN>/<drawN>/<avgSim>/<simPeak>/
    # <avgDraw>/<drawPeak>", every 2s). Present only in recent BAR builds.
    _FT_SCHEMA = {
        "playerNum": pl.Int64, "t": pl.Float64, "simN": pl.Int64,
        "drawN": pl.Int64, "avgSim": pl.Float64, "simPeak": pl.Float64,
        "avgDraw": pl.Float64, "drawPeak": pl.Float64,
    }

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
        # BAR game version, e.g. "Beyond All Reason test-28379-33ba377"
        # (locally recorded demos may carry the literal "$VERSION").
        game_version = game.get("gametype")
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
        return map_name, game_version, players

    def parse_demo_bytes(raw):
        sdf = gzip.decompress(raw)
        r = _Reader(sdf)
        h = _parse_header(r)
        r.off = h["headerSize"]
        script_text = r.read(h["scriptSize"]).decode("utf-8", "replace")
        stream = r.read(h["demoStreamSize"])
        map_name, game_version, players = _build_players(_parse_tdf(script_text))

        # columnar accumulators — avoid per-row tuple/object overhead
        c_pid, c_t, c_sim, c_q, c_isp, c_cpu, c_ping = [], [], [], [], [], [], []
        f_pid, f_t, f_fps = [], [], []
        ft_rows = []
        pause_t = []  # game time of each pause-start (NETMSG_PAUSE, bPaused=1)
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
                # cpuUsage-derived approximation (the Time Series tab). The real
                # engine-measured sim/draw times live in frametime_df and are
                # shown on the Frame Time tab instead.
                sim, qual = _derive_sim_frame_ms(
                    cpu_usage, internal_speed, last_fps.get(player_num))
                ap_pid(player_num); ap_t(t); ap_sim(sim); ap_q(qual)
                ap_isp(internal_speed); ap_cpu(cpu_usage); ap_ping(ping)
            elif pid == INTERNAL_SPEED:
                internal_speed = _F32.unpack_from(buf, off + 1)[0]
            elif pid == PAUSE:  # [playerNum u8, bPaused u8]; record pause starts
                if buf[off + 2] == 1:
                    pause_t.append(t)
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
                    elif msg[:3] == b"#ft":  # FRAME_TIME (real measured sim/draw ms)
                        pn = buf[off + 3]
                        try:
                            v = [float(x) for x in
                                 msg.decode("latin-1")[3:].split("/")]
                        except ValueError:
                            v = None
                        if v and len(v) >= 6:
                            ft_rows.append(
                                (pn, t, int(v[0]), int(v[1]),
                                 v[2], v[3], v[4], v[5]))
                    elif msg[:3] == b"$y$":  # SYSTEM_INFO
                        pn = buf[off + 3]
                        if pn not in hardware:
                            m = _SYSTEM_INFO_RE.match(msg.decode("latin-1")[5:])
                            if m:
                                hardware[pn] = m.groupdict()
            off += length

        if duration_s is None:
            # No GAMEOVER: prefer wallclockTime (demo-parser.ts), but locally
            # recorded skirmish demos carry wallclockTime=0, so fall back to the
            # last observed event time.
            last_event_t = max(
                (acc[-1] for acc in (c_t, f_t) if acc),
                default=(ft_rows[-1][1] if ft_rows else 0.0))
            duration_s = float(h["wallclockTime"]) or last_event_t
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

        frametime_df = pl.DataFrame(
            {"playerNum": [r[0] for r in ft_rows], "t": [r[1] for r in ft_rows],
             "simN": [r[2] for r in ft_rows], "drawN": [r[3] for r in ft_rows],
             "avgSim": [r[4] for r in ft_rows], "simPeak": [r[5] for r in ft_rows],
             "avgDraw": [r[6] for r in ft_rows], "drawPeak": [r[7] for r in ft_rows]},
            schema=_FT_SCHEMA,
        )

        # Pause-start markers (game time). Sim time freezes during a pause
        # (the chunk timestamp is frame-number-derived: startTime + frame/30,
        # NetProtocol.cpp:GetPacketTime), so a pause is a single instant on the
        # game-time axis — we mark where, not how long (real duration isn't
        # recoverable from a client demo).
        pause_df = pl.DataFrame({"t": pause_t}, schema=_PAUSE_SCHEMA).unique().sort("t")

        replay_row = {
            "map": map_name, "engine_version": h["versionString"],
            "bar_build": game_version, "duration_s": duration_s,
            "frame_rate": GAME_SPEED, "started": started,
        }
        return {
            "replay_row": replay_row, "frame_rate": GAME_SPEED,
            "duration_s": duration_s, "perf_df": perf_df, "fps_df": fps_df,
            "players_all": players_all, "players_enriched": players_enriched,
            "hardware_df": hardware_df, "frametime_df": frametime_df,
            "pause_df": pause_df,
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
            "frametime_df": pl.DataFrame(schema=_FT_SCHEMA),
            "pause_df": pl.DataFrame(schema=_PAUSE_SCHEMA),
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
    frametime_df = _res["frametime_df"]
    pause_df = _res["pause_df"]

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
        frametime_df,
        hardware_df,
        pause_df,
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

    # BAR game version from the start script's gametype, trimmed of the
    # "Beyond All Reason " prefix (e.g. "test-28379-33ba377").
    _bar_version = (replay_row.get("bar_build") or "").removeprefix(
        "Beyond All Reason ").strip() or "—"

    cards = [
        mo.stat(label="Map", value=replay_row.get("map") or "—"),
        mo.stat(label="Duration", value=_fmt_dur(duration_s)),
        mo.stat(label="BAR version", value=_bar_version),
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
    # Which measured frame-time stat(s) to draw on the Frame Time tab.
    stat_filter = mo.ui.multiselect(
        options=["mean", "peak"],
        value=["mean", "peak"],
        label="Frame time stat",
    )
    mo.vstack([smoothing, player_selector, stat_filter])
    return player_selector, smoothing, stat_filter


@app.cell
def _(players_df):
    # Bright, well-separated hues tuned for a dark background (Material 300/400
    # level) — the default matplotlib palette muddies on dark. Ordered so
    # adjacent entries (often teammates after the sort below) stay distinct.
    _palette = [
        "#4FC3F7", "#FF8A65", "#81C784", "#BA68C8", "#FFD54F", "#4DD0E1",
        "#F06292", "#AED581", "#9575CD", "#FFB74D", "#4DB6AC", "#E57373",
        "#7986CB", "#DCE775", "#A1887F", "#90A4AE",
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
def _(alt, frametime_df, mo, pause_df, perf_df, pl):
    # --- Zoom scrubbers ----------------------------------------------------
    # One thin overview strip per tab. Drag a horizontal interval on it to pick
    # a time window; the detail charts below read that range (zoom_domain) and
    # clamp their x-axis to it. Double-click the strip to clear/reset.
    # (An interval brush can't rescale its own chart in Vega, so the detail
    #  charts can't self-zoom — hence the dedicated scrubber.)
    def _scrubber(df, value_col, label):
        brush = alt.selection_interval(encodings=["x"])
        if df.is_empty():
            data = pl.DataFrame({"t": [0.0], "v": [0.0]})
        else:
            data = (
                df.filter(pl.col(value_col).is_not_null())
                .group_by((pl.col("t") // 1).alias("t"))
                .agg(v=pl.col(value_col).mean())
                .sort("t")
            )
        chart = (
            alt.Chart(data)
            .mark_area(opacity=0.4, color="#1f77b4", interpolate="monotone")
            .encode(
                x=alt.X("t:Q", title=label),
                y=alt.Y("v:Q", axis=None, title=None),
            )
            .properties(height=58, width="container")
            .add_params(brush)
        )
        return mo.ui.altair_chart(chart, legend_selection=False)

    def zoom_domain(scrubber):
        # The brushed window as [t0, t1]; None when nothing (or everything) is
        # selected, in which case detail charts use their full auto domain.
        try:
            ts = scrubber.value["t"]
            if len(ts) == 0:
                return None
            lo, hi = float(ts.min()), float(ts.max())
        except Exception:
            return None
        return [lo, hi] if hi > lo else None

    def pause_marks(dom, xscale):
        # Dashed vertical rules at each pause (game time). A pause is a single
        # instant on this axis — sim time freezes while paused — so we mark
        # where, not for how long. Returns a layerable Chart, or None.
        if pause_df.is_empty():
            return None
        pm = pause_df
        if dom:
            pm = pm.filter(pl.col("t").is_between(dom[0], dom[1]))
        if pm.is_empty():
            return None
        return (
            alt.Chart(pm)
            .mark_rule(color="#B0BEC5", strokeDash=[4, 3], strokeWidth=1.2,
                       opacity=0.8, clip=True)
            .encode(
                x=alt.X("t:Q", scale=xscale),
                tooltip=[alt.Tooltip("t:Q", format=".0f", title="pause @ game s")],
            )
        )

    _label = "drag to select a time window · double-click to reset"
    # One scrubber drives every chart on the combined Performance tab. Prefer
    # the measured avgSim series; fall back to the perf approximation so the
    # strip still renders for demos with no #ft broadcasts.
    if frametime_df.is_empty():
        scrubber = _scrubber(perf_df, "simFrameMs", _label)
    else:
        scrubber = _scrubber(frametime_df, "avgSim", _label)
    return pause_marks, scrubber, zoom_domain


@app.cell
def _(alt, duration_s, mo, pause_marks, pl, scrubber, speed_df, zoom_domain):
    # --- Sim speed (shown on the Performance tab) --------------------------
    _dom = zoom_domain(scrubber)
    _xscale = alt.Scale(domain=_dom) if _dom else alt.Undefined
    # Server/sim speed timeline — only `internalSpeed` (userSpeed carries no signal).
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

    # Clip the step series to the zoom window like the other charts (filter the
    # data rather than relying on a layered scale domain). Carry the speed
    # active at the window's start, and extend the last value to the window's
    # end so the step line spans edge-to-edge even with no changes inside.
    if _dom:
        _t0, _t1 = _dom
        _before = speed_long.filter(pl.col("t") <= _t0).tail(1)
        _win = speed_long.filter((pl.col("t") > _t0) & (pl.col("t") < _t1))
        _start = (float(_before["value"][-1]) if _before.height
                  else (float(_win["value"][0]) if _win.height else 1.0))
        _end = float(_win["value"][-1]) if _win.height else _start
        speed_plot = pl.concat([
            pl.DataFrame({"t": [float(_t0)], "value": [_start],
                          "signal": ["internalSpeed"]}),
            _win.select("t", "value", "signal"),
            pl.DataFrame({"t": [float(_t1)], "value": [_end],
                          "signal": ["internalSpeed"]}),
        ]).sort("t")
    else:
        speed_plot = speed_long

    # Dynamic y: floor at 0, top at the fastest speed visible (never below 1.05
    # so a steady-1x game still reads sensibly). Handles fast-forward up to ~20x.
    _ymax = max(1.05, float(speed_plot["value"].max()))

    health_lines = (
        alt.Chart(speed_plot)
        .mark_line(
            interpolate="step-after", strokeWidth=1.6, clip=True, color="#FF5252"
        )
        .encode(
            x=alt.X("t:Q", title="Game time (sim s)", scale=_xscale),
            y=alt.Y(
                "value:Q",
                title="Speed multiplier",
                scale=alt.Scale(domain=[0, _ymax]),
            ),
            tooltip=[
                alt.Tooltip("t:Q", format=".1f"),
                alt.Tooltip("value:Q", format=".3f", title="speed"),
            ],
        )
    )
    # Layer dashed pause markers; width must sit on the outer layered chart.
    _rule = pause_marks(_dom, _xscale)
    _speed = health_lines if _rule is None else alt.layer(health_lines, _rule)
    health_chart = mo.ui.altair_chart(
        _speed.properties(
            height=300,
            width="container",
            padding={"left": 5, "top": 5, "bottom": 5, "right": 50},
            title="Sim speed",
        ),
        chart_selection=False, legend_selection=False,
    )
    return (health_chart,)


@app.cell
def _(
    alt,
    fps_df,
    frametime_df,
    mo,
    pause_marks,
    perf_df,
    pl,
    player_colors,
    player_selector,
    players_df,
    scrubber,
    smoothing,
    stat_filter,
    zoom_domain,
):
    # --- Combined performance charts (sim, draw, FPS) ---------------------
    # Sim timing prefers the engine-measured broadcasts
    # (game_frametime_broadcast.lua, recent BAR builds); when a demo carries
    # none it falls back to the cpuUsage-derived approximation. Draw timing
    # exists only in the measured format. FPS is always shown. The sim-speed
    # chart is built in its own (health_chart) cell, appended after these.
    _dom = zoom_domain(scrubber)
    _xscale = alt.Scale(domain=_dom) if _dom else alt.Undefined
    _sel = list(player_selector.value)
    _stats = list(stat_filter.value)  # subset of {"mean", "peak"} for measured
    _measured = not frametime_df.is_empty()
    _color_scale = alt.Scale(
        domain=list(player_colors.keys()),
        range=list(player_colors.values()),
    )
    _height = max(280, 14 * len(_sel) + 60)
    _XTITLE = "Game time (sim s)"

    # One smoothing rule for every per-player line chart: average each series
    # into `win`-second tumbling bins (mean per bin), keyed on `by`. Applied
    # identically to measured sim/draw, approx sim, and FPS so the slider
    # behaves the same everywhere.
    _win = max(2, int(smoothing.value) or 2)

    def _smooth(df, value_col, by):
        if df.is_empty():
            return df
        return (
            df.with_columns(t_bin=(pl.col("t") // _win * _win))
            .group_by([*by, "t_bin"])
            .agg(pl.col(value_col).mean())
            .rename({"t_bin": "t"})
            .sort([*by, "t"])
        )

    def _finalize(base, height, title, subtitle):
        # Layer dashed pause markers under the title/width wrapper. width must
        # live on the outer (layered) chart, not the inner marks, to keep the
        # container-responsive sizing working.
        rule = pause_marks(_dom, _xscale)
        layered = base if rule is None else alt.layer(base, rule)
        chart = layered.properties(
            height=height, width="container",
            padding={"left": 5, "top": 5, "bottom": 5, "right": 50},
            title=alt.TitleParams(title, subtitle=subtitle, subtitleColor="#888"),
        )
        return mo.ui.altair_chart(
            chart, chart_selection=False, legend_selection=False)

    def _measured_base(avg_col, peak_col):
        # avg (solid) + peak (dashed), filtered by the mean/peak stat selector.
        ft = (
            frametime_df.filter(pl.col("playerNum").is_in(_sel))
            .join(players_df.select(["playerNum", "name"]), on="playerNum")
        )
        if _dom:
            ft = ft.filter(pl.col("t").is_between(_dom[0], _dom[1]))
        long = (
            ft.select("name", "t", mean=pl.col(avg_col), peak=pl.col(peak_col))
            .unpivot(
                index=["name", "t"], on=["mean", "peak"],
                variable_name="stat", value_name="ms",
            )
            .filter(pl.col("stat").is_in(_stats))
        )
        long = _smooth(long, "ms", ["name", "stat"])
        return (
            alt.Chart(long)
            .mark_line(opacity=0.8, strokeWidth=1.3, clip=True)
            .encode(
                x=alt.X("t:Q", title=_XTITLE, scale=_xscale),
                y=alt.Y("ms:Q", title="Frame time (ms)"),
                color=alt.Color(
                    "name:N", scale=_color_scale, title="Player",
                    legend=alt.Legend(orient="right", labelLimit=140),
                ),
                strokeDash=alt.StrokeDash(
                    "stat:N", title=None,
                    scale=alt.Scale(domain=["mean", "peak"],
                                    range=[[1, 0], [4, 3]]),
                ),
                tooltip=[
                    alt.Tooltip("name:N", title="Player"),
                    alt.Tooltip("t:Q", format=".0f", title="t (s)"),
                    alt.Tooltip("stat:N"),
                    alt.Tooltip("ms:Q", format=".1f"),
                ],
            )
        )

    def _approx_sim_base():
        # cpuUsage-derived fallback; trustworthy quality bands only, smoothed.
        clean = perf_df.filter(
            pl.col("simFrameMsQuality").is_in(("exact", "approx"))
            & pl.col("simFrameMs").is_not_null()
            & pl.col("playerNum").is_in(_sel)
        )
        if _dom:
            clean = clean.filter(pl.col("t").is_between(_dom[0], _dom[1]))
        plot = (
            _smooth(clean, "simFrameMs", ["playerNum"])
            .join(players_df.select(["playerNum", "name"]), on="playerNum")
        ) if not clean.is_empty() else pl.DataFrame()
        return (
            alt.Chart(plot)
            .mark_line(opacity=0.75, strokeWidth=1.2, clip=True)
            .encode(
                x=alt.X("t:Q", title=_XTITLE, scale=_xscale),
                y=alt.Y("simFrameMs:Q", title="Sim frame (ms)"),
                color=alt.Color(
                    "name:N", scale=_color_scale, title="Player",
                    legend=alt.Legend(orient="right", labelLimit=140),
                ),
                tooltip=[
                    alt.Tooltip("name:N", title="Player"),
                    alt.Tooltip("t:Q", format=".0f", title="t (s)"),
                    alt.Tooltip("simFrameMs:Q", format=".2f"),
                ],
            )
        )

    def _fps_base():
        clean = fps_df.filter(pl.col("playerNum").is_in(_sel))
        if _dom:
            clean = clean.filter(pl.col("t").is_between(_dom[0], _dom[1]))
        plot = (
            _smooth(clean, "fps", ["playerNum"])
            .join(players_df.select(["playerNum", "name"]), on="playerNum")
        ) if not clean.is_empty() else pl.DataFrame()
        return (
            alt.Chart(plot)
            .mark_line(opacity=0.75, strokeWidth=1.2, clip=True)
            .encode(
                x=alt.X("t:Q", title=_XTITLE, scale=_xscale),
                y=alt.Y("fps:Q", title="Render FPS"),
                color=alt.Color(
                    "name:N", scale=_color_scale, title="Player",
                    legend=alt.Legend(orient="right", labelLimit=140),
                ),
                tooltip=[
                    alt.Tooltip("name:N", title="Player"),
                    alt.Tooltip("t:Q", format=".0f", title="t (s)"),
                    alt.Tooltip("fps:Q", format=".0f"),
                ],
            )
        )

    _charts = []
    if _measured:
        _charts.append(_finalize(
            _measured_base("avgSim", "simPeak"), 300, "Measured sim frame time",
            "Engine profiler — avg (solid) & peak (dashed) ms per sim frame"))
        _charts.append(_finalize(
            _measured_base("avgDraw", "drawPeak"), 300, "Measured draw frame time",
            "Engine profiler — avg (solid) & peak (dashed) ms per draw frame"))
    else:
        _charts.append(_finalize(
            _approx_sim_base(), _height, "Sim frame timing per player (approx.)",
            "Lower is better — ms per sim frame, approximated from CPU load "
            "(no measured broadcasts in this demo)."))
    _charts.append(_finalize(
        _fps_base(), _height, "Render FPS per player",
        "Higher is better — FPS ≤ 2 means the engine hit its draw-floor"))

    perf_view = mo.vstack(_charts)
    return (perf_view,)


@app.cell
def _(
    alt,
    fps_df,
    frametime_df,
    mo,
    perf_df,
    pl,
    player_colors,
    players_df,
    players_enriched,
):
    # --- Whole-game summary tab -------------------------------------------
    perf_for_summary = perf_df.filter(
        pl.col("simFrameMsQuality").is_in(("exact", "approx"))
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
    _players = players_df.select(["playerNum", "name", "allyTeamId"])

    def _bar(src, title, x_title, peak_title):
        # Horizontal mean-per-player bar; `src` carries `value` (+ `peak`).
        d = (
            src.join(_players, on="playerNum")
            .drop_nulls("value")
            .sort("value", descending=True, nulls_last=True)
        )
        chart = (
            alt.Chart(d)
            .mark_bar(opacity=0.85)
            .encode(
                y=alt.Y("name:N", sort="-x", title="Player"),
                x=alt.X("value:Q", title=x_title),
                color=alt.Color("name:N", scale=_color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("name:N", title="Player"),
                    alt.Tooltip("allyTeamId:N", title="Ally team"),
                    alt.Tooltip("value:Q", format=".2f", title=x_title),
                    alt.Tooltip("peak:Q", format=".2f", title=peak_title),
                ],
            )
            .properties(
                height=24 * max(1, d.height), width="container", title=title)
        )
        return mo.ui.altair_chart(
            chart, chart_selection=False, legend_selection=False)

    # Sim bar: prefer engine-measured avgSim/simPeak; fall back to the cpuUsage
    # approximation (mean simFrameMs / p95) when a demo has no #ft broadcasts.
    if not frametime_df.is_empty():
        _sim_src = frametime_df.group_by("playerNum").agg(
            value=pl.col("avgSim").mean(), peak=pl.col("simPeak").max())
        sim_bar = _bar(
            _sim_src, "Sim frame time per player (measured)",
            "Mean sim frame (ms)", "peak sim ms")
    else:
        _sim_src = (
            perf_for_summary.group_by("playerNum").agg(
                value=pl.col("simFrameMs").mean(),
                peak=pl.col("simFrameMs").quantile(0.95))
            if not perf_for_summary.is_empty()
            else pl.DataFrame(schema={
                "playerNum": pl.Int64, "value": pl.Float64, "peak": pl.Float64})
        )
        sim_bar = _bar(
            _sim_src, "Sim frame time per player (approx)",
            "Mean sim frame (ms)", "p95 sim ms")

    # Draw bar: measured only — there's no cpuUsage approximation for draw time.
    if not frametime_df.is_empty():
        _draw_src = frametime_df.group_by("playerNum").agg(
            value=pl.col("avgDraw").mean(), peak=pl.col("drawPeak").max())
        draw_view = _bar(
            _draw_src, "Draw frame time per player (measured)",
            "Mean draw frame (ms)", "peak draw ms")
    else:
        draw_view = mo.md(
            "_No measured draw-frame data in this demo — draw timing only "
            "exists in recent BAR builds (`game_frametime_broadcast.lua`)._"
        )
    return draw_view, sim_bar, summary_table


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
def _(
    draw_view,
    hardware_view,
    health_chart,
    mo,
    perf_view,
    scrubber,
    sim_bar,
    summary_table,
):
    mo.ui.tabs(
        {
            "Performance": mo.vstack(
                [scrubber, perf_view, health_chart]
            ),
            "Summary": mo.vstack([summary_table, sim_bar, draw_view]),
            "Hardware": hardware_view,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
