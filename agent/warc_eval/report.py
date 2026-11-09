"""Strict complete-matrix aggregation and standalone HTML artifacts."""
import base64
import csv
import gzip
import hashlib
import html
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .core import digest, score_from_events

REPORT_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


STYLE = """body{font:15px system-ui;margin:30px auto;max-width:1250px;padding:0 20px;color:#17212b}
table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border-bottom:1px solid #dde3e9;padding:8px;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#eef3f7}a{color:#1765a1}small{color:#536373}
canvas{image-rendering:pixelated;border:1px solid #ccc}input[type=range]{width:100%}.error{color:#b92323}
details{margin:16px 0}svg{max-width:100%;height:auto}button,select{padding:5px;margin:5px}pre{white-space:pre-wrap}"""


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    """Hierarchical bootstrap: games, then independent runs within each game."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260914)
    g, s = array.shape
    gi = rng.integers(g, size=(2000, g))
    si = rng.integers(s, size=(2000, g, s))
    samples = array[gi[:, :, None], si].mean(axis=(1, 2))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def table(rows, columns):
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key, _ in columns) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def curve_svg(curves, budget, key, title, maximum):
    colors = ["#2874a6", "#dc7633", "#229954", "#884ea0", "#cb4335"]
    width, height, left, top, plotw, ploth = 850, 350, 65, 40, 590, 245
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="65" y="23" font-family="sans-serif" font-size="17">{html.escape(title)}</text>']
    for fraction in (0, .25, .5, .75, 1):
        y = top + ploth * (1-fraction)
        svg.append(f'<path d="M{left},{y}h{plotw}" stroke="#ddd"/><text x="55" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="12">{maximum*fraction:.2g}</text>')
        x = left + plotw * fraction
        svg.append(f'<text x="{x}" y="310" text-anchor="middle" font-family="sans-serif" font-size="12">{budget*fraction:.0f}</text>')
    for i, (agent, values) in enumerate(curves.items()):
        points = " ".join(f"{left+plotw*p['step']/budget:.2f},{top+ploth*(1-p[key]/maximum):.2f}" for p in values)
        color = colors[i % len(colors)]
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<text x="670" y="{65+i*24}" font-family="sans-serif" font-size="12" fill="{color}">{html.escape(agent)}</text>')
    svg.append('<text x="360" y="340" font-family="sans-serif" font-size="13">Cumulative environment operations (RESET included)</text></svg>')
    return "".join(svg)


def replay_page(result, encoded):
    label = html.escape(f"{result['agent']} | {result['game_id']} | seed {result['agent_seed']}")
    return f'''<!doctype html><meta charset="utf-8"><title>{label}</title><style>{STYLE}</style>
<a href="../../leaderboard.html">Leaderboard</a><h1>{label}</h1>
<p>Status: {html.escape(result['status'])}. Slider = environment operation; subframes are one response, not extra actions.</p>
<div style="display:flex;gap:20px;flex-wrap:wrap"><div>Before action<br><canvas id="before" width="64" height="64" style="width:448px;height:448px"></canvas></div>
<div>After action / response subframe<br><canvas id="board" width="64" height="64" style="width:448px;height:448px"></canvas></div></div>
<div><button id="prev">Previous</button><button id="play">Play / pause</button><button id="next">Next</button>
<select id="jump"><option value="0">Initial observation</option></select></div>
<input id="timeline" type="range" min="0" value="0" step="1"><div>Response subframe: <input id="subframe" type="range" min="0" value="0" step="1"></div>
<pre id="info">Loading compressed replay…</pre>
<script id="payload" type="text/plain">{encoded}</script>
<script>
(async()=>{{
const bytes=Uint8Array.from(atob(document.getElementById('payload').textContent.trim()),c=>c.charCodeAt(0));
const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
const text=await new Response(stream).text();
if(!text.trim()){{document.getElementById('info').textContent='No observation: initialization failed; see result.json';return}}
const events=text.trim().split('\\n').map(JSON.parse).filter(e=>e.observation);
const slider=document.getElementById('timeline'), sub=document.getElementById('subframe'), jump=document.getElementById('jump');
const palette=['#ffffff','#cccccc','#999999','#666666','#333333','#000000','#e53aa3','#ff7bcc','#f93c31','#1e93ff','#88d8f1','#ffdc00','#ff851b','#921231','#4fcc30','#a356d6'];
const context=document.getElementById('board').getContext('2d'); slider.max=events.length-1;
let timer=null,level=0;
events.forEach((e,i)=>{{if(e.observation.levels_completed>level){{level=e.observation.levels_completed;const opt=document.createElement('option');opt.value=i;opt.textContent='Cleared L'+level+' @ '+e.step;jump.appendChild(opt)}}}});
function draw(reset=true){{const e=events[+slider.value],frames=e.observation.frame;sub.max=frames.length-1;if(reset)sub.value=sub.max;
const before=events[Math.max(0,+slider.value-1)].observation.frame.at(-1),bc=document.getElementById('before').getContext('2d');
for(let y=0;y<before.length;y++)for(let x=0;x<before[y].length;x++){{bc.fillStyle=palette[before[y][x]];bc.fillRect(x,y,1,1)}}
const grid=frames[+sub.value];for(let y=0;y<grid.length;y++)for(let x=0;x<grid[y].length;x++){{context.fillStyle=palette[grid[y][x]];context.fillRect(x,y,1,1)}}
document.getElementById('info').textContent=JSON.stringify({{step:e.step,action:e.action||'INITIAL',state:e.observation.state,levels_completed:e.observation.levels_completed,subframe:+sub.value+1,response_frames:frames.length,wall_seconds:e.wall_seconds}},null,2)}}
slider.oninput=()=>draw();sub.oninput=()=>draw(false);jump.onchange=()=>{{slider.value=jump.value;draw()}};
function advance(n){{slider.value=Math.max(0,Math.min(events.length-1,+slider.value+n));draw()}}
document.getElementById('prev').onclick=()=>advance(-1);document.getElementById('next').onclick=()=>advance(1);
document.getElementById('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null}}else timer=setInterval(()=>{{advance(1);if(+slider.value===events.length-1){{clearInterval(timer);timer=null}}}},150)}};
document.onkeydown=e=>{{if(e.key==='ArrowRight')advance(1);if(e.key==='ArrowLeft')advance(-1)}};draw();
}})().catch(e=>document.getElementById('info').textContent=String(e));
</script>'''


def report(output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    if digest({k: v for k, v in config.items() if k != "config_id"}) != config["config_id"]:
        raise ValueError("Configuration checksum mismatch")
    l1_threshold = min(1000, config["budget"])
    game_l1_key = f"l1_success_at_{l1_threshold}"
    games = {g["game_id"]: g for g in config["games"]}
    results = {}
    for file in sorted((output / "runs").glob("*/result.json")):
        result = json.loads(file.read_text())
        key = (result["agent"], result["game_id"], result["agent_seed"])
        if result["config_id"] != config["config_id"] or key in results:
            raise ValueError("Mixed configuration or duplicate result")
        compressed = (file.parent / "events.jsonl.gz").read_bytes()
        if hashlib.sha256(compressed).hexdigest() != result["events_sha256"]:
            raise ValueError(f"Event log checksum mismatch: {file}")
        events = [json.loads(line) for line in gzip.decompress(compressed).decode().splitlines()]
        if events:
            rescored = score_from_events(events, games[result["game_id"]]["baseline_actions"])
            if rescored != result["metrics"]:
                raise ValueError(f"Event/result inconsistency: {file}")
        elif result["metrics"] is not None:
            raise ValueError("Metrics without event evidence")
        (file.parent / "replay.html").write_text(replay_page(result, base64.b64encode(compressed).decode()))
        result["replay"] = str((file.parent / "replay.html").relative_to(output))
        results[key] = result
    expected = set(itertools.product(config["agents"], games, config["seeds"]))
    if set(results) != expected:
        raise ValueError(f"Incomplete evaluation matrix: expected {len(expected)}, found {len(results)}")
    summaries, per_game, per_level, all_runs, curves, arrays = [], [], [], [], {}, {}
    checkpoints = sorted({0, config["budget"], *[n for n in (100, 200, 500, 1000) if n <= config["budget"]]})
    curve_steps = sorted(set(np.linspace(0, config["budget"], 101, dtype=int)) | set(checkpoints))
    for agent in config["agents"]:
        rows = [[results[agent, game, seed] for seed in config["seeds"]] for game in games]
        flat = sum(rows, [])
        def l1(r, n):
            step = (r["metrics"] or {}).get("l1_step")
            return float(step is not None and step <= n)
        def metric(r, k):
            value = (r["metrics"] or {}).get(k, 0)
            return 0 if value is None else value
        l1_array = np.array([[l1(r, min(1000, config["budget"])) for r in row] for row in rows])
        score_array = np.array([[metric(r, "rhae") for r in row] for row in rows])
        arrays[agent] = {"l1": l1_array, "rhae": score_array}
        summary = {"agent": agent, "runs": len(flat),
                   **{f"success_at_{n}": float(np.mean([l1(r, n) for r in flat])) for n in checkpoints if n},
                   "rhae": float(score_array.mean()), "rhae_ci95": interval(score_array),
                   "l1_ci95_at": min(1000, config["budget"]), "l1_ci95": interval(l1_array),
                   "mean_levels": float(np.mean([metric(r, "levels_completed") for r in flat])),
                   "full_clear_rate": float(np.mean([metric(r, "win") for r in flat])),
                   "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in flat])),
                   "max_agent_rss_mib": max(r["agent_peak_rss_mib"] or 0 for r in flat),
                   "total_gpu_seconds": 0 if all(r["gpu_seconds"] == 0 for r in flat) else None,
                   "status_counts": dict(Counter(r["status"] for r in flat))}
        summaries.append(summary)
        curves[agent] = [{"step": int(n), "l1_success": float(np.mean([l1(r, n) for r in flat])),
                          "mean_levels": float(np.mean([sum(s <= n for s in metric(r, "completion_steps")) if r["metrics"] else 0 for r in flat]))}
                         for n in curve_steps]
        for game, row in zip(games, rows):
            per_game.append({"agent": agent, "game_id": game,
                             game_l1_key: float(np.mean([l1(r, l1_threshold) for r in row])),
                             "rhae": float(np.mean([metric(r, "rhae") for r in row])),
                             "mean_levels": float(np.mean([metric(r, "levels_completed") for r in row]))})
            for level in range(1, len(games[game]["baseline_actions"]) + 1):
                per_level.append({"agent": agent, "game_id": game, "level": level,
                                  "reach_rate": float(np.mean([r["metrics"] is not None and metric(r, "levels_completed") >= level-1 for r in row])),
                                  "success_rate": float(np.mean([metric(r, "levels_completed") >= level for r in row]))})
        for r in flat:
            all_runs.append({"agent": agent, "game_id": r["game_id"], "seed": r["agent_seed"],
                             "status": r["status"], "steps": metric(r, "steps"),
                             "l1_step": (r["metrics"] or {}).get("l1_step"),
                             "levels": metric(r, "levels_completed"), "rhae": metric(r, "rhae"),
                             "wall_seconds": r["wall_seconds"], "replay": r["replay"]})
    pairs = []
    for a, b in itertools.combinations(config["agents"], 2):
        for key in ("l1", "rhae"):
            delta = arrays[b][key] - arrays[a][key]
            pairs.append({"comparison": f"{b} - {a}", "metric": key, "mean_delta": float(delta.mean()), "ci95": interval(delta)})
    aggregate = {"config_id": config["config_id"],
                 "report_source_sha256": REPORT_SOURCE_SHA256,
                 "summary": summaries, "paired": pairs,
                 "curves": curves, "per_game": per_game, "per_level": per_level,
                 "ci_method": "2000 hierarchical paired bootstrap draws; games then seeds; 95% percentile; descriptive public-set uncertainty"}
    (output / "leaderboard.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    for name, rows in (("summary", summaries), ("per_game", per_game), ("per_level", per_level), ("runs", all_runs), ("paired", pairs)):
        write_csv(output / f"{name}.csv", rows)
    l1_svg = curve_svg(curves, config["budget"], "l1_success", "L1 success vs. interaction budget", 1)
    depth_svg = curve_svg(curves, config["budget"], "mean_levels", "Mean completed levels vs. interaction budget",
                          max(1, max(p["mean_levels"] for v in curves.values() for p in v)))
    (output / "l1_success.svg").write_text(l1_svg)
    (output / "levels.svg").write_text(depth_svg)
    display = [{"agent": s["agent"], **{f"s{n}": f"{100*s[f'success_at_{n}']:.1f}%" for n in checkpoints if n},
                "rhae": f"{s['rhae']:.2f}", "levels": f"{s['mean_levels']:.2f}", "wall": f"{s['mean_wall_seconds']:.1f}",
                "status": str(s["status_counts"])} for s in summaries]
    order = {s["agent"]: (-s[f"success_at_{min(1000, config['budget'])}"] if min(1000, config['budget']) in checkpoints else 0,
                           -s["rhae"]) for s in summaries}
    display.sort(key=lambda row: order[row["agent"]])
    columns = [("agent", "Agent")] + [(f"s{n}", f"L1 @{n}") for n in checkpoints if n] + [("rhae", "RHAE /100"), ("levels", "Mean levels"), ("wall", "sec/run"), ("status", "Status")]
    run_links = "".join(f'<li>{html.escape(r["agent"])} | {html.escape(r["game_id"])} | seed {r["seed"]}: '
                        f'<a href="{html.escape(r["replay"])}">replay</a> — {html.escape(r["status"])}, L{r["levels"]}</li>' for r in all_runs)
    page = f'''<!doctype html><meta charset="utf-8"><title>WARC pub25 leaderboard</title><style>{STYLE}</style>
<h1>WARC · public-set online evaluation</h1><p>{len(games)} games × {len(config['seeds'])} seeds × {len(config['agents'])} agents;
mode={config['mode']}; budget={config['budget']} operations/game; run timeout={config['run_timeout_seconds']}s.</p>
<p>Latest SDK at snapshot: arc-agi 0.9.9 / arcengine 0.9.3. Local official scoring, not an official leaderboard submission.
Public development set, not an unseen-game generalization claim. All rows use the same protocol; no best-of-seed selection.</p>
{table(display, columns)}{l1_svg}{depth_svg}
<details><summary>Uncertainty and compute</summary>{table([
 {'agent': s['agent'], 'l1': f"[{s['l1_ci95'][0]:.3f}, {s['l1_ci95'][1]:.3f}]",
 'rhae': f"[{s['rhae_ci95'][0]:.2f}, {s['rhae_ci95'][1]:.2f}]", 'rss': f"{s['max_agent_rss_mib']:.1f}",
 'gpu': s['total_gpu_seconds']} for s in summaries],
 [('agent','Agent'),('l1','L1 @min(1000,budget) CI95'),('rhae','RHAE CI95'),('rss','Max agent CPU RSS MiB'),('gpu','GPU seconds')])}
<p>Hierarchical bootstrap over games then paired seeds (2,000 draws). GPU metrics for external agents are null unless instrumented.
Ordering: L1 @min(1000,budget), then RHAE. Rows with errors must be inspected before drawing conclusions.</p></details>
<p><a href="leaderboard.json">Full metrics + hierarchical CIs + paired differences (JSON)</a> ·
<a href="config.json">Protocol manifest</a> · <a href="runs.csv">Run CSV</a> · <a href="per_level.csv">Per-level CSV</a></p>
<details><summary>Game-level results</summary>{table(per_game, [('agent','Agent'),('game_id','Game'),(game_l1_key,f'L1 @{l1_threshold}'),('rhae','RHAE'),('mean_levels','Mean levels')])}</details>
<details><summary>All run replays</summary><ul>{run_links}</ul></details>'''
    (output / "leaderboard.html").write_text(page)
    from .game_report import game_report
    aggregate["per_game"] = game_report(output)
    print(f"Verified {len(results)} runs; leaderboard: {output / 'leaderboard.html'}", flush=True)
    return aggregate
