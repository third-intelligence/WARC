"""Game-wise tables derived from a completed, event-verified report."""
import csv
import hashlib
import html
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .core import digest


def game_report(output):
    from .report import STYLE, table, write_csv
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    board = json.loads((output / "leaderboard.json").read_text())
    if digest({k: v for k, v in config.items() if k != "config_id"}) != config["config_id"] or board["config_id"] != config["config_id"]:
        raise ValueError("Game report configuration mismatch")
    with (output / "runs.csv").open() as file:
        rows = list(csv.DictReader(file))
    expected = set(itertools.product(config["agents"], [g["game_id"] for g in config["games"]], config["seeds"]))
    if len(rows) != len(expected) or {(r["agent"], r["game_id"], int(r["seed"])) for r in rows} != expected:
        raise ValueError("Incomplete game report matrix")
    points = sorted({config["budget"], *[n for n in (100, 200, 500, 1000) if n <= config["budget"]]})
    aggregate, sections = [], []
    for game in config["games"]:
        performance, compute = [], []
        for agent in config["agents"]:
            runs = [r for r in rows if r["game_id"] == game["game_id"] and r["agent"] == agent]
            details = [json.loads((output / r["replay"]).with_name("result.json").read_text()) for r in runs]
            # Task metrics come from the event-verified CSV. Check raw-result identity
            # before attaching resource diagnostics (not derived from transitions).
            for r, d in zip(runs, details):
                if (d["config_id"], d["agent"], d["game_id"], d["agent_seed"], d["status"]) != (
                        config["config_id"], agent, game["game_id"], int(r["seed"]), r["status"]):
                    raise ValueError("Game report result identity mismatch")
                m = d["metrics"] or {}
                if (int(r["steps"]), int(r["levels"]), float(r["rhae"]),
                    int(r["l1_step"]) if r["l1_step"] else None, float(r["wall_seconds"])) != (
                        m.get("steps", 0), m.get("levels_completed", 0), m.get("rhae") or 0,
                        m.get("l1_step"), d["wall_seconds"]):
                    raise ValueError("Game report CSV/result inconsistency")
            status = Counter(r["status"] for r in runs)
            gpu = [d.get("gpu_seconds") for d in details]
            cpu = [d.get("agent_cpu_seconds") for d in details]
            row = {"agent": agent, "game_id": game["game_id"], "runs": len(runs),
                   **{f"l1_success_at_{n}": mean(bool(r["l1_step"]) and int(r["l1_step"]) <= n for r in runs) for n in points},
                   "rhae": mean(float(r["rhae"]) for r in runs),
                   "mean_levels": mean(int(r["levels"]) for r in runs),
                   "full_clear_rate": mean(int(r["levels"]) == len(game["baseline_actions"]) for r in runs),
                   "mean_wall_seconds": mean(float(r["wall_seconds"]) for r in runs),
                   "mean_agent_cpu_seconds": mean(cpu) if all(v is not None for v in cpu) else None,
                   "total_gpu_seconds": sum(gpu) if all(v is not None for v in gpu) else None,
                   "max_agent_rss_mib": max(d.get("agent_peak_rss_mib") or 0 for d in details),
                   "timeout_runs": status["timeout"],
                   "error_runs": sum(v for k, v in status.items() if k not in ("timeout", "budget_exhaustion", "win", "l1_complete"))}
            aggregate.append(row)
            performance.append({"agent": agent, **{f"s{n}": f"{100*row[f'l1_success_at_{n}']:.1f}%" for n in points},
                                "rhae": f"{row['rhae']:.4f}", "levels": f"{row['mean_levels']:.2f}",
                                "win": f"{100*row['full_clear_rate']:.1f}%"})
            compute.append({"agent": agent, "wall": f"{row['mean_wall_seconds']:.1f}",
                            "rss": f"{row['max_agent_rss_mib']:.1f}", "gpu": row["total_gpu_seconds"],
                            "timeouts": row["timeout_runs"], "errors": row["error_runs"]})
        sections.append(f'<details open><summary><strong>{html.escape(game["game_id"])}</strong> · {len(game["baseline_actions"])} levels</summary>'
                        + table(performance, [("agent", "Agent")] + [(f"s{n}", f"L1 @{n}") for n in points]
                                + [("rhae", "RHAE /100"), ("levels", "Mean levels"), ("win", "Full clear")])
                        + table(compute, [("agent", "Agent"), ("wall", "sec/run"), ("rss", "Peak agent RSS MiB"),
                                          ("gpu", "Total GPU sec"), ("timeouts", "Timeout runs"), ("errors", "Error runs")]) + '</details>')
    board["per_game"] = aggregate
    board["game_report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "leaderboard.json").write_text(json.dumps(board, indent=2) + "\n")
    write_csv(output / "per_game.csv", aggregate)
    page = f'<!doctype html><meta charset="utf-8"><title>Game-wise metrics</title><style>{STYLE}</style>'
    page += '<a href="leaderboard.html">Leaderboard</a><h1>Game-wise metrics</h1>'
    page += f'<p>{len(config["seeds"])} seeds/game/agent; same protocol. Rates include all runs; no success-only averaging. CPU RSS is not GPU memory.</p>'
    page += '<p><a href="per_game.csv">Game CSV</a> · <a href="per_level.csv">Level CSV</a></p>' + ''.join(sections)
    (output / "game_metrics.html").write_text(page)
    main = output / "leaderboard.html"
    content = main.read_text()
    if 'href="game_metrics.html"' not in content:
        content = content.replace('</h1>', '</h1><p><a href="game_metrics.html">All metrics by game</a> · <a href="per_game.csv">Game CSV</a></p>', 1)
        main.write_text(content)
    return aggregate


if __name__ == "__main__":
    import sys
    game_report(sys.argv[1])
