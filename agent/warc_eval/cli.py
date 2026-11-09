import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys

from .core import digest
from .processes import ProcessRegistry, stop_group, termination_signals


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def agent_bundle_sources(agents):
    """Snapshot first-party adapters, their helpers, and runtime-read upstream files.

    External plugins still need --artifacts for their dependency provenance.
    No game assets are included in the agent bundle.
    """
    if not any(name.startswith("warc_agents.") for name in agents):
        return {}
    package = importlib.util.find_spec("warc_agents")
    if package is None or package.origin is None:
        raise ValueError("warc_agents package not found")
    root = Path(package.origin).resolve().parent.parent
    paths = list((root / "warc_agents").rglob("*.py"))
    for name in agents:
        if name.startswith("warc_agents.graph:"):
            paths.extend(root / p for p in (
                "vendor/graph_exploration/graph_explorer.py",
                "vendor/graph_exploration/agents/heuristic_agent.py",
                "vendor/graph_exploration/LICENSE"))
    paths.extend((root / "configs/agents").glob("*.lock.txt"))
    manifest = root / "warc_agents/upstream_manifest.json"
    if manifest.exists():
        paths.append(manifest)
    return {str(p.relative_to(root)): {"path": str(p), "sha256": file_hash(p)}
            for p in sorted(set(paths))}


def verify_agent_bundle(source_root, bundle):
    for relative, spec in bundle.items():
        if file_hash(Path(source_root) / relative) != spec["sha256"]:
            raise ValueError(f"Agent source snapshot modified: {relative}")


def build_config(args):
    dataset = Path(args.dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    for package, version in manifest["sdk"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"SDK version mismatch: {package}")
    for game in manifest["games"]:
        stem, version = game["game_id"].split("-", 1)
        folder = dataset / "games" / stem / version
        if file_hash(folder / f"{stem}.py") != game["source_sha256"] or file_hash(folder / "metadata.json") != game["metadata_sha256"]:
            raise ValueError(f"Dataset modified: {game['game_id']}")
    games = manifest["games"]
    if args.games:
        games = [g for g in games if g["game_id"].split("-")[0] in args.games]
        if len(games) != len(set(args.games)):
            raise ValueError("Unknown/duplicate game selection")
    timeouts = (args.run_timeout, args.action_timeout, args.init_timeout)
    if args.budget <= 0 or any(not math.isfinite(t) or t < 0 for t in timeouts) or args.workers <= 0:
        raise ValueError("Budget/workers must be positive; timeouts finite and nonnegative (0 disables)")
    if len(set(args.agents)) != len(args.agents) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate agents/seeds")
    source = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    agent_sources = {}
    runtimes = json.loads(Path(args.agent_runtimes).read_text()) if args.agent_runtimes else {}
    if not isinstance(runtimes, dict) or set(runtimes) - set(args.agents):
        raise ValueError("Runtime mapping must contain only selected agent IDs")
    for name, spec in runtimes.items():
        if not isinstance(spec, dict) or set(spec) - {"command", "config"}:
            raise ValueError(f"Invalid runtime specification: {name}")
        command = spec.get("command")
        if command is not None and (not isinstance(command, list) or not command or
                                    not all(isinstance(s, str) and s for s in command)):
            raise ValueError("Agent command must be a nonempty argv list (no shell)")
        if command and (not Path(command[0]).is_absolute() or not os.access(command[0], os.X_OK)):
            raise ValueError("Agent executable must be an absolute executable path")
        if not isinstance(spec.get("config", {}), dict):
            raise ValueError("Agent config must be an object")
    json.dumps(runtimes, allow_nan=False)
    if args.workers != 1 and any(str(spec.get("config", {}).get("device", "cpu")).startswith("cuda") for spec in runtimes.values()):
        raise ValueError("GPU agents currently require --workers 1; scheduler assigns the visible GPU")
    for agent in args.agents:
        if ":" in agent:
            try:
                spec = importlib.util.find_spec(agent.split(":", 1)[0])
            except ModuleNotFoundError:
                spec = None
            if spec is None or spec.origin is None:
                if runtimes.get(agent, {}).get("command"):
                    continue  # Module can live exclusively in the agent runtime.
                raise ValueError(f"Agent module not found: {agent}")
            agent_sources[agent] = {"path": spec.origin, "sha256": file_hash(spec.origin)}
    artifacts = {str(Path(p).resolve()): file_hash(p) for p in args.artifacts}
    config = {"schema": 1, "dataset": str(dataset), "manifest_sha256": file_hash(dataset / "manifest.json"),
              "games": games, "agents": args.agents, "seeds": args.seeds, "env_seed": args.env_seed,
              "budget": args.budget, "mode": args.mode, "workers": args.workers, "threads_per_worker": 1,
              "run_timeout_seconds": args.run_timeout,
              "action_timeout_seconds": args.action_timeout, "source_hashes": source,
              "init_timeout_seconds": args.init_timeout, "agent_runtimes": runtimes,
              "agent_sources": agent_sources, "agent_bundle": agent_bundle_sources(args.agents),
              "artifacts": artifacts,
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "protocol": "public-all-frames;level-only-reset;initial-make-free;subsequent-ops-counted;trusted-agent-subprocess-v1"}
    config["config_id"] = digest(config)
    return config


def run_batch(args):
    config = build_config(args)
    output = Path(args.output).resolve()
    if output.exists():
        if not args.resume:
            raise ValueError("Output exists; use a new directory or --resume")
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume refused: configuration, source, dataset or packages changed")
    else:
        output.mkdir(parents=True)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        snapshot = output / "source" / "warc_eval"
        snapshot.mkdir(parents=True)
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(source, snapshot / source.name)
        for relative, spec in config.get("agent_bundle", {}).items():
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(spec["path"], destination)
    for filename, expected_hash in config["source_hashes"].items():
        if file_hash(output / "source" / "warc_eval" / filename) != expected_hash:
            raise ValueError("Source snapshot modified")
    verify_agent_bundle(output / "source", config.get("agent_bundle", {}))
    lock = output / ".running"
    # Advisory OS lock releases on process death, unlike persistent PID files.
    import fcntl
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = [(game, agent, seed) for game in config["games"] for agent in config["agents"] for seed in config["seeds"]]
        registry = ProcessRegistry()
        def execute(task):
            game, agent, seed = task
            run_id = f"{game['game_id']}_{digest(agent)[:10]}_{seed}"
            directory = output / "runs" / run_id
            if (directory / "result.json").exists():
                return json.loads((directory / "result.json").read_text())
            if directory.exists():
                raise RuntimeError(f"Incomplete previous run: {directory}; preserve it and start a new evaluation")
            (output / "runs").mkdir(exist_ok=True)
            with (output / "runs" / f"{run_id}.log").open("w") as log:
                process = registry.start([sys.executable, "-m", "warc_eval.cli", "_worker",
                                            str(output / "config.json"), game["game_id"], agent, str(seed), str(directory)],
                                           stdout=log, stderr=log, start_new_session=True,
                                           cwd=output / "source",
                                           env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                                "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
                                                "PYTHONPATH": str(output / "source") + os.pathsep + str(Path.cwd())
                                                + os.pathsep + os.environ.get("PYTHONPATH", "")})
                try:
                    limit = config["run_timeout_seconds"]
                    code = process.wait(timeout=limit + 90 if limit else None)
                except subprocess.TimeoutExpired:
                    stop_group(process)
                    raise RuntimeError(f"Hard worker timeout: {run_id}; see log")
                finally:
                    # Cancellation can already have reaped this worker.
                    registry.forget(process)
            if code != 0 or not (directory / "result.json").exists():
                stop_group(process)
                raise RuntimeError(f"Worker failed: {run_id}; see log")
            result = json.loads((directory / "result.json").read_text())
            print(f"{game['game_id']} {agent} seed={seed}: {result['status']} "
                  f"levels={(result['metrics'] or {}).get('levels_completed')} "
                  f"wall={result['wall_seconds']:.1f}s", flush=True)
            return result
        with termination_signals():
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
            try:
                results = list(executor.map(execute, tasks))
            except BaseException:
                registry.stop()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    from .report import report
    report(output)
    failures = [r for r in results if r["status"] not in ("win", "l1_complete", "budget_exhaustion", "timeout")]
    if failures:
        raise RuntimeError(f"{len(failures)} invalid/error runs; retained in report, inspect before interpretation")


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 public-game evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Snapshot official pub25 assets and human baselines")
    p.add_argument("destination")
    p = sub.add_parser("run", help="Evaluate agents, then build the leaderboard")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--agents", nargs="+", default=["random", "component_random", "count"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--games", nargs="+")
    p.add_argument("--env-seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=4000)
    p.add_argument("--mode", choices=["l1", "full"], default="full")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--run-timeout", type=float, default=600)
    p.add_argument("--action-timeout", type=float, default=10)
    p.add_argument("--init-timeout", type=float, default=60, help="Initialization limit; 0 disables")
    p.add_argument("--agent-runtimes", help="JSON map of agent ID to command argv and constructor config")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--artifacts", nargs="*", default=[], help="Checkpoint/config/helper files to hash for provenance")
    p = sub.add_parser("report", help="Recompute scores from logs and generate HTML/JSON/CSV")
    p.add_argument("output")
    p = sub.add_parser("_worker", help=argparse.SUPPRESS)
    for field in ("config", "game", "agent", "seed", "directory"):
        p.add_argument(field)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        prepare(args.destination)
    elif args.command == "run":
        run_batch(args)
    elif args.command == "report":
        from .report import report
        report(Path(args.output).resolve())
    else:
        from .runner import run_one
        config = json.loads(Path(args.config).read_text())
        game = next(g for g in config["games"] if g["game_id"] == args.game)
        run_one(config, game, args.agent, int(args.seed), args.directory)


if __name__ == "__main__":
    main()
