"""One isolated process per run; one public-observation agent subprocess."""
import gzip
import hashlib
import json
import logging
import math
import os
import resource
import selectors
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .core import public_observation, score_from_events, validate_action


class AgentFailure(Exception):
    pass


class AgentClient:
    def __init__(self, agent, seed, directory, deadline, command=None, agent_config=None, init_timeout=60):
        self.stderr = open(directory / "agent.stderr.log", "w")
        self.process = subprocess.Popen(command or [sys.executable, "-m", "warc_eval.agent_worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, bufsize=0,
                                        cwd=directory, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        os.set_blocking(self.process.stdin.fileno(), False)
        self.write_selector = selectors.DefaultSelector()
        self.write_selector.register(self.process.stdin, selectors.EVENT_WRITE)
        self.cpu_seconds = 0.0
        self.peak_rss_kib = 0
        self.deadline = deadline
        self.metrics_log = open(directory / "agent.metrics.jsonl", "w")
        self.latest_metrics = {}
        self.runtime = {}
        try:
            self.request({"op": "init", "agent": agent, "seed": seed, "config": agent_config or {}}, init_timeout)
        except BaseException:
            self.close()
            raise

    def request(self, message, timeout):
        reply_deadline = min(time.monotonic() + timeout if timeout else math.inf, self.deadline)
        payload = memoryview((json.dumps(message, separators=(",", ":")) + "\n").encode())
        while payload:
            remaining = reply_deadline - time.monotonic()
            if remaining <= 0 or not self.write_selector.select(remaining if math.isfinite(remaining) else None):
                raise TimeoutError("Agent input deadline exceeded")
            try:
                sent = os.write(self.process.stdin.fileno(), payload)
                payload = payload[sent:]
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise AgentFailure("Agent closed input pipe") from exc
        data = bytearray()
        while b"\n" not in data:
            remaining = reply_deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining if math.isfinite(remaining) else None):
                raise TimeoutError("Agent response deadline exceeded")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk or len(data) + len(chunk) > 1_000_000:
                raise AgentFailure("Agent exited or exceeded response size limit")
            data.extend(chunk)
        if bytes(data).count(b"\n") != 1 or not data.endswith(b"\n"):
            raise AgentFailure("Expected exactly one JSONL response")
        try:
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("Reply must be an object")
            if "error" in result:
                raise AgentFailure(str(result["error"]))
            self.cpu_seconds += float(result.get("cpu_seconds", 0))
            self.peak_rss_kib = max(self.peak_rss_kib, int(result.get("peak_rss_kib", 0)))
            metrics = result.get("metrics", {})
            if not isinstance(metrics, dict):
                raise ValueError("Agent metrics must be an object")
            json.dumps(metrics, allow_nan=False)
            self.latest_metrics = metrics
            if message["op"] == "init":
                self.runtime = result.get("runtime", {})
            self.metrics_log.write(json.dumps({"op": message["op"], "metrics": metrics}) + "\n")
            self.metrics_log.flush()
            return result["action"]
        except (ValueError, TypeError, KeyError) as exc:
            raise AgentFailure(f"Invalid agent reply: {exc}") from exc

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()
        self.write_selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()
        self.metrics_log.close()


def run_episode(env, choose_action, emit, budget, mode="full", observe=None):
    """Pure runner loop, injected environment/action selector for boundary tests."""
    from arcengine import GameAction
    raw = env.observation_space
    if raw is None:
        raise RuntimeError("Environment initialization returned no observation")
    observation = public_observation(raw)
    if observation["levels_completed"] != 0 or observation["state"] != "NOT_FINISHED":
        raise ValueError("Run must start at unsolved first level")
    emit({"step": 0, "observation": observation})
    for step in range(1, budget + 1):
        action = {"kind": 0} if observation["state"] == "GAME_OVER" else choose_action(observation)
        try:
            validate_action(action, observation)
        except ValueError as exc:
            emit({"step": step - 1, "attempt": step, "action": action, "error": str(exc)})
            return "invalid_action"
        started = time.monotonic()
        if action["kind"] == 0:
            raw = env.reset()
        else:
            data = {k: action[k] for k in ("x", "y")} if action["kind"] == 6 else None
            raw = env.step(GameAction.from_id(action["kind"]), data=data)
        if raw is None:
            raise RuntimeError("Environment operation returned no observation")
        following = public_observation(raw)
        emit({"step": step, "action": action, "observation": following,
              "env_seconds": time.monotonic() - started})
        if following["full_reset"] or following["levels_completed"] < observation["levels_completed"]:
            raise ValueError("Full reset/regression violates first-exposure protocol")
        if observe:
            observe(observation, action, following)
        if following["state"] == "WIN":
            return "win"
        if mode == "l1" and following["levels_completed"] >= 1:
            return "l1_complete"
        observation = following
    return "budget_exhaustion"


def run_one(config, game, agent, seed, directory, agent_command=None):
    # Force level-only RESET, including repeated RESET. Initial make is unscored.
    os.environ["ONLY_RESET_LEVELS"] = "false"
    os.environ["OPERATION_MODE"] = "OFFLINE"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(directory / "mpl_cache")
    from arc_agi import Arcade, OperationMode
    started = time.monotonic()
    deadline = started + config["run_timeout_seconds"] if config["run_timeout_seconds"] else math.inf
    events, client, card, failure = [], None, None, None
    status = "implementation_error"
    arc = None
    logger = logging.getLogger("warc.sdk")
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.NullHandler())
    def alarm(_signum, _frame):
        raise TimeoutError("Run wall-time budget exhausted")
    old_handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, config["run_timeout_seconds"])
    with gzip.open(directory / "events.jsonl.gz", "wt", compresslevel=1) as log:
        def emit(event):
            event["wall_seconds"] = time.monotonic() - started
            log.write(json.dumps(event, separators=(",", ":")) + "\n")
            # Keep only scoring metadata in RAM; full frames are streamed to disk.
            slim = {**event}
            if "observation" in event:
                slim["observation"] = {k: v for k, v in event["observation"].items() if k != "frame"}
            events.append(slim)
        try:
            arc = Arcade(operation_mode=OperationMode.OFFLINE,
                         environments_dir=str(Path(config["dataset"]) / "games"), logger=logger)
            env = arc.make(game["game_id"], seed=config["env_seed"], save_recording=False)
            if env is None:
                raise RuntimeError("Could not initialize environment")
            os.environ["ONLY_RESET_LEVELS"] = "true"
            spec = config.get("agent_runtimes", {}).get(agent, {})
            client = AgentClient(agent, seed, directory, deadline, command=agent_command or spec.get("command"),
                                 agent_config=spec.get("config"), init_timeout=config.get("init_timeout_seconds", 60))
            status = run_episode(env, lambda obs: client.request({"op": "act", "observation": obs},
                                                                 config["action_timeout_seconds"]),
                                 emit, config["budget"], config["mode"],
                                 observe=lambda previous, action, following: client.request(
                                     {"op": "observe", "previous": previous, "action": action, "observation": following},
                                     config["action_timeout_seconds"]))
        except TimeoutError as exc:
            status, failure = "timeout", str(exc)
        except AgentFailure as exc:
            status, failure = "agent_error", str(exc)
        except Exception:
            if time.monotonic() >= deadline:
                status, failure = "timeout", "Run deadline exceeded inside environment operation"
            else:
                failure = traceback.format_exc()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            if client:
                # Explicit, acknowledged finalization BEFORE terminating worker.
                # It performs no actions/learning. Include its I/O in wall time;
                # allow bounded artifact export even after a policy timeout.
                if spec.get('config', {}).get('finalize_checkpoint'):
                    client.deadline = math.inf
                    try:
                        client.request({'op': 'finalize', 'status': status}, 120)
                    except Exception:
                        failure = (failure or '') + '\nCheckpoint finalization failed: ' + traceback.format_exc()
                        status = 'agent_error'
                client.close()
            if arc:
                try:
                    sdk_card = arc.close_scorecard()
                    card = sdk_card.model_dump(mode="json") if sdk_card else None
                except Exception:
                    failure = (failure or "") + "\nScorecard close failed: " + traceback.format_exc()
                    status = "implementation_error"
    metrics = score_from_events(events, game["baseline_actions"]) if events and "observation" in events[0] else None
    if metrics and card:
        if (not abs(card["score"] - metrics["rhae"]) < 1e-8
                or card["total_actions"] != metrics["steps"]
                or card["total_levels_completed"] != metrics["levels_completed"]):
            status, failure = "score_mismatch", f"SDK score/actions/levels={card['score']}/{card['total_actions']}/{card['total_levels_completed']}, log={metrics}"
    wall = time.monotonic() - started
    result = {"schema": 1, "config_id": config["config_id"], "game_id": game["game_id"],
              "agent": agent, "agent_seed": seed, "env_seed": config["env_seed"],
              "status": status, "error": failure, "metrics": metrics, "wall_seconds": wall,
              "execution_host": os.uname().nodename, "pbs_job_id": os.environ.get("PBS_JOBID"),
              "agent_cpu_seconds": client.cpu_seconds if client else None,
              "agent_peak_rss_mib": client.peak_rss_kib / 1024 if client else None,
              "agent_metrics": client.latest_metrics if client else {},
              "agent_runtime": client.runtime if client else {},
              "worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "gpu_seconds": 0 if agent in ("random", "component_random", "count") else None,
              "events_sha256": hashlib.sha256((directory / "events.jsonl.gz").read_bytes()).hexdigest(),
              "scorecard": card}
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
