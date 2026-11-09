"""Trusted-team JSONL agent adapter. stdout is reserved for protocol replies."""
import contextlib
import importlib
import importlib.metadata
import json
import resource
import sys
import time

from .agents import BUILTINS


def main():
    agent = None
    for line in sys.stdin:
        request = json.loads(line)
        started = time.process_time()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] == "init":
                    name = request["agent"]
                    if name in BUILTINS:
                        cls = BUILTINS[name]
                    else:
                        module, attribute = name.split(":", 1)
                        cls = getattr(importlib.import_module(module), attribute)
                    options = request.get("config", {})
                    agent = cls(seed=request["seed"], **({"config": options} if options else {}))
                    action = None
                elif request["op"] == "act":
                    action = agent.act(request["observation"])
                elif request["op"] == "observe":
                    if hasattr(agent, "observe"):
                        agent.observe(request["previous"], request["action"], request["observation"])
                    action = None
                else:
                    raise ValueError("Unknown operation")
                metrics = agent.metrics() if agent is not None and hasattr(agent, "metrics") else {}
            reply = {"action": action, "cpu_seconds": time.process_time() - started,
                     "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "metrics": metrics}
            if request["op"] == "init":
                packages = {}
                for package in ("numpy", "scipy", "torch", "jax", "jaxlib", "tensorflow", "tensorflow-probability",
                                "gymnasium", "stable-baselines3", "sb3-contrib"):
                    try:
                        packages[package] = importlib.metadata.version(package)
                    except importlib.metadata.PackageNotFoundError:
                        pass
                reply["runtime"] = {"executable": sys.executable, "python": sys.version, "packages": packages}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
