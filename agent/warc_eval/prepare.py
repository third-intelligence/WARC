"""Download the public set without playing games or opening scorecards."""
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

PUB25 = set("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split())


def prepare(destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = "https://three.arcprize.org"
    session = requests.Session()
    response = session.get(base + "/api/games/anonkey", timeout=30)
    response.raise_for_status()
    session.headers["X-Api-Key"] = response.json()["api_key"]
    response = session.get(base + "/api/games", timeout=30)
    response.raise_for_status()
    listing = response.json()
    selected = [item for item in listing if item["game_id"].split("-")[0] in PUB25]
    if len(selected) != 25 or {x["game_id"].split("-")[0] for x in selected} != PUB25:
        raise ValueError("Public API did not return exactly one version of each pub25 game")
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_url": base + "/api/games", "games": [], "sdk": {}}
    for package in ("arc-agi", "arcengine"):
        response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=30)
        response.raise_for_status()
        latest = response.json()["info"]["version"]
        installed = importlib.metadata.version(package)
        if installed != latest:
            raise ValueError(f"Upgrade {package}: installed={installed}, latest={latest}")
        manifest["sdk"][package] = installed
    for item in sorted(selected, key=lambda x: x["game_id"]):
        game_id = item["game_id"]
        stem, version = game_id.split("-", 1)
        baseline = item.get("baseline_actions")
        if not baseline or any(type(b) is not int or b <= 0 for b in baseline):
            raise ValueError(f"Missing/invalid official human baselines: {game_id}")
        directory = root / "games" / stem / version
        directory.mkdir(parents=True)
        response = session.get(f"{base}/api/games/{game_id}/source", timeout=30)
        response.raise_for_status()
        source = response.content
        # Treat downloaded code as an opaque environment asset; never agent input.
        (directory / f"{stem}.py").write_bytes(source)
        metadata = {**item, "date_downloaded": manifest["created_utc"]}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest["games"].append({"game_id": game_id, "baseline_actions": baseline,
                                  "source_sha256": hashlib.sha256(source).hexdigest(),
                                  "metadata_sha256": hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest()})
        print("Prepared", game_id, flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
