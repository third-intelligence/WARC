"""Run semantics and independent, event-log-derived metrics."""
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def public_observation(obs):
    result = {"frame": [f.tolist() for f in obs.frame], "state": obs.state.name,
              "levels_completed": int(obs.levels_completed), "win_levels": int(obs.win_levels),
              "available_actions": [int(a) for a in obs.available_actions],
              "full_reset": bool(obs.full_reset)}
    if not result["frame"]:
        raise ValueError("Empty frame response")
    return result


def validate_action(action, observation):
    if not isinstance(action, dict) or type(action.get("kind")) is not int:
        raise ValueError("Action must contain integer kind")
    kind = action["kind"]
    if kind not in range(8) or (kind != 0 and kind not in observation["available_actions"]):
        raise ValueError("Unavailable action")
    expected = {"kind", "x", "y"} if kind == 6 else {"kind"}
    if set(action) != expected:
        raise ValueError("Unexpected/missing action fields")
    if kind == 6 and any(type(action[k]) is not int or not 0 <= action[k] < 64 for k in ("x", "y")):
        raise ValueError("Click coordinates must be integers in [0,63]")
    return action


def score_from_events(events, baselines):
    """SDK formula, independently rebuilt from successful operations.

    Initial make is excluded; all subsequent level RESET/Undo/no-op operations
    count. Full resets are disabled. Missing baselines yield null, not zero.
    """
    successes = [e for e in events if e.get("observation") is not None]
    initial = successes[0]["observation"]
    total_levels = initial["win_levels"]
    completions = []
    previous = initial["levels_completed"]
    if successes[0]["step"] != 0 or previous != 0 or total_levels < 1:
        raise ValueError("Invalid initial observation")
    for index, event in enumerate(successes[1:], 1):
        if event["step"] != index or event["observation"].get("full_reset"):
            raise ValueError("Non-contiguous operations or forbidden full reset")
        completed = event["observation"]["levels_completed"]
        if completed < previous or completed > previous + 1:
            raise ValueError("Unexpected level regression/jump")
        if completed > previous:
            completions.append(event["step"])
        previous = completed
    steps = successes[-1]["step"]
    level_actions, last = [], 0
    for step in completions:
        level_actions.append(step - last)
        last = step
    score = None
    if baselines and len(baselines) == total_levels and all(type(b) is int and b > 0 for b in baselines):
        denominator = total_levels * (total_levels + 1) / 2
        numerator = sum((i + 1) * min(1.15, (baselines[i] / n) ** 2)
                        for i, n in enumerate(level_actions))
        cap = len(completions) * (len(completions) + 1) / 2
        score = 100 * min(numerator, cap) / denominator
        if not math.isfinite(score):
            raise ValueError("Non-finite score")
    return {"steps": steps, "levels_completed": len(completions), "total_levels": total_levels,
            "completion_steps": completions, "level_actions": level_actions,
            "l1_step": completions[0] if completions else None,
            "win": successes[-1]["observation"]["state"] == "WIN", "rhae": score,
            "resets": sum(e.get("action", {}).get("kind") == 0 for e in successes[1:]),
            "undo": sum(e.get("action", {}).get("kind") == 7 for e in successes[1:])}
