"""Pinned upstream Graph Explorer with a public-observation-only adapter.

Load selected upstream classes, not its legacy API client/LLM agent registry.
Transition bookkeeping fixes are applied to a copy of the pinned vendor sources,
and both patched sources are hash checked.
"""
import ast
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from pathlib import Path
import random
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import numpy as np

from warc_eval.agents import RandomAgent as _RandomAgent

ROOT = Path(__file__).resolve().parents[1] / "vendor/graph_exploration"
HASHES = {
    "graph_explorer.py": "c4815374a46ebb9d2eccbac0c809a58835b52a726d2f12e1c77b126aa156cdbc",
    "agents/heuristic_agent.py": "93f47259855b7be3f0c3c605b9d33915145e332a6849d88f1eb8eb4381146c40",
}


class RandomAgent(_RandomAgent):
    """Upstream kind-then-coordinate sampling, restricted to public valid kinds.

    The released template sampled every enum kind, even unavailable ones.
    The mask is the explicit current-AA3 connection change.
    """


class State(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    GAME_OVER = "GAME_OVER"
    WIN = "WIN"


def load_classes(rng):
    """Execute hash-pinned class ASTs with local RNG and minimal SDK shims."""
    class Base:
        game_id = "adapter"  # No environment identity is exposed to the policy.

    class Action(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        def set_data(self, data):
            self.action_data = SimpleNamespace(**data)

        def is_simple(self):
            return self.value != 6

        def is_complex(self):
            return self.value == 6

    module = ModuleType(f"_warc_graph_{id(rng)}")
    namespace = module.__dict__
    namespace.update({k: globals()[k] for k in (
        "np", "deque", "dataclass", "field", "Dict", "Hashable", "List",
        "Optional", "Set", "Tuple", "Any", "hashlib", "time")})
    from collections import defaultdict
    namespace.update(defaultdict=defaultdict, random=rng, Agent=Base,
                     GameState=State, GameAction=Action, FrameData=SimpleNamespace,
                     trace_agent_session=lambda f: f, logger=logging.getLogger(__name__),
                     INFINITY=np.iinfo(np.int32).max,
                     edge_dtype=np.dtype([("group", "i4"), ("result", "i4"),
                                          ("target", "U32"), ("distance", "i4"), ("errors", "i4")]))
    sys.modules[module.__name__] = module  # Required while dataclasses are built.
    try:
        for filename, names in (
            ("graph_explorer.py", {"format_struct_table", "NodeInfo", "GraphExplorer"}),
            ("agents/heuristic_agent.py", {"FrameProcessor", "HeuristicAgent"}),
        ):
            path = ROOT / filename
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != HASHES[filename]:
                raise RuntimeError(f"Upstream source hash mismatch: {path}")
            tree = ast.parse(source, filename=str(path))
            selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                        and node.name in names]
            if len(selected) != len(names):
                raise RuntimeError(f"Missing upstream definitions: {filename}")
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    finally:
        del sys.modules[module.__name__]
    return namespace


class GraphAgent:
    frontier = True
    new_actions = False

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.upstream = load_classes(self.rng)
        self.policy = self.upstream["HeuristicAgent"]()
        self.rng.seed(seed)  # Replace upstream time/game-ID seed.
        self.policy.minimal_step_time = 0
        self.policy.favor_frontier_search = self.frontier
        self.policy.favor_new_actions = self.new_actions
        actions = self.upstream["GameAction"]
        self.policy.SIMPLE_ACTION_ID2GAME_ACTION = {k: actions(k) for k in (1, 2, 3, 4, 5, 7)}
        self.recoveries = 0
        self.action_kinds = None
        self.reset_boundaries = 0

    def act(self, observation):
        kinds = tuple(sorted(observation['available_actions']))
        if self.action_kinds is not None and kinds != self.action_kinds:
            # Candidate indices depend on available action kinds.
            self.policy.level_up = True
            self.policy.last_hashed_frame = None
            self.policy.last_action = None
        self.action_kinds = kinds
        frame = SimpleNamespace(frame=observation["frame"],
                                state=State(observation["state"]),
                                available_actions=list(kinds))
        # Unlike upstream main(), unexpected exceptions must surface as errors.
        try:
            action = self.policy.choose_action([], frame)
            # consume initialization here, not in observe(): recovery requests
            # must survive observe() until the next actual policy call.
            self.policy.level_up = False
        except (IndexError, ValueError) as exc:
            # Upstream retries the previous action and rebuilds after exhaustion.
            # Restrict that recovery to known exhausted-candidate cases.
            if str(exc) not in ("Cannot choose from an empty sequence",) and not str(exc).startswith("No available actions found for frame"):
                raise
            self.recoveries += 1
            self.policy.failed = True
            self.policy.level_up = True
            action = self.policy.last_action_object
            if action.value != 0 and action.value not in observation["available_actions"]:
                action = self.upstream["GameAction"].RESET
        result = {"kind": action.value}
        if action.value == 6:
            result.update(x=int(action.action_data.x), y=int(action.action_data.y))
        return result

    def observe(self, previous, action, observation):
        policy = self.policy
        if observation["levels_completed"] > previous["levels_completed"]:
            policy.level_up = True
            policy.status_bar_mask = None
            policy.last_hashed_frame = None
            policy.last_action = None
        # Runner executes RESET without an act() callback. Never assign its
        # resulting frame to the preceding non-RESET action. Keep level memory.
        if action['kind'] == 0 or observation['state'] == 'GAME_OVER':
            policy.last_hashed_frame = None
            policy.last_action = None
            policy.last_transition_suspicious = False
            self.reset_boundaries += 1

    def metrics(self):
        return {"recoveries": self.recoveries,
                "reset_boundaries": self.reset_boundaries,
                "transition_conflicts": getattr(self.policy, 'transition_conflicts', 0),
                "edge_revisions": getattr(self.policy.graph_explorer, 'edge_revisions', 0),
                "graph_nodes": len(self.policy.graph_explorer._nodes)}


class SegmentationAgent(GraphAgent):
    frontier = False


class NewActionAgent(GraphAgent):
    frontier = False
    new_actions = True


class LeastTriedAgent:
    """Additional control, not a claimed reproduction of the graph paper."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.processor = load_classes(self.rng)["FrameProcessor"]()
        self.counts = Counter()
        self.mask = None
        self.level = None

    def act(self, observation):
        grid = np.array(observation["frame"][-1], dtype=np.uint8)
        if self.level != observation["levels_completed"]:
            labels, segments = self.processor.segment_frame(grid)
            _, self.mask = self.processor.identify_status_bars(labels, segments)
            self.level = observation["levels_completed"]
            self.counts.clear()
        grid[self.mask] = 16
        labels, segments = self.processor.segment_frame(grid)
        grid[grid == 16] = 0
        state = self.processor.hash_frame(grid)
        candidates = [(6, i) for i in range(len(segments))] if 6 in observation["available_actions"] else []
        candidates += [(k, None) for k in observation["available_actions"] if k not in (0, 6)]
        least = min(self.counts[state, a] for a in candidates)
        selected = self.rng.choice([a for a in candidates if self.counts[state, a] == least])
        self.counts[state, selected] += 1
        kind, segment = selected
        if kind == 6:
            points = np.argwhere(labels == segment)
            y, x = points[self.rng.randrange(len(points))]
            return {"kind": 6, "x": int(x), "y": int(y)}
        return {"kind": kind}
