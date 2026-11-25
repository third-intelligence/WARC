# Derived from the transition-record module of a concurrent study (credited once it is published), with part of its records removed.
# MIT License, Copyright (c) 2026 [the authors of a concurrent study, credited once it is published]; the full text is in incl_licenses/LICENSE_instrumented_games.
"""The log of one action, shared by the hooked game programs.

A game writes one line per visual change, "[verb]:[target]:[details]:text", while it steps through the frames of an
action (ActionLogState.log), and marks events such as a win or a reset (set_event). At the end of the action the lines
of all frames are merged (default_aggregator: chained moves and colour changes become one line) and the event lines
are added (compose_default_summary). The annotator reads the merged lines.

Every log function also returns a second list, which mirrors the first and is not used.
"""

import re
from typing import Any, Callable


class ActionLogState:
    """"""

    def __init__(
        self,
        aggregator:   Callable[[list[list[str]], bool], list[str]] | None = None,
        post_process: Callable[[list[str], list[str], dict[str, Any]], tuple[list[str], list[str]]] | None = None,
    ) -> None:
        """







        """
        self._aggregator   = aggregator
        self._post_process = post_process

        self._step_log:    list[str] = []
        self._step_log_l2: list[str] = []

        self._frames:      list[list[str]] = []
        self._frames_l2:   list[list[str]] = []

        self._events:      dict[str, Any] = {}


    def reset(self) -> None:
        """"""
        self._step_log    = []
        self._step_log_l2 = []
        self._frames      = []
        self._frames_l2   = []
        self._events      = {}


    def log(self, en: str, l2: str) -> None:
        """

        """
        self._step_log.append(en)
        self._step_log_l2.append(l2)

    def set_event(self, key: str, value: Any) -> None:
        """


        """
        self._events[key] = value

    def get_event(self, key: str, default: Any = None) -> Any:
        """"""
        return self._events.get(key, default)

    def has_event(self, key: str) -> bool:
        return key in self._events

    def snapshot_step(self) -> None:
        """"""
        self._frames.append(list(self._step_log))
        self._frames_l2.append(list(self._step_log))
        self._step_log    = []
        self._step_log_l2 = []


    def get_frame_logs(self) -> tuple[list[list[str]], list[list[str]]]:
        """"""
        return self._frames, self._frames_l2

    def get_action_log(self) -> tuple[list[str], list[str]]:
        """
        """
        if self._aggregator is not None:
            agg = list(self._aggregator(self._frames, False))
        else:
            seen: set[str] = set()
            agg: list[str] = []
            for frame in self._frames:
                for line in frame:
                    if line not in seen:
                        seen.add(line)
                        agg.append(line)
        agg_l2 = list(agg)
        if self._post_process is not None:
            agg, agg_l2 = self._post_process(agg, agg_l2, self._events)
        return agg, agg_l2








_RE_MOVE_V2 = re.compile(
    r'(\[.+?\]):\[(.+?)\]:\[[^\]]*?position\((-?\d+)_(-?\d+)\)_\((-?\d+)_(-?\d+)\)[^\]]*\]:(.*)'
)
_RE_POS_SUB = re.compile(
    r'(position)\((-?\d+)_(-?\d+)\)_\((-?\d+)_(-?\d+)\)'
)
_RE_CHANGE_COLOR = re.compile(
    r'^(\[change\]):\[(.+?)\]:\[(.*?)color\((\w+)\)_\((\w+)\)(.*?)\]:(.*)$'
)
_RE_CHANGE_SHAPE = re.compile(
    r'^(\[change\]):\[(.+?)\]:\[(.*?)shape\((\w+)\)_\((\w+)\)(.*?)\]:(.*)$'
)
_RE_POS_IN_PARAMS = re.compile(r'position\(([^)]+)\)')
_RE_BRIGHT_FULL = re.compile(r'^\[bright\]:\[(.+?)\]:\[([^\]]*)\]:.*$')
_RE_EXT_SHR     = re.compile(r'(\[(?:extend|shrink)\]):\[(.+?)\]:')
_BRIGHT_RE      = re.compile(r'^\[bright\]:\[([^\]]+)\]:')
_APP_DIS_RE     = re.compile(r'^\[(?:appear|disappear)\]:\[([^\]]+)\]:')

_DIR_EN = {( 1,  0): "right",      (-1,  0): "left",
           ( 0,  1): "down",       ( 0, -1): "up",
           ( 1,  1): "down-right", (-1,  1): "down-left",
           ( 1, -1): "up-right",   (-1, -1): "up-left"}
_NON_CHAIN_VERBS = {"[extend]", "[shrink]"}


def _sign(n: int) -> int:
    return 0 if n == 0 else (1 if n > 0 else -1)


class _OpenSegment:
    """"""
    __slots__ = ("name", "x1", "y1", "x2", "y2",
                 "prev_dx", "prev_dy", "prev_axis", "orig", "count")

    def __init__(self, name, x1, y1, x2, y2, orig):
        self.name = name
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.prev_dx = x2 - x1
        self.prev_dy = y2 - y1
        self.prev_axis = None
        self.orig = orig
        self.count = 1


def _axis_of(dx_: int, dy_: int) -> str | None:
    if dx_ != 0 and dy_ == 0: return "x"
    if dy_ != 0 and dx_ == 0: return "y"
    return None


def _make_move(name: str, x1: int, y1: int, x2: int, y2: int,
               template: str | None, is_l2: bool) -> str:
    """







    """
    dx, dy = x2 - x1, y2 - y1
    dk = (_sign(dx), _sign(dy))
    if template is not None:
        m = _RE_MOVE_V2.fullmatch(template)
        if m is not None:
            head, _, rest      = template.partition(":")
            rest_subj, _, rest = rest.partition(":")
            params_with_brackets, _, _ = rest.partition(":")
            new_params = _RE_POS_SUB.sub(
                lambda mm: f"{mm.group(1)}({x1}_{y1})_({x2}_{y2})",
                params_with_brackets,
            )
            orig_desc = m.group(7) or ""


            _orig_normal, _sep, _orig_secret = orig_desc.partition('|')
            dir_str = _DIR_EN.get(dk, f"({dx:+d},{dy:+d})")

            _em = re.match(r"^(.+?) moves\b", _orig_normal)
            subj = _em.group(1) if _em else name
            desc = f"{subj} moves {dir_str} from ({x1},{y1}) to ({x2},{y2})."
            if _sep:
                _sm = re.match(r"^(.+?) moves\b", _orig_secret)
                s_subj = _sm.group(1) if _sm else "Object"
                desc += f"|{s_subj} moves {dir_str} from ({x1},{y1}) to ({x2},{y2})."
            return f"{head}:{rest_subj}:{new_params}:{desc}"
    dir_str = _DIR_EN.get(dk, f"({dx:+d},{dy:+d})")
    return (f"[move]:[{name}]:[position({x1}_{y1})_({x2}_{y2})]:"
            f"{name} moves {dir_str} from ({x1},{y1}) to ({x2},{y2}).")


def default_aggregator(frame_logs: list[list[str]], is_l2: bool = False) -> list[str]:
    """








    """
    open_segments: dict[str, list[_OpenSegment]] = {}
    all_segments: list[_OpenSegment] = []
    misc_seen:    set[str]  = set()
    misc_order:   list[str] = []
    _extent_key_idx: dict[tuple[str, str], int] = {}

    for frame in frame_logs:
        for entry in frame:
            m = _RE_MOVE_V2.fullmatch(entry)
            if not m or m.group(1) in _NON_CHAIN_VERBS:
                _es_match = _RE_EXT_SHR.match(entry)
                if _es_match:
                    _key = (_es_match.group(1), _es_match.group(2))
                    if _key in _extent_key_idx:
                        misc_order[_extent_key_idx[_key]] = entry
                    else:
                        _extent_key_idx[_key] = len(misc_order)
                        misc_order.append(entry)
                elif entry not in misc_seen:
                    misc_seen.add(entry)
                    misc_order.append(entry)
                continue
            name = m.group(2)
            x1, y1 = int(m.group(3)), int(m.group(4))
            x2, y2 = int(m.group(5)), int(m.group(6))
            dx, dy = x2 - x1, y2 - y1
            cur_axis = _axis_of(dx, dy)

            bucket = open_segments.setdefault(name, [])
            match: _OpenSegment | None = None
            match_idx = -1
            for i, seg in enumerate(bucket):
                if (seg.x2, seg.y2) == (x1, y1):
                    match = seg
                    match_idx = i
                    break

            if match is None:
                new_seg = _OpenSegment(name, x1, y1, x2, y2, entry)
                new_seg.prev_axis = cur_axis
                bucket.append(new_seg)
                all_segments.append(new_seg)
                continue

            prev_dir = (_sign(match.prev_dx), _sign(match.prev_dy))
            cur_dir  = (_sign(dx), _sign(dy))







            def _component_break(p, c):
                if p > 0 and c < 0: return True
                if p < 0 and c > 0: return True
                if p == 0 and c != 0: return True
                return False
            dir_change = (
                cur_dir != (0, 0)
                and prev_dir != (0, 0)
                and (_component_break(prev_dir[0], cur_dir[0])
                     or _component_break(prev_dir[1], cur_dir[1]))
            )
            if dir_change:
                bucket.pop(match_idx)
                new_seg = _OpenSegment(name, x1, y1, x2, y2, entry)
                new_seg.prev_axis = cur_axis
                bucket.append(new_seg)
                all_segments.append(new_seg)
            else:
                match.x2, match.y2 = x2, y2
                if dx != 0: match.prev_dx = dx
                if dy != 0: match.prev_dy = dy
                if cur_axis is not None: match.prev_axis = cur_axis
                match.count += 1

    result: list[str] = []
    for seg in all_segments:






        result.append(_make_move(seg.name, seg.x1, seg.y1, seg.x2, seg.y2,
                                 template=seg.orig, is_l2=is_l2))


    color_chain_idx:   dict[tuple[str, str], int] = {}
    color_chain_last:  dict[tuple[str, str], str] = {}
    color_chain_first: dict[tuple[str, str], str] = {}
    last_pair_per_key: dict[tuple[str, str], tuple[str, str]] = {}
    _suppress_idx: set[int] = set()

    for idx, entry in enumerate(misc_order):
        m = _RE_CHANGE_COLOR.match(entry)
        if not m:
            continue
        verb, target, params_pre, from_c, to_c, params_post, _desc = m.groups()
        params_full = params_pre + f"color({from_c})_({to_c})" + params_post
        pos_match = _RE_POS_IN_PARAMS.search(params_full)
        pos_key = pos_match.group(1) if pos_match else ""
        key = (target, pos_key)

        if last_pair_per_key.get(key) == (from_c, to_c):
            color_chain_idx.pop(key, None)
            color_chain_first.pop(key, None)
            color_chain_last.pop(key, None)
            continue

        if key in color_chain_idx and color_chain_last.get(key) == from_c:
            first_c = color_chain_first[key]
            merged_params = params_pre + f"color({first_c})_({to_c})" + params_post



            _has_secret = "|" in (_desc or "")
            merged_desc = f"{target} gradually changes color from {first_c} to {to_c}."
            if _has_secret:
                merged_desc += f"|Object gradually changes color from {first_c} to {to_c}."
            merged = f"{verb}:[{target}]:[{merged_params}]:{merged_desc}"
            existing_idx = color_chain_idx[key]
            misc_order[existing_idx] = merged
            _suppress_idx.add(idx)
            color_chain_last[key] = to_c
        else:
            color_chain_idx[key]   = idx
            color_chain_first[key] = from_c
            color_chain_last[key]  = to_c
        last_pair_per_key[key] = (from_c, to_c)


    for key, idx in color_chain_idx.items():
        if color_chain_first.get(key) == color_chain_last.get(key):
            _suppress_idx.add(idx)


    active_color_chain_keys: set[tuple[str, str]] = set()
    for key, idx in color_chain_idx.items():
        if idx not in _suppress_idx:
            active_color_chain_keys.add(key)
    if active_color_chain_keys:
        for idx, entry in enumerate(misc_order):
            if idx in _suppress_idx:
                continue
            bm = _RE_BRIGHT_FULL.match(entry)
            if not bm:
                continue
            b_target = bm.group(1)
            b_params = bm.group(2)
            pos_match = _RE_POS_IN_PARAMS.search(b_params)
            b_pos = pos_match.group(1) if pos_match else ""
            if (b_target, b_pos) in active_color_chain_keys:
                _suppress_idx.add(idx)





    shape_chain_idx:   dict[tuple[str, str], int] = {}
    shape_chain_last:  dict[tuple[str, str], str] = {}
    shape_chain_first: dict[tuple[str, str], str] = {}
    shape_last_pair:   dict[tuple[str, str], tuple[str, str]] = {}
    for idx, entry in enumerate(misc_order):
        if idx in _suppress_idx:
            continue
        m = _RE_CHANGE_SHAPE.match(entry)
        if not m:
            continue
        verb, target, params_pre, from_s, to_s, params_post, _desc = m.groups()
        params_full = params_pre + f"shape({from_s})_({to_s})" + params_post
        pos_match = _RE_POS_IN_PARAMS.search(params_full)
        pos_key = pos_match.group(1) if pos_match else ""
        key = (target, pos_key)

        if shape_last_pair.get(key) == (from_s, to_s):
            shape_chain_idx.pop(key, None)
            shape_chain_first.pop(key, None)
            shape_chain_last.pop(key, None)
            continue

        if key in shape_chain_idx and shape_chain_last.get(key) == from_s:
            first_s = shape_chain_first[key]
            merged_params = params_pre + f"shape({first_s})_({to_s})" + params_post

            _has_secret = "|" in (_desc or "")
            merged_desc = f"{target} gradually changes shape from {first_s} to {to_s}."
            if _has_secret:
                merged_desc += f"|Object gradually changes shape from {first_s} to {to_s}."
            merged = f"{verb}:[{target}]:[{merged_params}]:{merged_desc}"
            existing_idx = shape_chain_idx[key]
            misc_order[existing_idx] = merged
            _suppress_idx.add(idx)
            shape_chain_last[key] = to_s
        else:
            shape_chain_idx[key]   = idx
            shape_chain_first[key] = from_s
            shape_chain_last[key]  = to_s
        shape_last_pair[key] = (from_s, to_s)


    for key, idx in shape_chain_idx.items():
        if shape_chain_first.get(key) == shape_chain_last.get(key):
            _suppress_idx.add(idx)

    if _suppress_idx:
        misc_order = [e for i, e in enumerate(misc_order) if i not in _suppress_idx]





    active_shape_targets: set[str] = set()
    for entry in misc_order:
        m = _RE_CHANGE_SHAPE.match(entry)
        if m:
            active_shape_targets.add(m.group(2))
    if active_shape_targets:
        def _is_shape_variant_appear_disappear(entry: str) -> bool:
            m = _APP_DIS_RE.match(entry)
            if not m:
                return False
            ad_target = m.group(1)
            for tgt in active_shape_targets:
                prefix = tgt + "_"
                if ad_target.startswith(prefix) and ad_target.endswith("_shape"):
                    return True
            return False
        misc_order = [e for e in misc_order if not _is_shape_variant_appear_disappear(e)]


    bright_targets: set[str] = set()
    for entry in misc_order:
        m = _BRIGHT_RE.match(entry)
        if m:
            bright_targets.add(m.group(1))
    if bright_targets:
        misc_order = [
            e for e in misc_order
            if not (
                (m2 := _APP_DIS_RE.match(e)) is not None
                and m2.group(1) in bright_targets
            )
        ]

    result.extend(misc_order)
    return result



















_DEFAULT_SUMMARY_TEXTS = {
    "reset":    "[reset]:[game]:[]:The game state is reset.",
    "win":      "[win]:[game]:[]:The level is cleared.",
    "miss":     "[miss]:[game]:[]:Step limit reached, the player loses a life and resets to the start.",
    "gameover": "[gameover]:[game]:[]:All lives lost, game over.",
}


def compose_default_summary(
    agg_en: list[str],
    agg_l2: list[str],
    events: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """






    """
    out_en = list(agg_en)
    out_l2 = list(agg_l2)

    def _resolve(key: str, evt: Any) -> tuple[str, str]:
        text = (evt.get("en") if isinstance(evt, dict) else None) or _DEFAULT_SUMMARY_TEXTS[key]
        return text, text


    reset_evt = events.get("reset")
    if reset_evt:
        en, l2 = _resolve("reset", reset_evt)
        out_en.insert(0, en)
        out_l2.insert(0, l2)


    for key in ("win", "miss", "gameover"):
        evt = events.get(key)
        if not evt:
            continue
        en, l2 = _resolve(key, evt)
        out_en.append(en)
        out_l2.append(l2)






    layout_rows = events.get("initial_layout")
    if layout_rows:
        for r in layout_rows:
            role  = r["role"]
            sname = r["sname"]
            dx, dy = r["dx"], r["dy"]
            dw, dh = r["dw"], r["dh"]


            role_en = r.get("role_en") or " ".join(
                w.capitalize() for w in role.split("_")
            )


            _params = f"sname({sname});position({dx}_{dy});bbox({dw}_{dh})"
            out_en.append(
                f"[layout]:[{role}]:[{_params}]:"
                f"{role_en} is placed at ({dx},{dy}).|Object is placed at ({dx},{dy})."
            )
            out_l2.append(out_en[-1])
    return out_en, out_l2
