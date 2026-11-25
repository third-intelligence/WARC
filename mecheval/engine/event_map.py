# Adapted from the mapping module of a concurrent study (credited once it is published), at an earlier revision.
# MIT License, Copyright (c) 2026 [the authors of a concurrent study, credited once it is published]; the full text is in incl_licenses/LICENSE_instrumented_games.
"""Event maps: the tables in data/event_maps that assign the logged events of ten games to mechanics.

Every logged event is a line "[verb]:[target]:[details]:text". A table row matches a verb, a target and a rule on the
details (param_rule); its columns canonical_<variant>, status_<variant> and scope_<variant> give the mechanic and
whether it is scored.
"""

from __future__ import annotations

import csv

import re

from pathlib import Path


GT_RE = re.compile('^\\[([^]]+)\\]:\\[([^]]+)\\]:\\[([^]]*)\\]:')


def param_ok(rule: str, details: str) -> bool:
    if rule == 'ANY':
        return True
    if rule == 'HAS_EDGES':
        return 'edges(' in details
    if rule == 'NO_EDGES':
        return 'edges(' not in details
    if rule.startswith('SNAME='):
        m = re.search('(?:^|;)sname\\(([^)]+)\\)', details)
        return bool(m and m.group(1) == rule.split('=', 1)[1])
    raise ValueError(f'unsupported param_rule {rule}')


def load_mapping(path: Path, variant: str):
    rows = list(csv.DictReader(path.open(), delimiter='\t'))
    needed = {f'canonical_{variant}', f'status_{variant}', f'scope_{variant}'}
    if not rows or not needed <= set(rows[0]):
        raise ValueError(f'mapping missing columns {sorted(needed)}')
    return rows


def resolve_gt(gt: list, rows: list, variant: str) -> tuple[list[dict], list[dict]]:
    evaluated = []
    audit = []
    for gid, line in enumerate(gt, 1):
        if not isinstance(line, str):
            raise ValueError(f'gt_event_id {gid}: GT line not string')
        m = GT_RE.match(line)
        if not m:
            raise ValueError(f'gt_event_id {gid}: malformed GT line')
        verb, target, details = m.groups()
        hits = [r for r in rows if r['verb'] == verb and r['raw_target'] == target and param_ok(r['param_rule'], details)]
        if len(hits) != 1:
            raise ValueError(f'gt_event_id {gid}: mapping matches={len(hits)} verb={verb} target={target}')
        r = hits[0]
        status = r[f'status_{variant}']
        scope = r[f'scope_{variant}']
        category = r[f'canonical_{variant}']
        inst = []
        for key in ('sname', 'position', 'level'):
            z = re.search(f'(?:^|;){key}\\(([^)]+)\\)', details)
            if z:
                inst.append(f'{key}={z.group(1)}')
        rec = {'gt_event_id': gid, 'category': category, 'source_category': r['source_category'], 'raw_target': target, 'instance': ';'.join(inst) or target, 'scope': scope, 'status': status, 'mapping_source_order': int(r['source_order'])}
        audit.append(rec)
        if status == 'active' and scope in {'MAIN', 'AUX', 'HUD'}:
            evaluated.append(rec)
    return (evaluated, audit)


def declared_from_mapping(rows, variant):
    out = []
    for r in rows:
        if r[f'status_{variant}'] == 'active' and r[f'scope_{variant}'] in {'MAIN', 'AUX', 'HUD'}:
            x = r[f'canonical_{variant}']
            if x not in out:
                out.append(x)
    return out


