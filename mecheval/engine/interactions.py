"""Evaluator-only branch instrumentation and independent endpoint labels.

Never run this inside a WM/agent. Source files and shared imported classes are
unchanged. The private module evaluates each original condition exactly once.
"""
import ast
import hashlib
import inspect
import json
import sys
import types
import uuid
from pathlib import Path

import numpy as np
from .interaction_rules import RULES
from .mapping import raster, indices
from .endpoint_metrics import bounding_mask, rect_mask

from ..paths import DATA
MANIFEST = DATA / 'interaction_source_manifest.json'

# Explicit, source-audited counters. Do not interpret every nonsprite value as
# health or infer cost from a quantized bar image. Other games stay unmeasured.
BUDGET_PATHS = {
    'ar25': ('zdrbnrjbr', 'current_steps'),
    'sk48': ('qiercdohl',),
    'wa30': ('step_counter', 'current_steps'),
}


def read_budget(game_id, game):
    if game is None:
        return None
    if game_id == 'r11l':
        value = max(0, game._max_actions - game._action_count)
        source = '_max_actions - _action_count (clamped at zero)'
    elif game_id in BUDGET_PATHS:
        value = game
        for name in BUDGET_PATHS[game_id]:
            value = getattr(value, name)
        source = '.'.join(BUDGET_PATHS[game_id])
    else:
        return None
    return {'remaining': int(value), 'level': int(game.level_index + 1),
            'source': source, 'unit': 'game_native_budget'}


def budget_delta(before, after):
    result = {'before': before, 'after': after, 'consumed': None,
              'status': 'unmeasured'}
    if before is not None and after is not None:
        if before['level'] != after['level']:
            result['status'] = 'level_boundary'
        else:
            delta = before['remaining'] - after['remaining']
            result.update(consumed=delta, status='decreased' if delta > 0 else
                          'increased' if delta < 0 else 'unchanged')
    return result


class BranchObserver(ast.NodeTransformer):
    def __init__(self, rules):
        self.rules = {r.line: (i, r) for i, r in enumerate(rules)}
        if len(self.rules) != len(rules):
            raise ValueError('duplicate hook site')
        self.hits = set()

    def generic_visit(self, node):
        line = getattr(node, 'lineno', None)
        # Expressions also have line numbers; select statements only.
        match = self.rules.get(line) if isinstance(node, ast.stmt) else None
        node = super().generic_visit(node)
        if not match:
            return node
        i, rule = match
        call = ast.parse(f'_interaction_observer({i}, locals())').body[0]
        ast.copy_location(call, node)
        self.hits.add(i)
        if rule.branch == 'before':
            return [call, node]
        if not isinstance(node, ast.If):
            raise ValueError(f'expected If at {rule.game}:{rule.line}')
        branch = getattr(node, rule.branch)
        setattr(node, rule.branch, [call] + branch)
        return node


class ArmOverlapObserver(ast.NodeTransformer):
    """Observe the existing short-circuit predicate, without another query."""
    def __init__(self):self.hits=0
    def visit_FunctionDef(self,node):
        if node.name!='_check_arm_overlap':return node
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if ast.unparse(node)=='arm.collides_with(other)':
            self.hits+=1
            return ast.copy_location(ast.Call(ast.Name('_interaction_overlap',ast.Load()),
                [ast.Name('arm',ast.Load()),ast.Name('other',ast.Load()),node],[]),node)
        return node


def private_module(original, observer):
    path = Path(original.__file__)
    source = path.read_bytes()
    expected = json.loads(MANIFEST.read_text())[observer.game_id]
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError('GT source changed; re-audit interaction rules: ' + str(path))
    tree = ast.parse(source, filename=str(path))
    transform = BranchObserver(observer.rules)
    tree = transform.visit(tree)
    if observer.game_id=='s5i5':
        overlap=ArmOverlapObserver();tree=overlap.visit(tree)
        if overlap.hits!=1:raise ValueError('s5i5 overlap predicate changed')
    if transform.hits != set(range(len(observer.rules))):
        raise ValueError('missing interaction hook')
    ast.fix_missing_locations(tree)
    name = original.__name__ + '__interaction_' + uuid.uuid4().hex
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = original.__package__
    module.__dict__['_interaction_observer'] = observer.observe
    module.__dict__['_interaction_overlap'] = observer.observe_overlap
    sys.modules[name] = module  # dataclass/inspection support
    try:
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    observer.module_name = name
    return module


def objects(value):
    if value is None:
        return []
    if isinstance(value, dict):
        value = list(value)
    if isinstance(value, (list, tuple, set)):
        return [s for v in value for s in objects(v)]
    if hasattr(value, 'sprite'):
        value = value.sprite  # tn36 GameEntity
    if not all(hasattr(value, k) for k in ('x', 'y', 'width', 'height', 'name')):
        raise TypeError('interaction actor is not a sprite')
    return [value]


class InteractionObserver:
    def __init__(self, game_id):
        self.game_id = game_id
        self.rules = [r for r in RULES if r.game == game_id]
        self.game = None
        self.module_name = None
        self.records = []
        self._refs = {}
        self._engine_original = None
        self._budget_before = None
        self._last_overlap = None

    def attach(self, game):
        self.game = game
        if self.game_id == 'sc25':
            # Exact return of the real move, not a second geometry query.
            original = game.try_move_sprite
            self._engine_original = original
            def move(sprite, dx, dy):
                attempt = (sprite.x + dx, sprite.y + dy, sprite.width, sprite.height)
                collisions = original(sprite, dx, dy)
                if collisions:
                    self.record('collision', 'engine_solid_contact', sprite, collisions,
                                inspect.getsourcefile(original) + ':try_move_sprite', attempt)
                return collisions
            game.try_move_sprite = move

    def begin_action(self, action=None, available_actions=None):
        self.records = []
        self._refs = {}
        self._last_overlap = None
        self._budget_before = read_budget(self.game_id, self.game)
        if action is not None and available_actions is not None:
            aid = action.id.value
            allowed = {getattr(x, 'value', x) for x in available_actions}
            if aid != 0 and aid not in allowed:
                self.record('invalid_action', 'unsupported_action', None, None,
                            'environment.available_actions before action')

    def describe(self, sprite):
        self._refs[id(sprite)] = sprite
        return {'object_key': id(sprite), 'name': sprite.name,
                'tags': list(sprite.tags or []),
                'rect': [int(sprite.x), int(sprite.y), int(sprite.width), int(sprite.height)]}

    def record(self, kind, reason, actor, other, source, attempt=None):
        actors = [self.describe(s) for s in objects(actor)]
        others = [self.describe(s) for s in objects(other)]
        cam = self.game.camera
        scale, ox, oy = cam._calculate_scale_and_offset()
        self.records.append({'kind': kind, 'reason': reason, 'actors': actors,
            'counterparts': others, 'source': source, 'branch_confirmed': True,
            'attempt_rect': list(attempt) if attempt is not None else None,
            'camera': dict(x=cam.x, y=cam.y, scale=scale, offset_x=ox, offset_y=oy)})

    def observe_overlap(self, actor, other, hit):
        if hit:self._last_overlap=(actor,other)
        return hit

    def observe(self, i, local):
        if self.game is None:
            return  # construction is not an environment interaction
        rule = self.rules[i]
        safe = {'__builtins__': {}, 'bool': bool}
        if not eval(rule.guard, safe, local):
            return
        if self.game_id=='s5i5' and rule.reason=='arm_overlap_rollback' and self._last_overlap:
            actor,other=self._last_overlap
            if actor not in self.game.saved_state and other in self.game.saved_state:
                actor,other=other,actor
            self.record(rule.kind,rule.reason,actor,other,
                f'gamesofts/modify/s5i5/s5i5.py:{rule.line}:{rule.branch}:actual_first_overlap',
                (actor.x,actor.y,actor.width,actor.height))
            return
        self.record(rule.kind, rule.reason, eval(rule.actor, safe, local),
                    eval(rule.other, safe, local),
                    f'gamesofts/modify/{self.game_id}/{self.game_id}.py:{rule.line}:{rule.branch}',
                    eval(rule.attempt, safe, local))

    def finish(self, before, after, tracker):
        records = []
        for raw in self.records:
            record = dict(raw)
            for role in ('actors', 'counterparts'):
                record[role] = [{**{k:v for k,v in row.items() if k != 'object_key'},
                    'instance_id': tracker._ids.get(row['object_key'])} for row in raw[role]]
            records.append(record)
        a = {o['instance_id']: o for o in before['objects']}
        b = {o['instance_id']: o for o in after['objects']}
        # Net displacement only. Rollback/rotation is NOT mislabeled movement.
        movement = [{'instance_id': k, 'name': a[k]['name'],
                     'delta': [b[k]['x']-a[k]['x'], b[k]['y']-a[k]['y']]}
                    for k in sorted(a.keys() & b.keys())
                    if (a[k]['x'], a[k]['y']) != (b[k]['x'], b[k]['y'])]
        invalid = [r for r in records if r['kind'] == 'invalid_action']
        collisions = [r for r in records if r['kind'] == 'collision']
        return {'schema_version': 2, 'events': records, 'net_movement': movement,
                'budget': budget_delta(self._budget_before, read_budget(self.game_id, self.game)),
                'has_collision': bool(collisions), 'has_net_movement': bool(movement),
                'invalid_action': bool(invalid),
                'input_status': 'confirmed_invalid' if invalid else 'not_flagged_not_certified_valid',
                'collision_status': 'observed' if collisions else 'not_observed_not_certified_absent',
                'identity_transition': bool(np.array_equal(before['frame'], after['frame'])),
                'coverage': 'audited_execution_sites_only; masks require human review'}

    def close(self):
        if self._engine_original is not None:
            self.game.try_move_sprite = self._engine_original
        if self.module_name:
            sys.modules.pop(self.module_name, None)


def interaction_annotations(before, after, summary):
    """Separate scoring axis; never changes existing gimmick category unions.

    Primary actor ROI = endpoint bboxes; attempted bbox is context for blocked
    movement. Counterpart ROI is intersected with that local context, never a
    whole maze/background. All masks remain candidate, no semantic certification.
    """
    ra, rb = raster(before), raster(after)
    shape = np.asarray(before['frame']).shape
    empty = np.zeros(shape, bool)
    out = []
    def endpoint(uid):
        result = empty.copy()
        for r in (ra, rb):
            if uid in r['objects']:
                result |= bounding_mask(r['objects'][uid]['amodal'])
        return result
    def project(rect, cam):
        x,y,w,h = rect; s=cam['scale']
        return rect_mask(((x-cam['x'])*s+cam['offset_x'],
                          (y-cam['y'])*s+cam['offset_y'],w*s,h*s), shape)
    for move in summary['net_movement']:
        mask = endpoint(move['instance_id'])
        out.append(dict(category='movement', id='movement:'+move['instance_id'],
            instances=[move['instance_id']], bbox=indices(mask), support=indices(mask),
            local=indices(mask), visible=[], verified=False, visibility_status='not_yet_resolved',
            source='endpoint net object displacement', bbox_mode='parent_before_after'))
    for i, event in enumerate(summary['events']):
        if event['kind'] not in {'collision', 'constraint', 'contact'}:
            continue  # invalid action gets full-frame identity diagnostic, not guessed ROI
        mask = empty.copy()
        for actor in event['actors']:
            mask |= endpoint(actor['instance_id'])
        if event['attempt_rect'] is not None:
            mask |= project(event['attempt_rect'], event['camera'])
        elif event['kind'] == 'collision':
            # For rollback branches, the actor is already at attempted position.
            for actor in event['actors']:
                mask |= project(actor['rect'], event['camera'])
        local_counterpart = empty.copy()
        for other in event['counterparts']:
            local_counterpart |= project(other['rect'], event['camera']) & mask
        # Retain both actor and counterpart masks for review; their union is mask.
        if mask.any():
            out.append(dict(category=event['kind']+'.'+event['reason'], id=f'interaction:{i}',
                instances=[a['instance_id'] for a in event['actors'] if a['instance_id']],
                bbox=indices(mask), support=indices(mask), local=indices(mask), visible=[],
                counterpart_local=indices(local_counterpart), verified=False,
                visibility_status='not_yet_resolved', source=event['source'],
                bbox_mode='endpoint_plus_attempt_context_candidate'))
    return out
