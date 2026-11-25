"""Evaluator-only occurrence labels. Never inspect composite-frame differences.

The unit is one canonical category per external action. Object state changes,
source paint writes and executed interaction branches are evidence; ROI overlap
is not. Unbound source events become unknown, not fabricated occurrences.
"""
from functools import lru_cache
import re

from .game_regions import HUD
from .review_mapping import canonical_descriptions

VERSION = 2
FIELDS = ('x', 'y', 'rotation', 'scale', 'pixels', 'visible')


@lru_cache(None)
def definitions(game, level):
    return canonical_descriptions(game, level)


def object_changes(a, b):
    if a is None and b is None:
        return []
    if a is None:
        return ['appearance']
    if b is None:
        return ['disappearance']
    result = []
    for keys, name in ((('x', 'y'), 'movement'), (('rotation',), 'rotation'),
                       (('scale',), 'resize'), (('pixels',), 'appearance_change'),
                       (('visible',), 'visibility_change')):
        if any(a.get(k) != b.get(k) for k in keys):
            result.append(name)
    return result


def class_state(objects, predicate, fields=FIELDS):
    # Restore/merge can replace Python instances. Identical intrinsic endpoint
    # states must not be counted as disappear+appear solely due to identity churn.
    return sorted(repr(tuple(o.get(k) for k in fields)) for o in objects.values() if predicate(o))


def state_row(snapshot, name):
    row = next((r for r in snapshot.get('nonsprite', []) if r['name'] == name), None)
    if row is None:
        return None
    # Ignore geometric bounds; the state/value is the evidence, not its pixels.
    return {k: v for k, v in row.items() if k not in ('x', 'y', 'width', 'height', 'name')}


def detect(before, after, events, annotations, interactions):
    game, level = before['game'], before['level']
    if level != after['level']:
        raise ValueError('occurrence detector accepts within-level endpoints only')
    aa = {o['instance_id']: o for o in before['objects']}
    bb = {o['instance_id']: o for o in after['objects']}
    cats = {c: {'category': c, 'occurred': False, 'evidence': [], 'uncertainties': []}
            for c in definitions(game, level)}

    def emit(cat, rule, kinds=(), instances=(), detail=None):
        if cat not in cats:
            return
        row = cats[cat]
        row['occurred'] = True
        row['evidence'].append({'rule': rule, 'kinds': list(kinds), 'instances': list(instances), 'detail': detail})

    def unknown(cat, reason):
        if cat in cats:
            cats[cat]['uncertainties'].append(reason)

    special = set(HUD[game])
    # Source-level counters, independent of the finite pixel resolution of HUDs.
    for cat, (name, _) in HUD[game].items():
        x, y = state_row(before, name), state_row(after, name)
        if x is None or y is None:
            unknown(cat, 'missing_source_counter:' + name)
        elif x != y:
            emit(cat, 'nonsprite_state_change', ['counter_change'], detail={'name': name, 'before': x, 'after': y})

    if game == 'ar25':
        special |= {'piece', 'reflection', 'selection'}
        for uid in aa.keys() | bb.keys():
            a, b = aa.get(uid), bb.get(uid)
            if 'piece' in (b or a).get('tags', []):
                kinds = object_changes(a, b)
                if kinds:
                    emit('piece', 'piece_intrinsic_state', kinds, [uid])
        def reflection_state(snap, objs):
            return sorted((r['source_instance'], sorted(map(tuple, r['cells'])),
                           repr(objs.get(r['source_instance'], {}).get('pixels')))
                          for r in snap.get('special', {}).get('reflections', []) if r['cells'])
        if reflection_state(before, aa) != reflection_state(after, bb):
            emit('reflection', 'source_reflection_map_change', ['reflection_change'])
        # Pinned ar25 source emits these only inside `before_sel is not after_sel`.
        # Selection need not change rendered pixels; never require a pixel delta.
        if any(e.get('gimmick_id') == 'selection' and e.get('event', '').startswith('[bright]:') for e in events):
            emit('selection', 'executed_selected_identity_change', ['selection'])

    if game == 'dc22':
        special.add('player')
        pred = lambda o: 'player' in o.get('tags', [])
        if class_state(aa, pred) != class_state(bb, pred):
            emit('player', 'player_intrinsic_endpoint_state', ['object_state_change'])

    if game == 'su15':
        special.add('block')
        pred = lambda o: 'fruit' in o.get('tags', [])
        if class_state(aa, pred) != class_state(bb, pred):
            emit('block', 'fruit_intrinsic_endpoint_state', ['object_state_change'])

    if game == 'cd82':
        special |= {'stack_fill', 'reservoir_recolor'}
        writes = after.get('special', {}).get('paint', [])
        if any(r['cells'] for r in writes):
            emit('stack_fill', 'executed_canvas_write', ['paint'],
                 [r['source_instance'] for r in writes if r['source_instance']],
                 detail='includes same-color writes; no composite-pixel gate')
        x, y = state_row(before, 'selected_color'), state_row(after, 'selected_color')
        if x is None or y is None:
            unknown('reservoir_recolor', 'missing_selected_color')
        elif x != y:
            emit('reservoir_recolor', 'selected_color_change', ['recolor'])

    if game == 'r11l':
        special |= {'arm_selection', 'arm'}
        pred = lambda o: o.get('evaluation_roles', {}).get('arm')
        # Selection color has its own category. Here count physical arm changes.
        if class_state(aa, pred, ('x', 'y', 'rotation', 'scale', 'visible')) != class_state(bb, pred, ('x', 'y', 'rotation', 'scale', 'visible')):
            emit('arm', 'arm_physical_endpoint_state', ['physical_change'])
        for uid in aa.keys() | bb.keys():
            a, b = aa.get(uid, {}), bb.get(uid, {})
            if a.get('evaluation_roles', {}).get('selected_arm', False) != b.get('evaluation_roles', {}).get('selected_arm', False):
                emit('arm_selection', 'selected_arm_identity_change', ['selection'], [uid])

    if game == 'ls20':
        special.add('lock_match')
        x, y = before.get('special', {}).get('locks_unlocked'), after.get('special', {}).get('locks_unlocked')
        if x is None or y is None:
            unknown('lock_match', 'missing_lock_state')
        elif x != y:
            emit('lock_match', 'lock_unlocked_state_change', ['unlock'])
        for uid in aa.keys() | bb.keys():
            a, b = aa.get(uid), bb.get(uid)
            if set((b or a).get('tags', [])) & {'lock_highlight', 'lock_frame'}:
                kinds = object_changes(a, b)
                if kinds:
                    emit('lock_match', 'lock_intrinsic_endpoint_state', kinds, [uid])

    if game == 'sc25':
        special |= {'player_movement', 'player_resize', 'spell_slot_grid'}
        for uid in aa.keys() | bb.keys():
            a, b = aa.get(uid), bb.get(uid)
            if (b or a).get('evaluation_roles', {}).get('spell_slot'):
                kinds = object_changes(a, b)
                if kinds:
                    emit('spell_slot_grid', 'spell_slot_intrinsic_endpoint_change', kinds, [uid])
            if not (b or a).get('evaluation_roles', {}).get('player'):
                continue
            kinds = object_changes(a, b)
            if 'resize' in kinds:
                emit('player_resize', 'player_scale_change', ['resize'], [uid])
            elif 'movement' in kinds:
                emit('player_movement', 'player_position_change', ['movement'], [uid])
        for event in interactions.get('events', []):
            if event['reason'] == 'resize_cast':
                emit('player_resize', 'executed_resize_cast', ['resize_attempt'])

    if game == 'wa30':
        special |= {'target_facing', 'target_grab'}
        for uid in aa.keys() | bb.keys():
            a, b = aa.get(uid, {}), bb.get(uid, {})
            x = a.get('evaluation_roles', {}).get('target_border_state')
            y = b.get('evaluation_roles', {}).get('target_border_state')
            if x != y:
                if 'facing' in (x, y):
                    emit('target_facing', 'target_facing_state_change', ['selection'], [uid])
                if {x, y} & {'selector_grab', 'other_grab'}:
                    emit('target_grab', 'target_grab_state_change', ['selection'], [uid])

    if game == 'sb26':
        special |= {'checkbox_selection', 'tile'}
        for uid in aa.keys() | bb.keys():
            a, b = aa.get(uid), bb.get(uid)
            if (b or a)['name'] == 'cursor_box':
                kinds = object_changes(a, b)
                if kinds:
                    emit('checkbox_selection', 'cursor_box_intrinsic_endpoint_change', kinds, [uid])
            if (b or a)['name'] == 'color_tile':
                kinds = object_changes(a, b)
                if kinds:
                    emit('tile', 'color_tile_intrinsic_endpoint_change', kinds, [uid])

    # Source-bound object annotations are present independently of logged changes.
    for region in annotations:
        cat = region['category']
        if cat in special or cat not in cats:
            continue
        for uid in region.get('instances', []):
            if uid not in aa and uid not in bb:
                continue
            kinds = object_changes(aa.get(uid), bb.get(uid))
            if kinds:
                emit(cat, 'annotated_object_state_change', kinds, [uid])

    # Released logs determine category identity, never screen-space overlap.
    # Require independently confirmed intrinsic endpoint state change as well.
    for event in events:
        cat = event.get('gimmick_id')
        if cat not in cats or cat in special:
            continue
        ids = set(event.get('candidates', [])) & (aa.keys() | bb.keys())
        name = re.search(r'sname\(([^)]+)\)', event.get('event', ''))
        if name:
            named = {uid for uid in aa.keys() | bb.keys()
                     if any(o and o['name'] == name[1] for o in (aa.get(uid), bb.get(uid)))}
            if len(named) == 1:
                ids = named
        if len(ids) == 1:
            uid = next(iter(ids))
            a, b = aa.get(uid), bb.get(uid)
            kinds = object_changes(a, b)
            if game == 'ka59' and cat == 'player_center_selection' and a and b:
                # Body movement is not center selection. Compare local center colors.
                def center(o):
                    pixels, mask = o['pixels'], o.get('parts', {}).get('center')
                    if mask is None:
                        return None
                    return [v for row, mr in zip(pixels, mask) for v, use in zip(row, mr) if use]
                ca, cb = center(a), center(b)
                if ca is None or cb is None:
                    unknown(cat, 'center_part_missing')
                elif ca != cb:
                    emit(cat, 'intrinsic_center_color_change', ['selection'], [uid])
            elif kinds:
                verb = event.get('event', '').split(']:', 1)[0].lstrip('[')
                allowed = {'move': {'movement'}, 'rotate': {'rotation', 'appearance_change'},
                           'bright': {'appearance_change', 'visibility_change', 'appearance', 'disappearance'},
                           'appear': {'appearance', 'visibility_change', 'appearance_change'},
                           'disappear': {'disappearance', 'visibility_change', 'appearance_change'},
                           'change': {'appearance_change', 'visibility_change', 'resize'},
                           'shrink': {'resize', 'appearance_change', 'movement'},
                           'extend': {'resize', 'appearance_change', 'movement'}}.get(verb, set(kinds))
                matched = sorted(set(kinds) & allowed)
                if matched:
                    emit(cat, 'source_event_and_object_state', matched, [uid], event.get('event'))
        else:
            # Non-sprite final state (e.g. UI indicator). Only exact names qualify.
            target = re.match(r'^\[[^]]+\]:\[([^]]+)\]', event.get('event', ''))
            key = name[1] if name else target[1] if target else None
            x, y = state_row(before, key), state_row(after, key)
            if x is not None and y is not None:
                if x != y:
                    emit(cat, 'source_event_and_nonsprite_state', ['state_change'], detail=key)
            else:
                unknown(cat, 'ambiguous_or_missing_source_identity:' + event.get('event', ''))

    # Contact with a static obstacle is an event even with an identical image.
    for event in interactions.get('events', []):
        if game == 's5i5' and event['reason'] == 'arm_overlap_rollback':
            ids = [r.get('instance_id') for role in ('actors', 'counterparts') for r in event[role]]
            if any((bb.get(uid) or aa.get(uid) or {}).get('evaluation_roles', {}).get('arm_role') == 'fixed_arm' for uid in ids):
                emit('obstacle', 'executed_fixed_arm_collision', ['collision'], ids)
        cat = event['kind'] + '.' + event['reason']
        cats.setdefault(cat, {'category': cat, 'occurred': False, 'evidence': [], 'uncertainties': []})
        emit(cat, 'executed_source_branch', [event['kind']],
             [r.get('instance_id') for r in event['actors']], event.get('source'))
    for row in cats.values():
        # One proven instance suffices for a category occurrence on this step.
        if not row['occurred'] and row['uncertainties']:
            row['occurred'] = None
        row['rule_version'] = VERSION
    return list(cats.values())


class ExposureCounter:
    """Within-run history persists across levels/reset. Unknown exposure poisons exact x."""
    def __init__(self):
        self.counts = {}
        self.unknown = {}

    def update(self, rows):
        result = []
        for row in rows:
            cat = row['category']
            n, u = self.counts.get(cat, 0), self.unknown.get(cat, 0)
            result.append({**row, 'prior_occurrences_lower_bound': n,
                           'prior_occurrences': n if u == 0 else None,
                           'occurrence_index': n + 1 if row['occurred'] is True and u == 0 else None,
                           'unknown_prior_steps': u})
            self.counts[cat] = n + int(row['occurred'] is True)
            self.unknown[cat] = u + int(row['occurred'] is None)
        return result
