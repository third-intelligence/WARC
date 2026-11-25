"""MechEval overlay on the collector: count mechanics that fire but the collector does not attribute.

The other engine modules stay unchanged; install() patches three functions in the worker process
before the replay runs:

  collect_samples.migrate_events  adds bound events (one per object, or one region of
      cells) for the changes listed below, so the region builder takes their
      before / after extent as usual;
  gimmick_occurrences.detect      counts an occurrence of such a category when its object's
      settled state changes (or, for a cell region, when its cells change), adding the
      category's row when the catalog lacks it;
  collect_samples.endpoint_regions  adjusts the regions where two mechanics share cells (v5).

v1
  - dc22 course events that the collector withholds (no single object can be named) are assigned by
    the change they log: a [change] of pattern (the checkered / solid blue tiles the blue
    button toggles) -> color_tile_visibility, a [rotate] (the red and pink pieces turning
    around their dark centre) -> bridge.
  - tr87 slot1_glyph_cycle: the glyph is replaced in place by another sprite, which the collector
    reads as an appearance that a [change] event may not have; a same-place swap counts.
v2  tn36 reference: in level 2 the reference pose (object znhdjhhoai) changes when a program
    is selected and keeps the new pose, with no logged event.
v3 (after a review of the region pages)
  - Object families. Every changed object of the family joins the category's region and
    counts as an occurrence, since the collector binds a logged event to one object (or a subset)
    while the whole family moves:
      lp85 tiles <- every 'tile' (a ring of about 20 tiles moves per action);
      s5i5 extending_arm <- every 'arm' and the 'target_hole' at its tip;
      su15 block <- every 'fruit' (small fruits sucked into a larger one);
      sk48 skewer <- every 'skewer_segment' (the bar extends and retracts while the
        skewer's head stays put, which the collector did not count).
  - New or restored mechanics, counted from the objects' settled state:
      dc22 key_pickup: a key ('zbhi') leaves the board and its button ('jpug') appears in
        the right panel (new; not in the catalog);
      wa30 orange_agent: the orange agent moves (new; logged by the program, no category);
      su15 plus_cursor: the white cross hint ('hint') moves and stays (restored; it had been
        taken for an intermediate-only change because its logged object does not change).
  - r11l connection_line (restored): the string between the arm and the body is drawn on the
    background, not as an object, so its cells are the changed cells outside every changed
    object's extent that touch the arm or the body.
v4
  - ls20 reset_to_start (new): when the steps run out, the player jumps back to the start and
    the board is restored (refills reappear, the HUD piece resets). A player displacement
    larger than one move (5 cells) marks it; the region is every changed object plus the
    changed cells outside them.
  - dc22 switch_highlight (new): pressing the blue button removes the highlight frames around
    the switches in the right panel (objects tagged 'gate', logged as [disappear]:
    [switch_highlight]); they come back only with a RESET. The region is the frames.
    A fall (the course turned under the player) rebuilds the scene as it was; only the step
    bar changes, which budget_bar already scores, so it adds no mechanic.
  - A scene counts as rebuilt only when over 80% of its objects change identity (sk48 recreates
    a skewer's segments with new identities, which the earlier 50% took for a rebuild).
  - The r11l string is dotted, so its cells are the loose changed components reached from the
    arm or the body in hops of at most three cells.
  - r11l goal_flash (restored): when a robot body reaches a goal, a dark frame ('flash-goal')
    covers the goal and stays until the body leaves; it had been taken for an
    intermediate-only flash.
  - s5i5 extending_arm also takes the changed cells inside any arm's or target hole's extent
    when one of them changes (an arm passing over another).
  Steps at which the whole scene is rebuilt add nothing else.
v5 (after a review of 279 example transitions)
  When two mechanics fire on the same cells, the cells go to the mechanic that draws them,
  not to a marker drawn on or around it; and a mechanic fires only when its own change shows.
  - r11l robot (merges arm, body and connection_line): the legs, the body and the string
    drawn between them move together at every step (arm and body fire together at all 280
    scored steps, the string at 268), so they are one mechanic; its region is theirs combined.
    The goal flash under the robot counts for both (second review).
  - ls20 step_refill: stepping on the refill restores the step bar; at that step the bar's
    cells go to step_refill (the refill's own cells lie under the player and stay the player's),
    and steps_bar, still counted as an exposure, is not scored. When the player then leaves
    the cell, the refill is gone rather than drawn again: that step is a second occurrence of
    step_refill on the refill's cells (second review).
  - HUD bars: a bar's cells count only for its counter, not for an object whose box covers the
    bar (vc33 yellow_bird_blob, s5i5 extending_arm); the ls20 refill and reset, which restore
    the step bar, keep it.
  - sb26 checkbox_selection: when a tile moves into a slot, the cursor's region leaves out the
    tile's cells (the tile mechanic scores them) and keeps the cursor frame.
  - ka59 edge_indicator: its region is the band, the cells of the player whose colour changed
    apart from its move (its pixels after against before), not the player's whole extent.
  - cd82 container_tilt fires only when the bucket's footprint changes; the game redraws the
    bucket as a new object when only its colour changes, which reservoir_recolor scores.
  - wa30 target_facing fires only between free and facing; facing <-> grabbed is target_grab.
  - ka59 pushed_block (new): when the selected block (white centre) bumps into another block, the
    other block slides on while the selected one stays; that slide is pushed_block, not player,
    which keeps only the selected block's own moves (37 of 774 player firings were pushes).
v6 (fixes after a check by two raters)
  - vc33 yellow_bird_blob is split by object: boat (the 'joint' sprites, the dark boat-like
    shape with a coloured block) and waves (the 'pipe_slot' areas whose edge moves). Both fire
    when both change.
  - sk48 skewer is split by what moves: skewer_move when the skewer head ('skewer') moves, the
    whole bar with it (up or down, pushing blocks); skewer_extend when segments appear or
    disappear, or shift along the bar while the head stays. A step with both counts for both;
    skewer_extend then takes only the segments that appeared or disappeared.
  - lp85 tiles is split by direction: tiles_left and tiles_right. The tile that wraps around
    (a displacement over twice the median) is left out; on a ring the sense of rotation decides
    (counter-clockwise = left), on a row or column the sign of the shift (-x or -y = left).
  - su15 plus_cursor is dropped (the white cross is a one-time hint). block is split into
    block_move (fruits move) and block_merge (the number of fruits changes: small fruits merge
    into a larger one, their movement in that step included).
  - ar25: reflection is no longer its own mechanic. piece_with_shadow is the piece together with
    the reflection it causes, mirror_with_shadow the mirror axis together with its reflection
    (the three fired together at every step: piece+reflection 512, mirror+reflection 210, piece
    alone 12, reflection alone 0 in the seed-0 scoring). The region is the union.
  - ka59 edge_indicator leaves out the centre cells of the players (parts['center']), which the
    selection toggles; player_center_selection is scored on those centre cells only (the centres
    whose colour changed, else every changed player's centre), not on the players' boxes.
  - dc22 key_pickup leaves out the cells the player covers after the pickup (the player stands on
    the key's cells), as sb26's cursor leaves out the tile.
  74 scored mechanics become 76.

Everything the collector already counts is left as it is. OVERLAY_VERSION is stored on every row
this module adds, so scores made with different versions never mix.

    from .overlay import install; install()   # in each worker, before the engine imports run
"""
import re

import numpy as np

OVERLAY_VERSION = 6
COURSE_KIND = {'change': 'color_tile_visibility', 'rotate': 'bridge'}
VERB = re.compile(r'^\[([^]]+)\]')
SNAME = re.compile(r'sname\(([^)]+)\)')
# game -> [(category, test on an object)]: every changed matching object joins the category
FAMILY = {
    'lp85': [('tiles', lambda o: 'tile' in o.get('tags', []))],
    's5i5': [('extending_arm', lambda o: bool({'arm', 'target_hole'} & set(o.get('tags', []))))],
    'su15': [('block', lambda o: 'fruit' in o.get('tags', []))],           # plus_cursor dropped in v6
    'sk48': [('skewer', lambda o: 'skewer_segment' in o.get('tags', []))],
    'wa30': [('orange_agent', lambda o: o['name'] == 'orange_agent')],
    'tn36': [('reference', lambda o: o['name'] in {'znhdjhhoai', 'reference'})],
    'r11l': [('goal_flash', lambda o: o['name'].startswith('flash-goal'))],
}
NEW_CATEGORIES = {('dc22', 'key_pickup'), ('wa30', 'orange_agent'),
                  ('r11l', 'connection_line'), ('ls20', 'reset_to_start'), ('dc22', 'switch_highlight'),
                  ('r11l', 'robot'), ('ka59', 'pushed_block'),
                  # v6 splits
                  ('vc33', 'boat'), ('vc33', 'waves'), ('sk48', 'skewer_move'), ('sk48', 'skewer_extend'),
                  ('lp85', 'tiles_left'), ('lp85', 'tiles_right'), ('su15', 'block_move'), ('su15', 'block_merge'),
                  ('ar25', 'piece_with_shadow'), ('ar25', 'mirror_with_shadow')}
# v6: the split categories and what they replace
SPLIT_PARENTS = {'vc33': 'yellow_bird_blob', 'sk48': 'skewer', 'lp85': 'tiles', 'su15': 'block'}
ROBOT = ('arm', 'body', 'connection_line')     # r11l: one mechanic since v5
GRAB = {'selector_grab', 'other_grab'}         # wa30 target border states
KEYS = ('bbox', 'support', 'local', 'visible')
# per recording (install() resets it): ls20 refills picked up and not yet left, and the refill
# cells to score at the current step
_STATE = {'consumed': [], 'refill_cells': None}
REBUILT_SHARE = 0.8
MOVE = 5                       # ls20: one move shifts the player by 5 cells


def _objects(snapshot):
    return {o['instance_id']: o for o in snapshot['objects']}


def _scene_rebuilt(aa, bb):
    """True when most objects change identity (a reload), so no single change can be named."""
    keys = aa.keys() | bb.keys()
    return bool(keys) and len(aa.keys() ^ bb.keys()) > REBUILT_SHARE*len(keys)


def _object_row(game, cat, uid, rule, base=None):
    return {**(base or {'kind': 'event', 'event': '', 'category_status': 'mapped', 'mapping_verified': False,
                        'roi': [], 'changed': []}),
            'id': f'overlay:{game}:{cat}:{uid}', 'name': cat, 'gimmick_id': cat, 'candidates': [uid], 'parent': uid,
            'binding_status': 'candidate_whole_object', 'source_rule': {'overlay': OVERLAY_VERSION, 'rule': rule},
            'review_disposition': f'overlay_{rule}', 'overlay': OVERLAY_VERSION}


def _extent_mask(o, shape):
    m = np.zeros(shape, bool)
    pixels = (o or {}).get('pixels') or []
    if o and pixels and pixels[0] and o.get('visible', True) is not False:
        y, x = int(o['y']), int(o['x'])
        m[max(0, y):max(0, y + len(pixels)), max(0, x):max(0, x + len(pixels[0]))] = True
    return m


def _course_rows(row, aa, bb, object_changes):
    verb = VERB.match(row.get('event', ''))
    cat = COURSE_KIND.get(verb[1] if verb else None)
    if cat is None or (cat == 'color_tile_visibility' and 'pattern(' not in row.get('event', '')):
        return []
    name = SNAME.search(row.get('event', ''))
    stem = name[1] if name else ''
    out = []
    for uid in sorted(aa.keys() & bb.keys()):
        o = bb[uid]
        if not object_changes(aa[uid], o) or 'zone' not in o.get('tags', []):
            continue
        if cat == 'color_tile_visibility' and not o['name'].startswith(stem):
            continue
        out.append(dict(_object_row('dc22', cat, uid, 'dc22_change_kind', base=row), id=f"{row['id']}:ov{len(out)}",
                        component='course_gimmick'))
    return out


def _string_row(before, after, aa, bb, object_changes):
    """r11l: the string between arm and body, drawn on the background rather than as an object."""
    from .endpoint_metrics import frame_array
    a, b = frame_array(before['frame']), frame_array(after['frame'])
    diff = a != b
    covered = np.zeros(diff.shape, bool)
    robot = np.zeros(diff.shape, bool)
    for uid in aa.keys() | bb.keys():
        o = bb.get(uid) or aa.get(uid)
        ext = _extent_mask(aa.get(uid), diff.shape) | _extent_mask(bb.get(uid), diff.shape)
        if object_changes(aa.get(uid), bb.get(uid)):
            covered |= ext
        if o['name'].startswith(('robot_leg', 'robot_body')):
            robot |= ext
    loose = diff & ~covered
    if not loose.any() or not robot.any():
        return None
    from scipy import ndimage
    labels, n = ndimage.label(loose, structure=np.ones((3, 3)))
    keep, reach = np.zeros(diff.shape, bool), robot
    while True:                  # follow the dots: each hop at most three cells
        near = ndimage.binary_dilation(reach, structure=np.ones((3, 3)), iterations=3)
        grown = np.isin(labels, [k for k in np.unique(labels[near & (labels > 0)])])
        if not (grown & ~keep).any():
            break
        keep |= grown
        reach = robot | keep
    if not keep.any():
        return None
    return _cells_row('r11l', 'connection_line', keep, 'r11l_string_cells')


def _cells_row(game, cat, mask, rule):
    cells = [int(i) for i in np.flatnonzero(mask)]
    return {'id': f'overlay:{game}:{cat}:cells', 'kind': 'event', 'event': '', 'name': cat, 'gimmick_id': cat,
            'category_status': 'mapped', 'mapping_verified': False, 'roi': cells, 'changed': cells,
            'binding_status': 'overlay_cells', 'candidates': [],
            'source_rule': {'overlay': OVERLAY_VERSION, 'rule': rule},
            'review_disposition': f'overlay_{rule}', 'overlay': OVERLAY_VERSION}


def _place(snapshot, obj, local, shape):
    """A mask in a sprite's own pixels placed on the frame, with the camera as mapping.raster places it."""
    c = snapshot['camera']
    scale, vx, vy = int(c['scale']), int(c['offset_x']), int(c['offset_y'])
    x, y = int((obj['x'] - c['x'])*scale + vx), int((obj['y'] - c['y'])*scale + vy)
    big = np.asarray(local, bool).repeat(scale, 0).repeat(scale, 1)
    out = np.zeros(shape, bool)
    h, w = shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + big.shape[1]), min(h, y + big.shape[0])
    if x1 > x0 and y1 > y0 and obj.get('visible', True) is not False:
        out[y0:y1, x0:x1] = big[y0 - y:y1 - y, x0 - x:x1 - x]
    return out


def _recoloured(before, after, a, b, shape):
    """Cells of an object whose colour changed apart from its move: its pixels after against before."""
    pa, pb = np.asarray(a['pixels'], int), np.asarray(b['pixels'], int)
    if pa.shape != pb.shape:
        return _place(after, b, pb >= 0, shape)
    return _place(after, b, (pa != pb) & ((pa >= 0) | (pb >= 0)), shape)


def _footprint(o):
    return int(o['x']), int(o['y']), np.asarray(o['pixels'], int) >= 0


def _same_footprint(a, b):
    (xa, ya, ma), (xb, yb, mb) = _footprint(a), _footprint(b)
    return (xa, ya) == (xb, yb) and ma.shape == mb.shape and bool((ma == mb).all())


def _pushed(aa, bb):
    """ka59: blocks that moved without being the selected one (white centre before the move)."""
    def selected(o):
        centre = np.asarray(o.get('parts', {}).get('center', []), bool)
        pixels = np.asarray(o['pixels'], int)
        return centre.shape == pixels.shape and bool(centre.any()) and int(pixels[centre][0]) == 0
    return sorted(u for u in aa.keys() & bb.keys() if 'player' in bb[u].get('tags', [])
                  and (aa[u]['x'], aa[u]['y']) != (bb[u]['x'], bb[u]['y']) and not selected(aa[u]))


def _extent(snapshot, o, shape):
    """The bounding box of an object's full extent on the frame."""
    from .endpoint_metrics import bounding_mask
    return bounding_mask(_place(snapshot, o, np.asarray(o['pixels'], int) >= 0, shape))


def _diff(before, after):
    from .endpoint_metrics import frame_array
    return frame_array(before['frame']) != frame_array(after['frame'])


# ---- v6 splits: the same classifier serves detect (which category fires) and endpoint_regions (its cells)

def _tags(aa, bb, uid):
    return set((bb.get(uid) or aa.get(uid)).get('tags', []))


def _moved(aa, bb, uid):
    return uid in aa and uid in bb and (aa[uid]['x'], aa[uid]['y']) != (bb[uid]['x'], bb[uid]['y'])


def _changed(aa, bb, object_changes):
    return [u for u in aa.keys() | bb.keys() if object_changes(aa.get(u), bb.get(u))]


def _vc33_split(aa, bb, object_changes):
    """Changed joints (the boat) and changed pipe slots (the waves)."""
    ch = _changed(aa, bb, object_changes)
    return ([u for u in ch if 'joint' in _tags(aa, bb, u)], [u for u in ch if 'pipe_slot' in _tags(aa, bb, u)])


def _sk48_split(aa, bb, object_changes):
    """move: the head moved (whole bar with it); extend: segments appeared, disappeared or shifted with the head still."""
    ch = _changed(aa, bb, object_changes)
    head = [u for u in ch if 'skewer' in _tags(aa, bb, u) and _moved(aa, bb, u)]
    segs = [u for u in ch if 'skewer_segment' in _tags(aa, bb, u)]
    count = [u for u in segs if u not in aa or u not in bb]
    move = bool(head)
    extend = bool(count) or (not move and bool(segs))
    return dict(move=move, extend=extend, head=head, segs=segs, count=count,
                skewer=[u for u in ch if {'skewer', 'skewer_segment'} & _tags(aa, bb, u)])


def _lp85_direction(aa, bb):
    """tiles_left / tiles_right from the tiles' displacements; the wrap-around tile is left out."""
    pts = [(np.array([aa[u]['x'], aa[u]['y']], float), np.array([bb[u]['x'], bb[u]['y']], float))
           for u in aa.keys() & bb.keys() if _moved(aa, bb, u) and ({'tile', 'goal'} & set(bb[u].get('tags', [])))]
    if not pts:
        return None
    d = np.array([q - p for p, q in pts])
    mag = np.abs(d).sum(1)
    keep = mag <= 2*np.median(mag)
    P, D = np.array([p for p, _ in pts])[keep], d[keep]
    if np.ptp(P[:, 0]) > 0 and np.ptp(P[:, 1]) > 0:          # a ring: the sense of rotation
        c = P.mean(0)
        s = float(((P - c)[:, 0]*D[:, 1] - (P - c)[:, 1]*D[:, 0]).sum())
    else:                                                     # a row or a column: the sign of the shift
        s = float(D[:, 0 if np.ptp(P[:, 1]) == 0 else 1].sum())
    return None if s == 0 else ('tiles_right' if s > 0 else 'tiles_left')


def _su15_split(aa, bb, object_changes):
    fruits = [u for u in _changed(aa, bb, object_changes) if 'fruit' in _tags(aa, bb, u)]
    if not fruits:
        return None
    return 'block_merge' if any(u not in aa or u not in bb for u in fruits) else 'block_move'


def _centre(snapshot, o, shape):
    """The cells of a ka59 player's centre part on the frame (empty when the object has no centre part)."""
    centre = np.asarray((o.get('parts') or {}).get('center', []), bool)
    pixels = np.asarray(o['pixels'], int)
    if centre.shape != pixels.shape or not centre.any():
        return np.zeros(shape, bool)
    return _place(snapshot, o, centre, shape)


def _extents(before, after, aa, bb, uids, shape):
    m = np.zeros(shape, bool)
    for u in uids:
        if u in aa:
            m |= _extent(before, aa[u], shape)
        if u in bb:
            m |= _extent(after, bb[u], shape)
    return m


def install():
    from . import collect_samples
    from . import gimmick_occurrences
    _STATE['consumed'], _STATE['refill_cells'] = [], None      # every caller installs once per recording
    if getattr(collect_samples, '_mecheval_overlay', None) == OVERLAY_VERSION:
        return
    object_changes = gimmick_occurrences.object_changes
    migrate, detect = collect_samples.migrate_events, gimmick_occurrences.detect
    regions_of = collect_samples.endpoint_regions

    def migrate_events(game_id, raw_events, before, after):
        kept, withheld = migrate(game_id, raw_events, before, after)
        game = str(game_id)[:4]
        aa, bb = _objects(before), _objects(after)
        if _scene_rebuilt(aa, bb):
            return kept, withheld
        changed = {u: object_changes(aa.get(u), bb.get(u)) for u in aa.keys() | bb.keys()}
        changed = {u: k for u, k in changed.items() if k}
        if game == 'ls20':
            players = [u for u in aa.keys() & bb.keys() if aa[u]['name'] == 'player']
            if any(abs(bb[u]['x'] - aa[u]['x']) + abs(bb[u]['y'] - aa[u]['y']) > MOVE for u in players):
                kept += [_object_row(game, 'reset_to_start', u, 'ls20_reset_to_start') for u in sorted(changed)]
                diff = _diff(before, after)
                for u in changed:
                    diff &= ~(_extent_mask(aa.get(u), diff.shape) | _extent_mask(bb.get(u), diff.shape))
                if diff.any():
                    kept.append(_cells_row(game, 'reset_to_start', diff, 'ls20_reset_to_start_cells'))
                return kept, withheld
        if game == 's5i5' and any({'arm', 'target_hole'} & set((bb.get(u) or aa.get(u)).get('tags', [])) for u in changed):
            diff = _diff(before, after)
            arms = np.zeros(diff.shape, bool)
            for u in aa.keys() | bb.keys():
                if {'arm', 'target_hole'} & set((bb.get(u) or aa.get(u)).get('tags', [])):
                    arms |= _extent_mask(aa.get(u), diff.shape) | _extent_mask(bb.get(u), diff.shape)
            if (diff & arms).any():
                kept.append(_cells_row(game, 'extending_arm', diff & arms, 's5i5_arm_cells'))
        for cat, test in FAMILY.get(game, []):
            for uid in sorted(changed):
                if test(bb.get(uid) or aa.get(uid)):
                    kept.append(_object_row(game, cat, uid, f'{game}_{cat}_family'))
        if game == 'dc22':
            keys = [u for u, k in changed.items() if 'disappearance' in k and 'zbhi' in (aa[u].get('tags', []))]
            if keys:
                doors = [u for u in changed if 'jpug' in (bb.get(u) or aa.get(u)).get('tags', [])]
                kept += [_object_row(game, 'key_pickup', u, 'dc22_key_pickup') for u in sorted(keys + doors)]
            kept += [_object_row(game, 'switch_highlight', u, 'dc22_switch_highlight') for u in sorted(changed)
                     if 'gate' in (bb.get(u) or aa.get(u)).get('tags', [])]
            still = []
            for row in withheld:
                rows = _course_rows(row, aa, bb, object_changes) if row.get('review_disposition') in (
                    'pending_dc22_endpoint_identity', 'pending_dc22_mixed_object_types') else []
                if rows:
                    kept.extend(rows)
                else:
                    still.append(row)
            withheld = still
        if game == 'r11l':
            row = _string_row(before, after, aa, bb, object_changes)
            if row:
                kept.append(row)
        return kept, withheld

    def detect_with_overlay(before, after, events, annotations, interactions):
        rows = detect(before, after, events, annotations, interactions)
        by = {r['category']: r for r in rows}
        aa, bb = _objects(before), _objects(after)
        game = before['game']

        def emit(cat, rule, kinds, instances, detail=None):
            r = by.get(cat)
            if r is None:
                if (game, cat) not in NEW_CATEGORIES:
                    return
                r = by[cat] = {'category': cat, 'occurred': False, 'evidence': [], 'uncertainties': [],
                               'rule_version': 2}
                rows.append(r)
            if r['occurred'] is True:
                return
            r['occurred'] = True
            r['evidence'].append({'rule': rule, 'kinds': kinds, 'instances': instances, 'detail': detail,
                                  'overlay': OVERLAY_VERSION})

        for ev in events:
            cat = ev.get('gimmick_id')
            if ev.get('overlay') == OVERLAY_VERSION:
                if ev.get('parent'):
                    uid = ev['parent']
                    kinds = object_changes(aa.get(uid), bb.get(uid))
                    if kinds:
                        emit(cat, ev['source_rule']['rule'], kinds, [uid], ev.get('event'))
                elif ev.get('changed'):
                    emit(cat, ev['source_rule']['rule'], ['state_change'], [], len(ev['changed']))
            elif game == 'tr87' and cat == 'slot1_glyph_cycle':
                verb = VERB.match(ev.get('event', ''))
                cands = [u for u in ev.get('candidates', []) if u in bb and u not in aa]
                if not verb or verb[1] != 'change' or len(cands) != 1:
                    continue
                new = bb[cands[0]]
                gone = [u for u, o in aa.items() if u not in bb and (o.get('x'), o.get('y')) == (new.get('x'), new.get('y'))]
                if gone:
                    emit(cat, 'overlay_same_place_sprite_swap', ['appearance_change'], gone + cands, ev.get('event'))

        def drop(cat, reason):
            by[cat]['occurred'] = False
            by[cat].setdefault('uncertainties', []).append({'overlay': OVERLAY_VERSION, 'dropped': reason})

        r = by.get('container_tilt')
        if game == 'cd82' and r and r['occurred'] is True:
            gone = [u for e in r['evidence'] for u in e.get('instances', []) if u in aa and u not in bb]
            new = [u for e in r['evidence'] for u in e.get('instances', []) if u in bb and u not in aa]
            kinds = {k for e in r['evidence'] for k in e.get('kinds', [])}
            if gone and new and kinds <= {'appearance', 'disappearance', 'recolor', 'appearance_change'} and all(
                    any(_same_footprint(aa[g], bb[n]) for n in new) for g in gone):
                drop('container_tilt', 'bucket redrawn with the same footprint (a recolor)')
        if game == 'ls20':
            shape = _diff(before, after).shape
            players = [u for u in aa.keys() & bb.keys() if aa[u]['name'] == 'player']
            if any(abs(bb[u]['x'] - aa[u]['x']) + abs(bb[u]['y'] - aa[u]['y']) > MOVE for u in players):
                _STATE['consumed'] = []                      # back to the start: the refills come back
            r = by.get('step_refill')
            if r and r['occurred'] is True:
                for u in aa.keys() - bb.keys():
                    if aa[u]['name'] == 'step_refill':
                        _STATE['consumed'].append(_place(before, aa[u], np.asarray(aa[u]['pixels'], int) >= 0, shape))
            elif _STATE['consumed'] and players:
                diff = _diff(before, after)
                here = [_place(after, bb[u], np.asarray(bb[u]['pixels'], int) >= 0, shape) for u in players]
                left = [m for m in _STATE['consumed'] if not any((h & m).any() for h in here) and (diff & m).any()]
                if left:
                    _STATE['refill_cells'] = np.logical_or.reduce(left) & diff
                    _STATE['consumed'] = [m for m in _STATE['consumed'] if not any(m is x for x in left)]
                    emit('step_refill', 'ls20_refill_gone', ['disappearance'], [])
        r = by.get('player')
        if game == 'ka59' and r and r['occurred'] is True:
            pushed = _pushed(aa, bb)
            if pushed:
                emit('pushed_block', 'ka59_pushed_block', ['movement'], pushed)
                own = [u for u in aa.keys() & bb.keys() if 'player' in bb[u].get('tags', [])
                       and (aa[u]['x'], aa[u]['y']) != (bb[u]['x'], bb[u]['y']) and u not in pushed]
                if not own:
                    drop('player', 'only pushed blocks moved (pushed_block)')
        r = by.get('target_facing')
        if game == 'wa30' and r and r['occurred'] is True:
            state = lambda o: (o or {}).get('evaluation_roles', {}).get('target_border_state')  # noqa: E731
            free_facing = [u for u in aa.keys() | bb.keys() if state(aa.get(u)) != state(bb.get(u))
                           and 'facing' in (state(aa.get(u)), state(bb.get(u)))
                           and not {state(aa.get(u)), state(bb.get(u))} & GRAB]
            if not free_facing:
                drop('target_facing', 'facing <-> grabbed is target_grab')
        if game == 'r11l':
            parts = [by[c] for c in ROBOT if c in by and by[c]['occurred'] is True]
            if parts:
                emit('robot', 'r11l_robot', sorted({k for p in parts for e in p['evidence'] for k in e.get('kinds', [])}),
                     sorted({i for p in parts for e in p['evidence'] for i in e.get('instances', [])}))
            for c in ROBOT:
                if c in by and by[c]['occurred'] is True:
                    drop(c, 'merged into robot')
        # v6 splits
        fired = lambda c: c in by and by[c]['occurred'] is True  # noqa: E731
        if game == 'vc33' and fired('yellow_bird_blob'):
            joints, slots = _vc33_split(aa, bb, object_changes)
            if joints:
                emit('boat', 'vc33_boat', ['movement'], joints)
            if slots:
                emit('waves', 'vc33_waves', ['appearance_change'], slots)
            if joints or slots:
                drop('yellow_bird_blob', 'split into boat / waves')
        if game == 'sk48' and fired('skewer'):
            s = _sk48_split(aa, bb, object_changes)
            if s['move']:
                emit('skewer_move', 'sk48_skewer_move', ['movement'], s['head'] + s['segs'])
            if s['extend']:
                emit('skewer_extend', 'sk48_skewer_extend', ['appearance' if s['count'] else 'movement'],
                     s['count'] or s['segs'])
            if s['move'] or s['extend']:
                drop('skewer', 'split into skewer_move / skewer_extend')
        if game == 'lp85' and fired('tiles'):
            direction = _lp85_direction(aa, bb)
            if direction:
                emit(direction, 'lp85_direction', ['movement'], [u for u in aa.keys() & bb.keys() if _moved(aa, bb, u)])
                drop('tiles', f'split into {direction}')
        if game == 'su15':
            if fired('plus_cursor'):
                drop('plus_cursor', 'one-time hint, not a mechanic (v6)')
            if fired('block'):
                kind = _su15_split(aa, bb, object_changes)
                if kind:
                    emit(kind, 'su15_' + kind, ['appearance' if kind == 'block_merge' else 'movement'],
                         [u for u in _changed(aa, bb, object_changes) if 'fruit' in _tags(aa, bb, u)])
                    drop('block', f'split into {kind}')
        if game == 'ar25':
            if fired('piece'):
                emit('piece_with_shadow', 'ar25_piece_with_shadow',
                     sorted({k for e in by['piece']['evidence'] for k in e.get('kinds', [])}) or ['movement'],
                     sorted({i for e in by['piece']['evidence'] for i in e.get('instances', [])}))
                drop('piece', 'merged into piece_with_shadow')
            if fired('mirror_axis'):
                emit('mirror_with_shadow', 'ar25_mirror_with_shadow',
                     sorted({k for e in by['mirror_axis']['evidence'] for k in e.get('kinds', [])}) or ['movement'],
                     sorted({i for e in by['mirror_axis']['evidence'] for i in e.get('instances', [])}))
                drop('mirror_axis', 'merged into mirror_with_shadow')
            if fired('reflection'):
                drop('reflection', 'merged into piece_with_shadow / mirror_with_shadow')
        return rows

    def endpoint_regions(before, after, events, groups=None, annotations=()):
        from .mapping import indices, mask_from_indices
        regions = regions_of(before, after, events, groups=groups or (), annotations=annotations)
        if groups is None:     # the first pass only names the categories that joint groups may use
            return regions
        cats, game = regions['categories'], str(before.get('game', ''))[:4]
        diff = _diff(before, after)
        shape = diff.shape

        def mask(cat, key='bbox'):
            return mask_from_indices(cats[cat][key], shape)

        def assign(cat, masks):
            for k in KEYS:
                cats[cat][k] = indices(masks[k] if isinstance(masks, dict) else masks)

        if game == 'ls20' and 'step_refill' in cats and 'steps_bar' in cats:
            assign('step_refill', {k: mask('steps_bar', k) for k in KEYS})
            assign('steps_bar', np.zeros(shape, bool))
        if game == 'ls20' and _STATE['refill_cells'] is not None:
            cells, _STATE['refill_cells'] = _STATE['refill_cells'], None
            if 'step_refill' not in cats:
                cats['step_refill'] = {'id': 'step_refill', **{k: [] for k in KEYS}, 'evidence': [],
                                       'verified': False, 'visibility_known': True, 'fallback_count': 0}
            assign('step_refill', {k: mask('step_refill', k) | cells for k in KEYS})
        # v6 splits (before the HUD rule, so the bar cells leave the new regions as they left the old)
        def derive(new, parents, masks):
            ps = [cats[p] for p in parents if p in cats]
            cats[new] = {'id': new, **{k: [] for k in KEYS},
                         'evidence': [e for p in ps for e in p.get('evidence', [])],
                         'verified': all(p.get('verified', False) for p in ps) if ps else False,
                         'visibility_known': all(p.get('visibility_known', False) for p in ps) if ps else False,
                         'fallback_count': sum(p.get('fallback_count', 0) for p in ps)}
            assign(new, masks)

        def rename(old, new):
            cats[new] = dict(cats.pop(old), id=new)

        aa, bb = _objects(before), _objects(after)
        if game == 'vc33' and 'yellow_bird_blob' in cats:
            joints, slots = _vc33_split(aa, bb, object_changes)
            if joints:
                derive('boat', ['yellow_bird_blob'], _extents(before, after, aa, bb, joints, shape))
            if slots:
                derive('waves', ['yellow_bird_blob'], _extents(before, after, aa, bb, slots, shape))
            if joints or slots:
                del cats['yellow_bird_blob']
        if game == 'sk48' and 'skewer' in cats:
            s = _sk48_split(aa, bb, object_changes)
            if s['move']:
                derive('skewer_move', ['skewer'], _extents(before, after, aa, bb, s['skewer'], shape))
            if s['extend']:
                derive('skewer_extend', ['skewer'],
                       _extents(before, after, aa, bb, s['count'] if s['move'] else s['segs'], shape))
            if s['move'] or s['extend']:
                del cats['skewer']
        if game == 'lp85' and 'tiles' in cats:
            direction = _lp85_direction(aa, bb)
            if direction:
                rename('tiles', direction)
        if game == 'su15':
            if 'plus_cursor' in cats:
                del cats['plus_cursor']
            if 'block' in cats:
                kind = _su15_split(aa, bb, object_changes)
                if kind:
                    rename('block', kind)
        if game == 'ar25' and ({'piece', 'mirror_axis'} & cats.keys()):
            for parent, new in (('piece', 'piece_with_shadow'), ('mirror_axis', 'mirror_with_shadow')):
                if parent in cats:
                    derive(new, [parent, 'reflection'],
                           {k: mask(parent, k) | (mask('reflection', k) if 'reflection' in cats else np.zeros(shape, bool))
                            for k in KEYS})
            for c in ('piece', 'mirror_axis', 'reflection'):
                cats.pop(c, None)
        bars = [c for c in cats if c.endswith('_bar') or c == 'click_budget']
        if bars:
            hud = np.logical_or.reduce([mask(c) for c in bars])
            keep_bar = {'step_refill', 'reset_to_start'} if game == 'ls20' else set()
            for c in list(cats):
                if c not in bars and '.' not in c and c not in keep_bar and (mask(c) & hud).any():
                    assign(c, {k: mask(c, k) & ~hud for k in KEYS})
        if game == 'sb26' and {'tile', 'checkbox_selection'} <= cats.keys() and (mask('tile') & diff).any():
            assign('checkbox_selection', {k: mask('checkbox_selection', k) & ~mask('tile') for k in KEYS})
        if game == 'ka59' and 'player' in cats:
            aa, bb = _objects(before), _objects(after)
            pushed = _pushed(aa, bb)
            if pushed:
                box = np.logical_or.reduce([_extent(before, aa[u], shape) | _extent(after, bb[u], shape) for u in pushed])
                cats['pushed_block'] = {'id': 'pushed_block', **{k: [] for k in KEYS}, 'evidence': [],
                                        'verified': False, 'visibility_known': cats['player']['visibility_known'],
                                        'fallback_count': 0}
                assign('pushed_block', box)
                assign('player', {k: mask('player', k) & ~box for k in KEYS})
        if game == 'ka59' and ({'edge_indicator', 'player_center_selection'} & cats.keys()):
            aa, bb = _objects(before), _objects(after)
            band, centres, toggled = np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool)
            for uid in aa.keys() & bb.keys():
                if 'player' in bb[uid].get('tags', []) and object_changes(aa[uid], bb[uid]):
                    recoloured = _recoloured(before, after, aa[uid], bb[uid], shape)
                    centre = _centre(after, bb[uid], shape)
                    band |= recoloured
                    centres |= centre
                    if (recoloured & centre).any():
                        toggled |= centre
            if 'edge_indicator' in cats:                      # the band without the centre cells (v6)
                assign('edge_indicator', band & ~centres)
            if 'player_center_selection' in cats:             # the centre cells only (v6)
                assign('player_center_selection', toggled if toggled.any() else centres)
        if game == 'dc22' and 'key_pickup' in cats:           # v6: the player standing on the key is not the key
            aa, bb = _objects(before), _objects(after)
            players = [u for u in bb if 'player' in bb[u].get('tags', [])]
            if players:
                covered = np.logical_or.reduce([_extent(after, bb[u], shape) for u in players])
                assign('key_pickup', {k: mask('key_pickup', k) & ~covered for k in KEYS})
        if game == 'r11l':
            parts = [c for c in ROBOT if c in cats]
            if parts:
                cats['robot'] = {'id': 'robot', **{k: [] for k in KEYS},
                                 'evidence': [e for c in parts for e in cats[c]['evidence']],
                                 'verified': all(cats[c]['verified'] for c in parts),
                                 'visibility_known': all(cats[c]['visibility_known'] for c in parts),
                                 'fallback_count': sum(cats[c]['fallback_count'] for c in parts)}
                assign('robot', {k: np.logical_or.reduce([mask(c, k) for c in parts]) for k in KEYS})
                for c in parts:
                    del cats[c]
        return regions

    collect_samples.migrate_events = migrate_events
    gimmick_occurrences.detect = detect_with_overlay
    collect_samples.endpoint_regions = endpoint_regions
    collect_samples._mecheval_overlay = OVERLAY_VERSION
