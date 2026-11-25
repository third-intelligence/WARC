"""Geometry-first ROI mapping. Input pixels are ALREADY rotated/scaled.

No game-specific direction branches, no inference of causality from pixel diff.
Snapshot objects use stable instance_id (not sprite name). All masks are flat
screen pixel indices in JSON; arrays are [y,x]. Evaluation-only information.
"""
from __future__ import annotations
import numpy as np


def indices(mask):
    return np.flatnonzero(mask).astype(int).tolist()


def mask_from_indices(values, shape):
    a = np.zeros(shape[0] * shape[1], dtype=bool)
    v = np.asarray(values, dtype=int)
    if np.any(v < 0) or np.any(v >= a.size):
        raise ValueError('mask index outside frame')
    a[v] = True
    return a.reshape(shape)


def raster(snapshot):
    """Compose sprite ownership, preserving amodal and part masks.

    Camera projection is supplied explicitly. HUD/custom drawing with no sprite
    attribution is detected by comparing composition with the observed frame.
    Those pixels are kept as unattributed, not assigned to a guessed sprite.
    """
    frame = np.asarray(snapshot['frame'], dtype=int)
    if frame.ndim != 2: raise ValueError('frame must be HxW')
    h, w = frame.shape
    c = snapshot['camera']
    if c['scale'] < 1 or int(c['scale']) != c['scale']:
        raise ValueError('only positive integer camera scale supported')
    scale = int(c['scale'])
    owner = np.full((h, w), '', dtype=object)
    composed = np.full((h, w), c.get('letter_box', 0), dtype=int)
    vx, vy = int(c['offset_x']), int(c['offset_y'])
    vw, vh = int(c['width'] * scale), int(c['height'] * scale)
    viewport = np.zeros((h, w), bool)
    viewport[max(0, vy):min(h, vy+vh), max(0, vx):min(w, vx+vw)] = True
    composed[viewport] = c.get('background', 0)
    objects = {}
    for obj in sorted(snapshot['objects'], key=lambda o: o.get('layer', 0)):
        uid = str(obj['instance_id'])
        if uid in objects: raise ValueError('duplicate instance_id: ' + uid)
        pixels = np.asarray(obj['pixels'], dtype=int)
        if pixels.size==0:pixels=np.empty((0,0),dtype=int)
        if pixels.ndim != 2: raise ValueError('sprite pixels must be 2D')
        parts = {k: np.asarray(v, dtype=bool) for k, v in obj.get('parts', {}).items()}
        if any(p.shape != pixels.shape for p in parts.values()):
            raise ValueError('part mask must use rendered local coordinates')
        amodal = np.zeros((h, w), bool)
        part_masks = {k: np.zeros((h, w), bool) for k in parts}
        x = int((obj['x']-c['x'])*scale + vx)
        y = int((obj['y']-c['y'])*scale + vy)
        scaled = pixels.repeat(scale, 0).repeat(scale, 1)
        ph, pw = scaled.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x+pw), min(h, y+ph)
        if obj.get('visible', True) and x1 > x0 and y1 > y0:
            sub = scaled[y0-y:y1-y, x0-x:x1-x]
            m = (sub >= 0) & viewport[y0:y1, x0:x1]
            amodal[y0:y1, x0:x1] = m
            composed[y0:y1, x0:x1][m] = sub[m]
            owner[y0:y1, x0:x1][m] = uid
            for key, part in parts.items():
                pm = part.repeat(scale, 0).repeat(scale, 1)[y0-y:y1-y, x0-x:x1-x]
                part_masks[key][y0:y1, x0:x1] = pm & m
        objects[uid] = {'object': obj, 'amodal': amodal, 'parts': part_masks}
    mismatch = composed != frame
    for uid, data in objects.items():
        data['visible'] = (owner == uid) & ~mismatch
        data['visible_parts'] = {k: v & data['visible'] for k,v in data['parts'].items()}
    return {'objects': objects, 'unattributed': mismatch, 'composed': composed}


def shifted(mask, dx, dy):
    """Screen displacement without wrap-around; includes any cardinal direction."""
    out = np.zeros_like(mask)
    y,x = np.nonzero(mask)
    nx,ny = x+int(dx),y+int(dy)
    ok = (nx>=0)&(ny>=0)&(nx<mask.shape[1])&(ny<mask.shape[0])
    out[ny[ok],nx[ok]] = True
    return out


def map_transition(before, after, groups=None, probes=None, intermediate=None):
    """Map object/part/group ROI including stationary objects.

    groups: {id, members:[instance ids], gimmick_id, verified:bool}.
    probes: {id, actor, obstacle, delta:[screen dx,dy], confirmed:bool,
             evidence:string}; requires an explicit user/adapter annotation.
    Group membership/blocked causes are never guessed from co-occurrence.
    """
    a,b = np.asarray(before['frame']),np.asarray(after['frame'])
    if a.shape != b.shape: raise ValueError('frame shape changed')
    if before.get('level') != after.get('level'):
        raise ValueError('level boundary must be reviewed separately')
    ra,rb = raster(before),raster(after)
    middle=[]
    for snapshot in intermediate or []:
        if snapshot.get('level') != before.get('level'):
            raise ValueError('intermediate frame crossed a level boundary')
        middle.append(raster(snapshot))
    delta = a != b
    empty = np.zeros(a.shape, bool)
    mapped = []
    uid_masks = {}
    all_ids=ra['objects'].keys() | rb['objects'].keys()
    for mid in middle: all_ids |= mid['objects'].keys()
    for uid in sorted(all_ids):
        oa,ob = ra['objects'].get(uid),rb['objects'].get(uid)
        ma = oa['visible'] if oa else empty
        mb = ob['visible'] if ob else empty
        aa = oa['amodal'] if oa else empty
        ab = ob['amodal'] if ob else empty
        roi = ma|mb
        uid_masks[uid] = roi
        temporal=roi.copy()
        middle_obj=None
        for mid in middle:
            if uid in mid['objects']:
                middle_obj=mid['objects'][uid]
                temporal |= middle_obj['visible']
        obj = (ob or oa or middle_obj)['object']
        if oa is None and ob is None: change = 'intermediate_only'
        elif oa is None: change = 'appear'
        elif ob is None: change = 'disappear'
        else:
            x,y=oa['object'],ob['object']
            change = 'move' if (x['x'],x['y']) != (y['x'],y['y']) else 'transform_or_recolor' if not np.array_equal(x['pixels'],y['pixels']) else 'stationary'
        mapped.append({'id': uid, 'kind': 'object', 'name': obj.get('name',uid),
                       'tags': obj.get('tags',[]), 'change': change,
                       'gimmick_id': obj.get('gimmick_id'),
                       'mapping_verified': obj.get('mapping_verified',False),
                       'before': indices(ma), 'after': indices(mb),
                       'roi': indices(roi), 'amodal_roi': indices(aa|ab),
                       'trajectory_roi':indices(temporal),'intermediate_only':indices(temporal&~roi),
                       'changed': indices(roi & delta),
                       'unattributed_overlap': indices((aa|ab)&(ra['unattributed']|rb['unattributed']))})
        keys = set((oa or {}).get('parts',{})) | set((ob or {}).get('parts',{}))
        for key in sorted(keys):
            pa=(oa or {}).get('visible_parts',{}).get(key,empty)
            pb=(ob or {}).get('visible_parts',{}).get(key,empty)
            mapped.append({'id': uid+'/'+key, 'parent': uid, 'name':key, 'kind':'part',
                           'before':indices(pa),'after':indices(pb),'roi':indices(pa|pb),
                           'changed':indices((pa|pb)&delta), 'mapping_verified':False})
    for group in groups or []:
        missing = set(group['members']) - uid_masks.keys()
        if missing: raise ValueError('unknown group members: '+str(missing))
        roi=np.zeros(a.shape,bool)
        for uid in group['members']: roi |= uid_masks[uid]
        mapped.append({'id':group['id'],'kind':'group','name':group['id'],
                       'gimmick_id':group.get('gimmick_id'),'members':group['members'],
                       'mapping_verified':group.get('verified',False),
                       'roi':indices(roi),'changed':indices(roi&delta)})
    for probe in probes or []:
        actor=probe['actor']; obstacle=probe['obstacle']
        if actor not in ra['objects'] or obstacle not in uid_masks:
            raise ValueError('probe needs existing actor and obstacle')
        if probe.get('confirmed') and not probe.get('evidence'):
            raise ValueError('confirmed contact needs evidence')
        dx,dy=probe['delta']
        start=ra['objects'][actor]['visible']
        attempted=shifted(ra['objects'][actor]['amodal'],dx,dy)
        roi=start | uid_masks[actor] | uid_masks[obstacle] | attempted
        mapped.append({'id':probe['id'],'kind':'interaction','name':probe['id'],
                       'roi':indices(roi),'attempted':indices(attempted),
                       'changed':indices(roi&delta),'evidence':probe.get('evidence'),
                       'mapping_verified':probe.get('confirmed',False)})
    covered=np.zeros(a.shape,bool)
    for x in mapped:
        if x['kind']=='object' and x['change']!='stationary':
            covered |= mask_from_indices(x['roi'],a.shape)
    overlaps=np.zeros(a.shape,int)
    for mask in uid_masks.values(): overlaps += mask
    return {'before':a.tolist(),'after':b.tolist(),'level':before.get('level'),
            'objects':mapped,'changed':indices(delta),'unexplained_changed':indices(delta&~covered),
            'overlap':indices(overlaps>1),
            'unattributed_before':indices(ra['unattributed']),
            'unattributed_after':indices(rb['unattributed']),
            'status':'candidate_geometry_not_human_reviewed'}


def score(prediction, mapped, object_id):
    """Empty denominators -> None; raw numerators retained for aggregation."""
    pred=np.asarray(prediction); a=np.asarray(mapped['before']); b=np.asarray(mapped['after'])
    if pred.shape != b.shape: raise ValueError('prediction shape differs')
    obj=next(o for o in mapped['objects'] if o['id']==object_id)
    roi=mask_from_indices(obj['roi'],b.shape)
    changed=roi & (a!=b); keep=roi & ~changed; correct=pred==b
    def ratio(mask):
        n=int(mask.sum()); k=int((correct&mask).sum())
        return {'correct':k,'pixels':n,'accuracy':k/n if n else None}
    return {'roi':ratio(roi),'changed':ratio(changed),'unchanged':ratio(keep),
            'false_edit_rate':float((~correct&keep).sum()/keep.sum()) if keep.any() else None}


def category_scores(prediction,mapped,include_candidates=False):
    """Union event masks per category; never sum parent+parts as independent pixels.

    Default excludes unreviewed bindings. Diagnostic preview must explicitly opt
    into candidate ROIs. Object/part-level scores remain available via score().
    """
    by_category={}
    for obj in mapped['objects']:
        cat=obj.get('gimmick_id')
        if obj['kind'] not in {'event','group'} or not cat:continue
        if not obj.get('mapping_verified') and not include_candidates:continue
        by_category.setdefault(cat,set()).update(obj['roi'])
    result={}
    for cat,roi in by_category.items():
        tmp=dict(mapped,objects=[{'id':'category','roi':sorted(roi)}])
        result[cat]=score(prediction,tmp,'category')
    return result
