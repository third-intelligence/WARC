"""Endpoint-only region evaluation, schema 2.

Ground truth is ALWAYS the actual final composite frame. Geometry/semantic
provenance is separate from pixel correctness and remains review-gated.
No TTT, policy, or simulator access is performed by this module.
"""
from __future__ import annotations
import numpy as np
from .mapping import raster, indices, mask_from_indices

SCHEMA_VERSION = 2


def frame_array(frame):
    a = np.asarray(frame)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 15)):
        raise ValueError('frame must be a 2D integer palette array (0..15)')
    return a


def bounding_mask(mask):
    out = np.zeros(mask.shape, bool)
    y, x = np.nonzero(mask)
    if len(x):
        out[y.min():y.max()+1, x.min():x.max()+1] = True
    return out


def component_bounding_mask(mask):
    """Diagnostic 4-connected component bboxes, NOT inferred object IDs."""
    remaining=set(zip(*np.nonzero(mask)));out=np.zeros(mask.shape,bool)
    while remaining:
        seed=remaining.pop();component=[seed];stack=[seed]
        while stack:
            y,x=stack.pop()
            for xy in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if xy in remaining:
                    remaining.remove(xy);component.append(xy);stack.append(xy)
        ys,xs=zip(*component)
        out[min(ys):max(ys)+1,min(xs):max(xs)+1]=True
    return out


def rect_mask(rect, shape):
    """Screen-space x,y,width,height, clipped; negative extent is an error."""
    x,y,w,h = rect
    if any(int(v) != v for v in rect) or w < 0 or h < 0:
        raise ValueError('invalid screen rectangle')
    x,y,w,h = map(int,rect)
    out = np.zeros(shape,bool)
    if w and h:
        out[max(0,y):max(0,min(shape[0],y+h)),max(0,x):max(0,min(shape[1],x+w))] = True
    return out


def union(rows, key, shape):
    out=np.zeros(shape,bool)
    for row in rows:
        out |= mask_from_indices(row.get(key, []), shape)
    return out


def endpoint_regions(before, after, events, groups=(), annotations=()):
    """Build direct category and explicit joint-group regions.

    Geometry: union each object's BEFORE bbox and AFTER bbox separately. Never
    bbox an entire group (would include unrelated space between distant objects).
    Events may bind a part; context bbox still comes from the parent object.
    Annotations must already contain screen masks and explicit provenance.
    Merely co-occurring events are NOT automatically labeled causally coupled.
    """
    a,b=frame_array(before['frame']),frame_array(after['frame'])
    if a.shape!=b.shape: raise ValueError('frame shape changed')
    if before.get('level')!=after.get('level'): raise ValueError('level boundary')
    ra,rb=raster(before),raster(after)
    geometry={}
    empty=np.zeros(a.shape,bool)
    for uid in ra['objects'].keys() | rb['objects'].keys():
        x,y=ra['objects'].get(uid),rb['objects'].get(uid)
        sa=x['amodal'] if x else empty; sb=y['amodal'] if y else empty
        va=x['visible'] if x else empty; vb=y['visible'] if y else empty
        geometry[uid]={'bbox':indices(bounding_mask(sa)|bounding_mask(sb)),
                       'support':indices(sa|sb),'visible':indices(va|vb),
                       'visibility_status':'generic_renderer_candidate'}
    bound=[];unresolved=[]
    for ev in events:
        cat=ev.get('gimmick_id')
        if not cat:
            unresolved.append({'id':ev['id'],'reason':ev.get('category_status','no_category')});continue
        candidates=ev.get('candidates',[])
        uid=ev.get('parent') or (candidates[0] if len(candidates)==1 else None)
        status=ev.get('binding_status')
        if uid in geometry and status in {'candidate_whole_object','candidate_part'}:
            g=geometry[uid]
            local=ev['roi'] if status=='candidate_part' else g['support']
            bound.append({'id':ev['id'],'category':cat,'instances':[uid],**g,
                          'local':local,'source':ev.get('source_rule'),
                          'verified':bool(ev.get('mapping_verified')), 'mask_status':status,
                          'released_gimmick_id':ev.get('released_gimmick_id'),
                          'component':ev.get('component','whole_object'),
                          'operation_id':ev.get('operation_id'), 'bbox_mode':'parent_before_after'})
        else:
            # A text bbox is not enough to promise exact parent/block coverage.
            # Preserve it explicitly as a diagnostic fallback, never verified.
            if ev.get('roi'):
                bound.append({'id':ev['id'],'category':cat,'instances':[],
                              'bbox':ev['roi'],'support':ev['roi'],'local':ev['roi'],
                              'visible':[], 'visibility_status':'not_yet_resolved', 'source':ev.get('source_rule'),
                              'verified':False,'mask_status':status,
                              'bbox_mode':'local_fallback_not_parent'})
            unresolved.append({'id':ev['id'],'category':cat,'reason':status})
    for row in annotations:
        if not row.get('source') or not row.get('category'):
            raise ValueError('annotation requires category and provenance')
        for key in ('bbox','support','local','visible'):
            mask_from_indices(row.get(key,[]),a.shape)
        bound.append(dict(row))
    categories={}
    for cat in sorted({r['category'] for r in bound}):
        entries=[r for r in bound if r['category']==cat]
        masks={k:indices(union(entries,k,a.shape)) for k in ('bbox','support','local','visible')}
        categories[cat]={'id':cat,**masks,'evidence':entries,
                         'verified':all(r.get('verified',False) for r in entries),
                         'visibility_known':all(r.get('visibility_status') in {'generic_renderer_candidate','verified'} for r in entries),
                         'fallback_count':sum(r.get('bbox_mode')=='local_fallback_not_parent' for r in entries)}
    joint={}
    for group in groups:
        if group['id'] in joint: raise ValueError('duplicate group id')
        if not group.get('source'): raise ValueError('joint group needs provenance')
        missing=set(group['categories'])-categories.keys()
        if missing: raise ValueError('unknown group categories: '+str(missing))
        selected=[categories[c] for c in set(group['categories'])]
        joint[group['id']]={'id':group['id'], 'categories':sorted(set(group['categories'])),
            'source':group['source'], 'relation':group.get('relation','explicit'),
            **{k:indices(union(selected,k,a.shape)) for k in ('bbox','support','local','visible')},
            'visibility_known':all(r['visibility_known'] for r in selected),
            'verified':bool(group.get('verified')) and all(r['verified'] for r in selected)}
    # Keep per-instance and released-component views after category merges.
    # Never sum these rows to get the category denominator: use its mask union.
    instances={};components={}
    for cat,category in categories.items():
        for field,destination in (('instances',instances),('component',components)):
            keys={k for row in category['evidence'] for k in
                  (row.get(field,[]) if field=='instances' else [row.get(field,'whole_object')])}
            for key in sorted(keys):
                if field=='instances':
                    entries=[r for r in category['evidence'] if key in r.get(field,[])]
                else:
                    entries=[r for r in category['evidence'] if r.get(field,'whole_object')==key]
                masks={k:indices(union(entries,k,a.shape)) for k in ('bbox','support','local','visible')}
                destination[cat+'/'+key]={'id':cat+'/'+key,'category':cat,**masks,
                    'evidence':entries,'verified':all(r.get('verified',False) for r in entries),
                    'visibility_known':category['visibility_known'],
                    'fallback_count':sum(r.get('bbox_mode')=='local_fallback_not_parent' for r in entries)}
    return {'schema_version':SCHEMA_VERSION,'before':a.tolist(),'after':b.tolist(),
            'categories':categories,'groups':joint,'unresolved':unresolved,
            'instances':instances,'components':components,
            'camera_changed':before.get('camera')!=after.get('camera'),
            'status':'diagnostic_until_semantic_mask_and_replay_verified'}


def mask_metrics(pred, before, after, mask):
    correct=pred==after; changed=before!=after
    def accuracy(m):
        n=int(m.sum());k=int((correct&m).sum())
        return {'correct':k,'pixels':n,'accuracy':k/n if n else None}
    keep=mask&~changed
    errors=int((keep&~correct).sum());n=int(keep.sum())
    return {'all':accuracy(mask),'changed':accuracy(mask&changed),
            'unchanged':accuracy(keep),'false_edits':errors,
            'unchanged_pixels':n,'false_edit_rate':errors/n if n else None}


def evaluate(prediction, regions, include_candidates=False, invalid_action=None):
    a,b,p=frame_array(regions['before']),frame_array(regions['after']),frame_array(prediction)
    if p.shape!=b.shape or a.shape!=b.shape: raise ValueError('prediction shape mismatch')
    selected={k:v for k,v in regions['categories'].items() if include_candidates or v['verified']}
    bbox_counts=np.zeros(a.shape,int);support_counts=np.zeros(a.shape,int)
    for row in selected.values():
        bbox_counts+=mask_from_indices(row['bbox'],a.shape)
        support_counts+=mask_from_indices(row['support'],a.shape)
    def score_row(row):
        m=mask_from_indices(row['bbox'],a.shape);s=mask_from_indices(row['support'],a.shape)
        visible=mask_from_indices(row['visible'],a.shape)
        return {'bbox':mask_metrics(p,a,b,m),'local':mask_metrics(p,a,b,mask_from_indices(row['local'],a.shape)),
                'support':mask_metrics(p,a,b,s),'visible':mask_metrics(p,a,b,visible),
                'bbox_shared':mask_metrics(p,a,b,m&(bbox_counts>1)),
                'bbox_exclusive':mask_metrics(p,a,b,m&(bbox_counts==1)),
                'support_shared_pixels':int((s&(support_counts>1)).sum()),
                'occluded_support_pixels':int((s&~visible).sum()) if row.get('visibility_known') else None,
                'visibility_known':row.get('visibility_known',False),
                'verified':row['verified'],'fallback_count':row.get('fallback_count',0)}
    out={'schema_version':SCHEMA_VERSION,'diagnostic':include_candidates,
         'categories':{k:score_row(v) for k,v in selected.items()},
         'groups':{k:score_row(v) for k,v in regions['groups'].items() if include_candidates or v['verified']},
         'instances':{k:score_row(v) for k,v in regions.get('instances',{}).items() if include_candidates or v['verified']},
         'components':{k:score_row(v) for k,v in regions.get('components',{}).items() if include_candidates or v['verified']},
         'union':mask_metrics(p,a,b,bbox_counts>0),
         'outside':mask_metrics(p,a,b,bbox_counts==0),
         'bound_categories':len(selected),'unresolved_count':len(regions['unresolved']),
         'outside_is_whole_frame':not bool(selected),
         'camera_changed':regions['camera_changed']}
    # Invalid-click evidence must come from an adapter/human, not unchanged
    # pixels. A valid blocked action can also look unchanged.
    invalid_confirmed=bool(invalid_action and invalid_action.get('confirmed') and invalid_action.get('source'))
    identity=bool(np.array_equal(a,b))
    out['identity_diagnostic']={'invalid_action_confirmed':invalid_confirmed,
        'observed_full_frame_identity':identity,'primary_inclusion':False,
        'identity_accuracy':float(np.mean(p==a)) if identity else None,
        'actual_unchanged_region':mask_metrics(p,a,b,a==b),
        'reason':'actual_identity' if identity else 'actual_transition_not_identity_do_not_override_gt'}
    return out
