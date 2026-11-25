"""Source-grounded candidate regions for all 20 games.

Unimplemented or ambiguous bindings stay unresolved. No category is inferred
from screen colour alone.
"""
import numpy as np
from .endpoint_metrics import rect_mask,bounding_mask,component_bounding_mask
from .mapping import indices,raster

# Exact names from get_nonsprite_status, checked against render code. Explicit
# corrections for tr87 (metadata incorrectly says top row), re86 (no coords),
# and m0r0 (metadata omits the second bar) are recorded below.
HUD={
 'ar25':{'step_bar':('steps',(63,0,1,64))},
 'cd82':{'energy_bar':('step_budget',(0,63,64,1))},
 'dc22':{'budget_bar':('step_counter',(0,63,64,1))},
 'ft09':{'click_budget':('click_budget',(0,63,64,1))},
 'ka59':{'budget_bar':('step_timer',(0,63,64,1))},
 'lp85':{'life_bar':('step_bar',(0,0,1,64)),'level_pips':('level_progress',None)},
 'ls20':{'steps_bar':('steps',None)},
 'm0r0':{'life_bar':('steps_remaining',(0,0,64,1)),
          'mirror_life_bar':('steps_remaining',(0,63,64,1))},
 'r11l':{'budget_bar':('step_counter',(0,0,1,64))},
 're86':{'budget_bar':('step_counter',(0,63,64,1))},
 's5i5':{'budget_bar':('step_counter',(0,63,64,1))},
 'sb26':{'budget_bar':('energy_bar',(0,53,64,1))},
 'sc25':{'budget_bar':('action_budget',None)},
 'sk48':{'budget_bar':('steps_remaining',(0,53,64,1))},
 'su15':{'budget_bar':('step_counter',(0,63,64,1))},
 'tn36':{'budget_bar':('progress_bar',None)},
 'tr87':{'bottom_timer_bar':('moves_budget',(0,63,64,1))},
 'tu93':{'bottom_energy_bar':('step_counter',(0,63,64,1))},
 'vc33':{'top_timer_bar':('click_budget',(0,0,64,1))},
 'wa30':{'bottom_timer_bar':('step_counter',(0,63,64,1))},
}


def annotation(cat,bbox,support,local,visible,source,instances=(),**extra):
    return {'id':source+':'+cat,'category':cat,'bbox':indices(bbox),
            'support':indices(support),'local':indices(local),'visible':indices(visible),
            'source':source,'instances':list(instances),'verified':False,
            'semantic_status':'candidate','mask_status':'source_geometry_candidate',
            'bbox_mode':'parent_before_after','visibility_status':'not_yet_resolved',**extra}


def project_cells(snapshot,cells):
    """World-space integer (x,y) cells -> actual camera screen projection."""
    shape=np.asarray(snapshot['frame']).shape;c=snapshot['camera'];out=np.zeros(shape,bool)
    scale=c['scale']
    viewport=rect_mask((c['offset_x'],c['offset_y'],c['width']*scale,c['height']*scale),shape)
    for x,y in cells:
        out |= rect_mask(((x-c['x'])*scale+c['offset_x'],(y-c['y'])*scale+c['offset_y'],scale,scale),shape)
    return out&viewport


def annotations(before,after):
    game=before['game'];shape=np.asarray(before['frame']).shape
    result=[];notes=[]
    for cat,(name,override) in HUD[game].items():
        masks=[]
        for snap in (before,after):
            row=next((r for r in snap.get('nonsprite',[]) if r['name']==name),None)
            if row is None:continue
            rect=override or tuple(row.get(k,0) for k in ('x','y','width','height'))
            masks.append(rect_mask(rect,shape))
        if masks:
            mask=np.logical_or.reduce(masks)
            result.append(annotation(cat,mask,mask,mask,mask,
                f'{game}:HUD:{name}',bbox_mode='hud_full_band',
                visibility_status='renderer_candidate_not_occlusion_verified'))
    if game=='ar25':
        # Use original reflection traversal instrumented at its actual branch,
        # not the separate depth-2 text-bbox approximation.
        rows=[s.get('special',{}).get('reflections',[]) for s in (before,after)]
        ids={r['source_instance'] for rs in rows for r in rs}
        for uid in sorted(ids,key=str):
            masks=[]
            for snap,rs in zip((before,after),rows):
                cells=[xy for r in rs if r['source_instance']==uid for xy in r['cells']]
                masks.append(project_cells(snap,cells))
            support=masks[0]|masks[1]
            if support.any():
                result.append(annotation('reflection',component_bounding_mask(masks[0])|component_bounding_mask(masks[1]),
                    support,support,np.zeros(shape,bool),'ar25:_compute_all_reflection_maps',
                    instances=[uid] if uid else [],visibility_status='not_yet_resolved',
                    bbox_mode='connected_component_candidate_not_full_reflected_object'))
        # Keep each real piece as a direct object even if reflection logs were
        # coalesced at an overlap. Do not equate selected/highlight with motion.
        ra,rb=raster(before),raster(after)
        for uid in ra['objects'].keys()|rb['objects'].keys():
            aa=ra['objects'].get(uid);bb=rb['objects'].get(uid);o=(bb or aa)['object']
            if 'piece' not in o.get('tags',[]):continue
            empty=np.zeros(shape,bool);sa=aa['amodal'] if aa else empty;sb=bb['amodal'] if bb else empty
            visible=(aa['visible'] if aa else empty)|(bb['visible'] if bb else empty)
            result.append(annotation('piece',bounding_mask(sa)|bounding_mask(sb),sa|sb,sa|sb,visible,
                'ar25:piece_tag',instances=[uid],visibility_status='generic_renderer_candidate'))
    if game=='cd82':
        # Raw writes are kept as provenance. review_mapping now binds actual
        # canvas writes to stack_fill with parent bbox + painted local mask.
        for row in after.get('special',{}).get('paint',[]):
            notes.append({'kind':'paint_write_provenance_mask_review_pending',**row})
    return result,notes


def joint_groups(game,categories):
    groups=[]
    if game=='ar25' and {'piece','reflection'}<=set(categories):
        groups.append({'id':'piece_and_reflection','categories':['piece','reflection'],
            'source':'ar25:_compute_all_reflection_maps piece-to-reflection relation',
            'relation':'source_coupled','verified':False})
    # Co-occurrence is a useful combined diagnostic, but never a claim of causal
    # coupling. Source-coupled groups will be added game by game after review.
    board=sorted(set(categories)-set(HUD[game]))
    if len(board)>1:
        groups.append({'id':'joint_observed_board','categories':board,
            'source':'categories observed in same external action; not causal attribution',
            'relation':'co_occurrence_only','verified':False})
    return groups
