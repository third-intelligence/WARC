"""Versioned review overlay. Released catalog/answers and raw events stay intact.

All masks are candidates, NOT human-certified GT. Use real endpoint geometry;
do not turn an animation event bbox into an endpoint scoring region.
"""
import re
import numpy as np
from .mapping import raster, indices
from .endpoint_metrics import bounding_mask,rect_mask
from .game_regions import annotation

VERSION=3
ALIASES={
    're86':{'marker_visibility':'piece'},
    's5i5':{'chained_arm':'extending_arm','moving_dot':'extending_arm','fixed_arm':'obstacle'},
    'sc25':{'teleport_indicator':'player_movement'},
}
# These decisions are L1/L2 only. Unexpected observed evidence is retained in
# the audit rather than interpreted as proof that the category never exists.
LEVEL_EXCLUDED={
    'ar25':{1:{'selection'}},
    'sc25':{1:{'door','energy_pack','fireball','teleport_indicator'},2:{'door','energy_pack','fireball'}},
    'r11l':{1:{'paint'},2:{'paint'}},
    'su15':{1:{'moving_ufo'},2:{'moving_ufo'}},
    'tn36':{1:{'barrier','ghost_overlay'},2:{'barrier','ghost_overlay'}},
}
NEW_DESCRIPTIONS={
    'cd82':{'container_tilt':'the basket moves, turns or changes shape',
            'reservoir_recolor':'the selected colour inside the basket or the crane changes',
            'selection_marker':'the underline that marks the selected colour moves',
            'stack_fill':'paint is applied to the canvas'},
    'ls20':{'step_refill':'the refill item is taken and disappears; the refilled step bar is counted as well'},
    'r11l':{'arm_selection':'the colours of the selected and the deselected arm change'},
    'sc25':{'player_movement':'the player moves; a shift caused by resizing belongs to player_resize',
            'player_resize':'a spell enlarges or shrinks the player; a blocked resize is scored on its unchanged region'},
    's5i5':{'obstacle':'a fixed obstacle; a collision is scored near the contact'},
    'dc22':{'terrain_object':'an object that moves, rotates or changes pattern and is none of the known bridges, floors or coloured tiles'},
    'wa30':{'target_facing':'the grey frame of the object in front of the agent appears or disappears',
            'target_grab':'the frame of a grabbed object appears or disappears (white for the selector, black for other agents)'},
}


def canonical_descriptions(game,level):
    from .catalog import descriptions
    from .endpoint_scope import scope
    result={}
    replaced={'dc22':{'ghost','course_gimmick'},'sc25':{'player'},
              'wa30':{'yellow_button_toggle'},'su15':{'vacuum'}}
    for old,desc in descriptions()[game].items():
        if old in LEVEL_EXCLUDED.get(game,{}).get(level,set()):continue
        if old in replaced.get(game,set()):continue
        if scope(game,old)['status']!='endpoint_candidate':continue
        cat=ALIASES.get(game,{}).get(old,old)
        if cat not in result or cat==old:result[cat]=desc
    result.update(NEW_DESCRIPTIONS.get(game,{}))
    if game=='ls20' and level==1:result.pop('step_refill',None)
    return result


def endpoint_objects(before,after):
    return {o['instance_id']:o for s in (before,after) for o in s['objects']}


def migrate_events(game,events,before,after):
    """Return canonical candidates and withheld evidence, without mutating input."""
    objects=endpoint_objects(before,after);kept=[];withheld=[]
    for original in events:
        row=dict(original);old=row.get('gimmick_id');reason=None
        row['released_gimmick_id']=old;row['review_mapping_version']=VERSION
        if game=='cd82' and old is None:
            text=row.get('event','');m=re.search(r'sname\(([^)]+)\)',text)
            name=m[1] if m else None
            cat={'ActiveBasket':'container_tilt','crane_arm':'container_tilt',
                 'canvas':'stack_fill','color_indicator':'selection_marker','step_budget':'energy_bar'}.get(name)
            if cat:
                row['gimmick_id']=cat;row['name']=cat;row['component']=name
                kept.append(row);continue
        if old in LEVEL_EXCLUDED.get(game,{}).get(before['level'],set()):
            reason='excluded_L1_L2_by_review; observed raw event retained for contradiction check'
        elif game=='su15' and old=='vacuum':
            reason='vacuum animation excluded; actual block endpoint retained separately'
        elif game=='wa30' and old=='yellow_button_toggle':
            reason='replaced_by_source_target_border_annotations'
        elif game=='sc25' and old=='player':
            reason='replaced_by_source_player_movement_resize_annotations'
        elif game=='r11l' and old=='arm' and row.get('event','').startswith('[bright]:'):
            reason='arm selection uses before/after selected_arm identities instead'
        elif game=='sb26' and old in {'reference_cursor','bracket','outline_frame','target_box','background_flash'}:
            reason='animation-prone category uses endpoint-only object annotations instead'
        if reason:
            withheld.append({**row,'review_disposition':reason});continue
        cat=ALIASES.get(game,{}).get(old,old)
        if game=='dc22' and old in {'course_gimmick','ghost'}:
            candidates=[objects[x] for x in row.get('candidates',[]) if x in objects]
            if not candidates:
                # A coalesced change can replace an instance/name at the same
                # location. A text box alone cannot certify the new identity.
                withheld.append({**row,'review_disposition':'pending_dc22_endpoint_identity'});continue
            categories=set()
            for o in candidates:
                ts=set(o.get('tags',[]))
                categories.add('bridge' if 'movable_bridge' in ts else
                    'movable_floor' if 'movable_floor' in ts else
                    'color_tile_visibility' if 'itki' in ts else 'terrain_object')
            if len(categories)!=1:
                withheld.append({**row,'review_disposition':'pending_dc22_mixed_object_types'});continue
            cat=categories.pop()
        row['gimmick_id']=cat;row['name']=cat or row['name']
        if cat!=old:row['component']=old
        kept.append(row)
    return kept,withheld


def _object_region(cat,uid,ra,rb,source,part=None,**extra):
    x,y=ra['objects'].get(uid),rb['objects'].get(uid)
    shape=ra['composed'].shape
    empty=np.zeros(shape,bool)
    sa=x['amodal'] if x else empty;sb=y['amodal'] if y else empty
    local=sa|sb
    if part:
        # raster stores projected part masks alongside object geometry.
        local=(x['parts'].get(part,empty) if x else empty)|(y['parts'].get(part,empty) if y else empty)
    changed=(x is None or y is None or any(x['object'].get(k)!=y['object'].get(k)
             for k in ('x','y','pixels','visible','rotation','scale')))
    return annotation(cat,bounding_mask(sa)|bounding_mask(sb),sa|sb,local,
        (x['visible'] if x else empty)|(y['visible'] if y else empty),source,
        instances=[uid],component=part or 'whole_object',
        activation='endpoint_object_changed' if changed else 'persistent_endpoint_state',
        visibility_status='generic_renderer_candidate',**extra)


def intrinsic_changed(a,b,fields=('x','y','rotation','scale','pixels','visible')):
    """Same endpoint identity/state criterion as the occurrence detector."""
    return a is None or b is None or any(a.get(k)!=b.get(k) for k in fields)


def reviewed_annotations(before,after,interactions=None,action_id=None):
    game=before['game'];ra,rb=raster(before),raster(after);out=[];notes=[]
    aa={o['instance_id']:o for o in before['objects']};bb={o['instance_id']:o for o in after['objects']}
    events=(interactions or {}).get('events',[])
    for uid in sorted(aa.keys()|bb.keys()):
        a,b=aa.get(uid),bb.get(uid);obj=b or a;tags=set(obj.get('tags',[]))
        ar=(a or {}).get('evaluation_roles',{});br=(b or {}).get('evaluation_roles',{})
        def add(cat,source,**kw):out.append(_object_region(cat,uid,ra,rb,source,**kw))
        if game=='cd82':
            if obj['name'] in {'ActiveBasket','crane_arm'}:
                add('container_tilt','cd82:_refresh_active_bucket_or_crane')
                # The source changes these interior pixels using selected_color.
                colors=[next((r.get('value') for r in s.get('nonsprite',[]) if r['name']=='selected_color'),None)
                        for s in (before,after)]
                if None not in colors and colors[0]!=colors[1]:
                    add('reservoir_recolor','cd82:selected_color:interior',part='interior')
            elif obj['name']=='color_indicator':
                add('selection_marker','cd82:color_indicator')
            elif obj['name']=='canvas' and b and 'painted' in b.get('parts',{}):
                painted=rb['objects'][uid]['parts']['painted']
                if painted.any():
                    region=_object_region('stack_fill',uid,ra,rb,'cd82:actual_paint_writes')
                    region.update(local=indices(painted),component='painted',operation_id='paint')
                    out.append(region)
        if game=='ls20' and 'step_refill' in tags and a and not b:
            add('step_refill','ls20:_process_tile_effects:step_refill',operation_id='step_refill')
        if game=='r11l' and (ar.get('arm') or br.get('arm')):
            # Duplicate sprite names are not identities. Selection color alone
            # is NOT physical arm motion; keep it on the separate category.
            if intrinsic_changed(a,b,('x','y','rotation','scale','visible')):
                add('arm','r11l:all_legs:physical_endpoint_identity',operation_id='arm_physical_change')
            if ar.get('selected_arm')!=br.get('selected_arm'):
                add('arm_selection','r11l:select_leg:old_and_new_selection',part='selection_color',operation_id='arm_selection')
        if game=='sc25' and (ar.get('spell_slot') or br.get('spell_slot')) and intrinsic_changed(a,b):
            add('spell_slot_grid','sc25:clickable_spell_slots:endpoint_identity',operation_id='spell_slot_change')
        if game=='sc25' and (ar.get('player') or br.get('player')):
            resizing=bool(a and b and a['scale']!=b['scale']) or any(e['reason']=='resize_cast' for e in events)
            if resizing:add('player_resize','sc25:_cast_resize',operation_id='player_resize')
            elif (a and b and (a['x'],a['y'])!=(b['x'],b['y'])) or action_id in (1,2,3,4):
                add('player_movement','sc25:player_identity:movement_or_direction_attempt')
        if game=='wa30' and 'target' in tags and 'border' in obj.get('parts',{}):
            states={ar.get('target_border_state'),br.get('target_border_state')}
            if 'facing' in states:
                add('target_facing','wa30:_update_target_colors:facing',part='border',states=sorted(s for s in states if s))
            if states&{'selector_grab','other_grab'}:
                add('target_grab','wa30:_update_target_colors:grab',part='border',states=sorted(s for s in states if s))
        if game=='sb26':
            direct={'cursor_box':'checkbox_selection','color_tile':'tile'}.get(obj['name'])
            if direct and intrinsic_changed(a,b):
                add(direct,'sb26:endpoint_identity:'+obj['name'],operation_id='endpoint_object_change')
                if a and b and not a.get('visible',True) and not b.get('visible',True):
                    notes.append({'kind':'roi_unavailable','category':direct,'instance_id':uid,
                        'reason':'invisible_at_both_endpoints','source':'sb26:endpoint_identity:'+obj['name'],
                        'endpoint_visible':[False,False],'endpoint_support_pixels':0,
                        'scoring_policy':'no_endpoint_pixels; no_intermediate_bbox_substitution'})
            names={'selected_highlight':'reference_cursor','bracket_box':'bracket',
                   'frame_cursor':'outline_frame','target_box':'target_box','bg_tile':'background_flash'}
            cat=names.get(obj['name'])
            if cat:
                # Clipped endpoint support, never intermediate bbox/path. A
                # persistent endpoint object remains a static false-edit test.
                region=_object_region(cat,uid,ra,rb,'sb26:endpoint_object:'+obj['name'])
                if region['support']:out.append(region)
    if game=='s5i5':
        # Only the pair returned by the actual short-circuit overlap query.
        # Keep fixed-obstacle ROI local to the attempted actor footprint.
        shape=np.asarray(before['frame']).shape
        for event in events:
            if event['reason']!='arm_overlap_rollback':continue
            members=event['actors']+event['counterparts'];cam=event['camera']
            for member in members:
                uid=member.get('instance_id');o=bb.get(uid) or aa.get(uid)
                if not o or o.get('evaluation_roles',{}).get('arm_role')!='fixed_arm':continue
                context=np.zeros(shape,bool)
                for other in members:
                    if other is member:continue
                    x,y,w,h=other['rect'];s=cam['scale']
                    context|=rect_mask(((x-cam['x'])*s+cam['offset_x'],
                        (y-cam['y'])*s+cam['offset_y'],w*s,h*s),shape)
                support=np.zeros(shape,bool)
                for r in (ra,rb):
                    if uid in r['objects']:support|=r['objects'][uid]['amodal']
                local=support&context
                if local.any():out.append(annotation('obstacle',bounding_mask(local),local,local,
                    np.zeros(shape,bool),'s5i5:actual_overlap_pair:fixed_arm',instances=[uid],
                    bbox_mode='local_contact_not_full_obstacle',visibility_status='not_yet_resolved'))
                else:
                    # This is not occlusion: support is amodal (other objects
                    # cannot erase it). A solid border may exist wholly outside
                    # the camera while still preventing visible arm movement.
                    endpoints=[s[uid] for s in (aa,bb) if uid in s]
                    opaque=[int(np.sum(np.asarray(o['pixels'])>=0)) for o in endpoints]
                    outside=(len(endpoints)==2 and all(opaque) and
                             all(o.get('visible',True) for o in endpoints) and not support.any())
                    notes.append({'kind':'roi_unavailable','category':'obstacle','instance_id':uid,
                        'reason':'no_in_frame_obstacle_pixels' if outside else 'contact_support_unresolved',
                        'source':'s5i5:actual_overlap_pair:fixed_arm',
                        'endpoint_opaque_cells':opaque,'endpoint_support_pixels':int(support.sum()),
                        'linked_interaction':'collision.arm_overlap_rollback',
                        'scoring_policy':'separate_collision_context; never fabricate obstacle pixels'})
    return out,notes


def reviewed_groups(game,categories):
    definitions={
        'ls20':[('refill_and_budget',['step_refill','steps_bar'],'source_coupled')],
        'wa30':[('facing_and_grab',['target_facing','target_grab'],'shared_object_state')],
        'sc25':[('resize_and_spell',['player_resize','spell_slot_grid'],'same_action_not_certified_causal')],
    }
    return [{'id':name,'categories':cats,'relation':relation,
             'source':'review_mapping:'+game+':'+name,'verified':False}
            for name,cats,relation in definitions.get(game,[]) if set(cats)<=set(categories)]
