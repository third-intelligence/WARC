"""Evaluation scope v2: predict action-completed observations, not animations.

Keep released categories/review answers intact. These are evaluation exclusions,
not claims that a gimmick is absent from the game. Physical outcomes remain in
their original categories (e.g. blast excluded, pushed player/bomb retained).
"""
import numpy as np
import re

VERSION=3
# Semantic review decisions are separate from temporal exclusions. Preserve the
# released catalog and raw events so old review files remain importable.
SEMANTIC_EXCLUDED={
    'dc22':{'course_gimmick':'excluded as a category of its own; the moves, rotations and pattern changes of real objects are kept under their objects'},
}
EXCLUDED={
    'ft09':{'border_flash':'the frame flash ends within the action (step blink_counter)'},
    'ka59':{'explosion_blast':'the blast sprite is removed before the action completes (step explosion cleanup)'},
    'ls20':{'hud_flash':'the HUD flash is cleared before complete_action (step akoadfsur)'},
    'su15':{'block_flash':'the cells return to their colours (_finish_flash_animation)'},
    'tn36':{'target_flash':'the target highlight is cleared (tick_animation)'},
}
PENDING={}
ENDPOINT_ONLY={
    'r11l':{'goal_flash':'the flashing in between is excluded; the goal overlay left at the end of the action is scored'},
    'sk48':{'block_flash':'the flashing in between is excluded; the white left at complete_action is scored'},
    'ls20':{'lock_match':'the transient lock highlight is excluded; the lock state at the end of the action is scored'},
    'sb26':{'background_flash':'the final colour of bg_tile, separate from wrong_answer_flash; intermediate colours are excluded'},
    'tn36':{'ghost_overlay':'the visibility of the barrier, separate from the transient player ghost; the display at the end of the action is scored'},
}
ENDPOINT_ONLY['sb26'].update({
    'reference_cursor':'moves and flashes during the check are excluded; only the visibility of selected_highlight at the end of the action is a candidate',
    'outline_frame':'the frame left at the end of the action is scored; its path during the check is excluded',
    'tile':'the tile after placement or a move is scored',
})
ENDPOINT_ONLY['su15']={'reference_display':'the display at the end of the action is scored, including the grey left after a failure',
                       'goal_9x9':'the display at the end of the action is scored; level transitions are separate'}
ENDPOINT_ONLY['tu93']={'player_block_on_path':'the moves and turns of the player are scored'}
ENDPOINT_ONLY['s5i5']={'fixed_arm':'a fixed obstacle'}
ENDPOINT_ONLY['dc22']={'ghost':'kept apart from the visibility of coloured tiles, since it also covers moves and rotations of bridges and other objects'}


def scope(game,category):
    if category in SEMANTIC_EXCLUDED.get(game,{}):
        return {'status':'excluded_semantic_category','reason':SEMANTIC_EXCLUDED[game][category]}
    if category in EXCLUDED.get(game,{}):
        return {'status':'excluded_intermediate_only','reason':EXCLUDED[game][category]}
    if category in PENDING.get(game,{}):
        return {'status':'pending_semantics','reason':PENDING[game][category]}
    return {'status':'endpoint_candidate','reason':ENDPOINT_ONLY.get(game,{}).get(category,'the state at the end of the action is scored')}


def filter_events(game,events,objects=()):
    kept=[];excluded=[]
    for ev in events:
        cat=ev.get('gimmick_id');decision=scope(game,cat)
        if game=='ls20' and cat=='lock_match' and ev.get('event','').startswith('[bright]:'):
            decision={'status':'excluded_intermediate_only','reason':'a transient lock highlight; the change of the lock state at the end is kept as another event'}
        name=re.search(r'sname\(([^)]+)\)',ev.get('event',''))
        matching=[o for o in objects if o.get('kind')=='object' and name and o.get('name')==name[1]]
        if matching and all(o.get('change')=='intermediate_only' for o in matching):
            decision={'status':'excluded_intermediate_only','reason':'the object exists only during the action, in neither the frame before nor the frame after'}
        if decision['status']=='endpoint_candidate':kept.append(ev)
        else:excluded.append({**ev,'evaluation_scope':decision})
    return kept,excluded


def compare_frames(actual,recorded):
    final=bool(len(actual) and len(recorded) and np.array_equal(actual[-1],recorded[-1]))
    all_equal=len(actual)==len(recorded) and all(np.array_equal(a,b) for a,b in zip(actual,recorded))
    return {'final_equal':final,'all_subframes_equal':bool(all_equal),
            'actual_subframes':len(actual),'recorded_subframes':len(recorded)}
