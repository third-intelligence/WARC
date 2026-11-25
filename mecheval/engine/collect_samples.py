"""Replay a play through a hooked game program and collect, per action, the logged events and the object snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import random
import numpy as np
from .arc_adapter import load_game,Snapshotter,action_with_snapshots,ROOT
from .mapping import map_transition,raster,indices
from .event_binding import EventBinder
from .source_hooks import SourceHooks
from .endpoint_metrics import endpoint_regions,evaluate
from .game_regions import annotations,joint_groups
from .endpoint_scope import filter_events,compare_frames,VERSION as SCOPE_VERSION
from .interactions import InteractionObserver,interaction_annotations
from .review_mapping import migrate_events,reviewed_annotations,reviewed_groups,VERSION as REVIEW_VERSION
from .replay_io import observations,action_id
from .environment_compat import metadata_diff


def collect(game_id, steps, environment_profile='reviewed_v1', *, recording_path=None,
            transition_sink=None,boundary_sink=None,include_initial=True,compute_metrics=True,
            compute_occurrences=False,recover_initial=False):
    observer=InteractionObserver(game_id)
    partial={'game':game_id,'transitions':[]}
    try:return _collect(game_id,steps,observer,partial,environment_profile,recording_path,
                        transition_sink,boundary_sink,include_initial,compute_metrics,compute_occurrences,recover_initial)
    except Exception as exc:
        partial['error']=type(exc).__name__+': '+str(exc)
        partial.setdefault('alignment',{}).update(status='collector_error',error=partial['error'])
        return partial
    finally:
        if getattr(observer,'source_hooks',None):observer.source_hooks.close()
        observer.close()


def _collect(game_id, steps, observer, result, environment_profile,recording_path,
             transition_sink,boundary_sink,include_initial,compute_metrics,compute_occurrences,recover_initial):
    from arcengine import ActionInput,GameAction
    random.seed(0);np.random.seed(0)
    game=load_game(game_id,interactions=observer,environment_profile=environment_profile)
    hooks=SourceHooks(game_id,game)
    observer.source_hooks=hooks
    reset_response=game.perform_action(ActionInput(id=GameAction.RESET),raw=True)
    available_actions=reset_response.available_actions
    result.update(initial=[],transitions=[],boundary_transitions=[],alignment={},
                  evaluation_scope_version=SCOPE_VERSION,review_mapping_version=REVIEW_VERSION,
                  environment=dict(game._evaluation_environment))
    # Diagnostic direct L2 initialization, NOT proof of reaching L2 in replay.
    for level in ((1,2) if include_initial else ()):
        g=load_game(game_id,environment_profile=environment_profile)
        g.perform_action(ActionInput(id=GameAction.RESET),raw=True)
        if level==2:g.set_level(1)
        snap=Snapshotter(game_id).capture(g);rr=raster(snap)
        result['initial'].append({'game':game_id,'level':level,'frame':snap['frame'],
            'label':f'GT source + {environment_profile} / direct level initialization / not human reviewed',
            'engine':snap['engine_version'],
            'objects':[{'id':uid,'name':x['object']['name'],'tags':x['object'].get('tags',[]),
                        'roi':indices(x['visible']),'amodal_roi':indices(x['amodal'])}
                       for uid,x in rr['objects'].items()],
            'unattributed':indices(rr['unattributed'])})
    if steps==0:return result
    if recording_path is None:raise ValueError('a recording path is required')
    paths=[Path(recording_path)]
    if not paths:
        result['alignment']={'status':'no_recording'};return result
    path=paths[0]
    # Default remains one prefix; explicit paths and sinks permit bounded-memory audits.
    rows=observations(path,require_initial_reset=not recover_initial)
    first_row=next(rows)
    reconstructed = action_id(first_row['action_input']['id']) != 0
    proof = None
    if reconstructed:
        # A candidate initial state is not accepted from one matching image.
        # Replay the complete L1/L2 segment independently before scoring any row.
        from .validate_environment_compat import validate
        proof = validate(game_id,path,environment_profile,reconstruct_initial=True)
        if proof['status'] != 'matched':
            result['alignment']={'status':'unverified_initial_state','recovery_validation':proof}
            return result
        ai=first_row['action_input'];aid=action_id(ai['id'])
        reset_response=game.perform_action(ActionInput(id=getattr(GameAction,f'ACTION{aid}'),data=ai.get('data') or {}),raw=True)
        available_actions=reset_response.available_actions
    expected=np.asarray(first_row['frame'][-1]); tracker=Snapshotter(game_id,hooks=hooks)
    before=tracker.capture(game,frame=reset_response.frame[-1])
    result['alignment']={'recording':str(path),'recorded_game_id':first_row.get('game_id'),
                         'engine':before['engine_version'],'initial_equal':bool(np.array_equal(expected,before['frame'])),
                         'matched_actions':0,'requested_actions':steps,'status':'prefix_checked',
                         'policy':'action_completed_final_frame','intermediate_mismatch_actions':[],
                         'metadata_warnings':[]}
    if reconstructed:
        result['alignment'].update(initialization='reconstructed_standard_reset_full_replay_verified',
            first_unscored_action_index=0,exposure_origin='after_first_recorded_observation',
            recovery_validation={k:v for k,v in proof.items() if k!='first_mismatch'})
    if not result['alignment']['initial_equal']:
        result['alignment']['status']='initial_frame_mismatch';return result
    initial_metadata=metadata_diff(reset_response,first_row)
    if 'available_actions' in initial_metadata:
        result['alignment']['metadata_warnings'].append({'step':0,'available_actions':initial_metadata['available_actions']})
    if any(k!='available_actions' for k in initial_metadata):
        result['alignment'].update(status='initial_metadata_mismatch',metadata_diff=initial_metadata);return result
    binder=EventBinder(game_id)
    from itertools import islice
    for i,row in enumerate(rows if steps is None else islice(rows,steps),1):
        if before['level']>2:
            result['alignment']['stop_reason']='reached_L3';break
        ai=row['action_input']; aid=action_id(ai['id'])
        action=ActionInput(id=GameAction.RESET if aid==0 else getattr(GameAction,f'ACTION{aid}'),data=ai.get('data') or {})
        hooks.begin_action()
        observer.begin_action(action,available_actions)
        try:
            response, snapshots=action_with_snapshots(game,tracker,action)
            available_actions=response.available_actions
        except Exception as e:
            result['alignment'].update(status='collector_error',mismatch_action=i,
                error=type(e).__name__+': '+str(e));break
        if (not snapshots and not response.frame and not row['frame'] and
            str(getattr(response.state,'value',response.state)) in ('GAME_OVER','WIN') and
            not metadata_diff(response,row)):
            result['alignment'].setdefault('empty_terminal_actions',[]).append(i)
            result['alignment']['matched_actions']+=1
            # These terminal no-ops have no visual target. Preserve raw index
            # and wait for a RESET; never manufacture a frame or train on it.
            continue
        if not snapshots:
            result['alignment'].update(status='missing_render_snapshot',mismatch_action=i);break
        after=snapshots[-1]
        comparison=compare_frames(response.frame,row['frame'])
        metadata=metadata_diff(response,row)
        if not comparison['final_equal']:
            result['alignment'].update(status='frame_mismatch',mismatch_action=i,
                metadata_diff=metadata,**comparison)
            break
        if 'available_actions' in metadata:
            result['alignment']['metadata_warnings'].append({'step':i,'available_actions':metadata['available_actions']})
        if any(k!='available_actions' for k in metadata):
            result['alignment'].update(status='metadata_mismatch',mismatch_action=i,
                metadata_diff=metadata,**comparison)
            break
        if not comparison['all_subframes_equal']:
            result['alignment']['intermediate_mismatch_actions'].append(i)
        result['alignment']['matched_actions']+=1
        if aid!=0 and before['level']==after['level'] and all(s['level']==before['level'] for s in snapshots):
            mapped=map_transition(before,after,intermediate=snapshots[:-1])
            interaction=observer.finish(before,after,tracker)
            interaction_regions=endpoint_regions(before,after,[],
                annotations=interaction_annotations(before,after,interaction))
            invalid=({'confirmed':True,'source':[e['source'] for e in interaction['events']
                if e['kind']=='invalid_action']} if interaction['invalid_action'] else None)
            mapped['interactions']=interaction
            mapped['interaction_regions']=interaction_regions
            if compute_metrics:
                mapped['interaction_copy_last']=evaluate(before['frame'],interaction_regions,
                    include_candidates=True,invalid_action=invalid)
            logs=game.get_action_log()[0] if hasattr(game,'get_action_log') else list(game.action_log)
            raw_events=binder.bind(logs,before,after,mapped)
            events,withheld=migrate_events(game_id,raw_events,before,after)
            eligible,excluded=filter_events(game_id,events,mapped['objects'])
            mapped['eligible_event_categories']=sorted({e['gimmick_id'] for e in eligible if e.get('gimmick_id')})
            mapped['released_events']=raw_events
            mapped['review_withheld_events']=withheld
            mapped['review_mapping_version']=REVIEW_VERSION
            mapped['objects'].extend(eligible)
            mapped['excluded_events']=excluded
            extra,notes=annotations(before,after)
            reviewed,review_notes=reviewed_annotations(before,after,interaction,aid)
            extra.extend(reviewed);notes.extend(review_notes)
            if compute_occurrences:
                from .gimmick_occurrences import detect
                mapped['gimmick_occurrences']=detect(before,after,eligible,extra,interaction)
            # Prefer explicitly source-grounded annotations over stale text
            # bbox aliases (ar25 reflections and fixed HUD metadata).
            covered={r['category'] for r in extra}
            scoring_events=[r for r in eligible if r.get('gimmick_id') not in covered]
            regions=endpoint_regions(before,after,scoring_events,annotations=extra)
            groups=joint_groups(game_id,regions['categories'])+reviewed_groups(game_id,regions['categories'])
            regions=endpoint_regions(before,after,scoring_events,groups=groups,annotations=extra)
            mapped['endpoint_regions']=regions
            if compute_metrics:
                mapped['endpoint_copy_last']=evaluate(before['frame'],regions,include_candidates=True)
            mapped['endpoint_notes']=notes
            mapped['interaction_step']=i
            mapped['observation_policy']='action_completed_final_frame'
            mapped['gt_log']=logs
            mapped.update(game=game_id,action_index=i,action=ai,source='human recording prefix, completed frames matched',
                          exposure_origin='after_first_recorded_observation' if reconstructed else 'recorded_reset',
                          initialization='reconstructed_standard_reset_full_replay_verified' if reconstructed else 'recorded_reset',
                          frame_comparison=comparison,evaluation_scope_version=SCOPE_VERSION,
                          sample_name=f'{game_id} human L{before["level"]} action {i}',
                          recording=str(path),kind='real_human_prefix' if steps is not None else 'real_human_L1_L2')
            if transition_sink is None:result['transitions'].append(mapped)
            else:transition_sink(mapped)
        else:
            # Keep the exact WM target at RESET/level/terminal transitions for
            # later review. Never substitute an internal win flash for the
            # actual returned final observation or mix two levels' identities.
            boundary={'action_index':i,'action':ai,
                'before_level':before['level'],'after_level':after['level'],
                'before':before['frame'],'after':after['frame'],
                'subframe_levels':[s['level'] for s in snapshots],
                'state':str(response.state),'frame_comparison':comparison,
                'reason':'reset' if aid==0 else 'level_boundary',
                'gimmick_scoring_status':'not_scored_across_level_boundary'}
            if boundary_sink is None:result['boundary_transitions'].append(boundary)
            else:boundary_sink(boundary)
        before=after
        if after['level']>2:
            result['alignment']['stop_reason']='reached_L3';break
    if steps is None and result['alignment']['status']=='prefix_checked':
        result['alignment']['status']='L1_L2_checked'
        result['alignment'].setdefault('stop_reason','recording_end')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--game',required=True);p.add_argument('--steps',type=int,default=0)
    p.add_argument('--environment-profile',choices=['raw_gt','reviewed_v1'],default='reviewed_v1')
    args=p.parse_args()
    try:
        result=collect(args.game,args.steps,args.environment_profile)
        if args.steps:
            from check_noninterference import check
            try:result['noninterference']=check(args.game,args.steps,args.environment_profile)
            except Exception as e:result['noninterference']={'status':'error','error':type(e).__name__+': '+str(e)}
        print(json.dumps(result,ensure_ascii=False))
    except Exception as e:
        print(json.dumps({'game':args.game,'error':type(e).__name__+': '+str(e)}));sys.exit(1)
