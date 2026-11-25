"""Offline snapshot adapter for the local arcengine, without editing games.

Uses real sprite.render(): rotations, mirroring, scaling are already applied.
This is an analysis adapter, not a WM input or a benchmark agent.
"""
from __future__ import annotations
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import sys
import numpy as np

from ..paths import GAME_SOURCES as SOURCE
ROOT=None  # the research tree is not part of the package; recordings are always given explicitly


def load_game(game, interactions=None, environment_profile='raw_gt'):
    if str(SOURCE) not in sys.path: sys.path.insert(0,str(SOURCE))
    from arcengine import ARCBaseGame
    module=importlib.import_module(f'gamesofts.modify.{game}.{game}')
    if interactions is not None:
        from .interactions import private_module
        module=private_module(module, interactions)
    classes=[v for v in vars(module).values() if inspect.isclass(v)
             and issubclass(v,ARCBaseGame) and v is not ARCBaseGame
             and v.__module__==module.__name__]
    if len(classes)!=1: raise ValueError('ambiguous game class')
    instance=classes[0]()
    from .environment_compat import apply_profile
    instance=apply_profile(game,instance,environment_profile)
    if interactions is not None:interactions.attach(instance)
    return instance


class Snapshotter:
    def __init__(self, game_id, hooks=None):
        self.game_id=game_id
        self.hooks=hooks
        self._refs={}  # retain objects so Python id cannot be reused mid-run
        self._ids={}

    def capture(self, game, frame=None):
        cam=game.camera
        scale,ox,oy=cam._calculate_scale_and_offset()
        objects=[]
        for spr in game.current_level.get_sprites():
            key=id(spr)
            if key not in self._ids:
                self._ids[key]=f'{self.game_id}:L{game.level_index+1}:i{len(self._ids)}'
                self._refs[key]=spr
            obj={'instance_id':self._ids[key], 'name':spr.name,'tags':list(spr.tags),
                 'x':spr.x,'y':spr.y,'layer':spr.layer,'visible':spr.is_visible,
                 # These two source cameras composite raw pixels, not render().
                 'pixels':(spr.pixels if self.game_id in {'s5i5','wa30'} else spr.render()).astype(int).tolist(),
                 'rotation':spr.rotation,'scale':spr.scale,'parts':{}}
            # Evaluator-only source identities, never inferred from screen color.
            obj['evaluation_roles']={}
            if self.game_id=='cd82':
                if spr.name in {'ActiveBasket','crane_arm'}:
                    obj['parts']['interior']=(np.asarray(obj['pixels'])==game.selected_color).tolist()
                if self.hooks and spr.name=='canvas':
                    mask=np.zeros(spr.pixels.shape,bool)
                    for row in self.hooks.paint:
                        if row['sprite'] is spr:mask|=row['mask']
                    mask=np.rot90(mask,k=int((-spr.rotation%360)/90))
                    if spr.mirror_ud:mask=np.flipud(mask)
                    if spr.mirror_lr:mask=np.fliplr(mask)
                    obj['parts']['painted']=mask.repeat(spr.scale,0).repeat(spr.scale,1).tolist()
            elif self.game_id=='r11l':
                obj['evaluation_roles']['selected_arm']=spr is game.selected_leg
                obj['evaluation_roles']['arm']=any(spr is leg for leg in game.all_legs)
                if obj['evaluation_roles']['arm']:
                    obj['parts']['selection_color']=np.isin(obj['pixels'],[0,3]).tolist()
            elif self.game_id=='s5i5':
                obj['evaluation_roles']['arm_role']=game._arm_role_by_id.get(key)
            elif self.game_id=='sc25':
                obj['evaluation_roles']['player']=spr is game.cflwkfoggh
                obj['evaluation_roles']['spell_slot']=any(spr is s for row in game.clickable_spell_slots for s in row)
            elif self.game_id=='wa30' and 'target' in spr.tags:
                # _paint_target_border writes these exact raw sprite pixels;
                # wa30 camera composites raw pixels without Sprite.render().
                if spr.pixels.shape==(4,4):
                    border=np.zeros((4,4),bool);border[[0,3],:]=True;border[:,[0,3]]=True
                    obj['parts']['border']=border.tolist()
                    colors=set(int(c) for c in spr.pixels[border])
                    obj['evaluation_roles']['target_border_state']=({0:'selector_grab',3:'facing',5:'other_grab',4:'free'}.get(next(iter(colors))) if len(colors)==1 else None)
            if self.game_id=='ka59':
                match=next((t for t in ('player','bomb','enemy','explosion_blast') if t in spr.tags),None)
                obj['gimmick_id']={'enemy':'cross_object'}.get(match,match)
                obj['mapping_verified']=False
                if 'player' in spr.tags and spr.scale>=1:
                    # Source ka59.set_center_color: keep the center separate
                    # but retain the ENTIRE body ROI as the parent object.
                    h,w=spr.pixels.shape
                    center=np.zeros((h,w),bool)
                    if (h,w)==(3,3): center[1,1]=True
                    elif (h,w)==(6,3): center[2:4,1]=True
                    elif (h,w)==(3,6): center[1,2:4]=True
                    else: center[2:4,2:4]=True
                    center &= spr.pixels>=0
                    def transform(a):
                        a=np.rot90(a,k=int((-spr.rotation%360)/90))
                        if spr.mirror_ud:a=np.flipud(a)
                        if spr.mirror_lr:a=np.fliplr(a)
                        return a.repeat(spr.scale,0).repeat(spr.scale,1)
                    obj['parts']={'center':transform(center).tolist(),
                                  'body_without_center':transform((spr.pixels>=0)&~center).tolist()}
            objects.append(obj)
        # Only direct diagnostic captures render. Action collection passes the
        # REAL output, never renders a second time to obtain a snapshot.
        frame=np.asarray(frame if frame is not None else cam.render(game.current_level.get_sprites()),dtype=int)
        special={}
        if self.game_id=='ls20':
            special['locks_unlocked']=[bool(x) for x in game.lvrnuajbl]
        if self.hooks:
            special['reflections']=[{'source_instance':self._ids.get(key),
                'source_name':row['sprite'].name,'cells':sorted(row['cells'])}
                for key,row in self.hooks.reflections.items()]
            special['paint']=[{'source_instance':self._ids.get(id(row['sprite'])),
                'cells':np.argwhere(row['mask']).tolist(),'x':row['x'],'y':row['y'],
                'rotation':row['rotation'],'scale':row['scale'],'source':row['source']}
                for row in self.hooks.paint]
        return {'game':self.game_id,'level':game.level_index+1,'frame':frame.tolist(),
                'camera':{'x':cam.x,'y':cam.y,'width':cam.width,'height':cam.height,
                          'scale':scale,'offset_x':ox,'offset_y':oy,
                          'background':cam.background,'letter_box':cam.letter_box},
                'objects':objects,'special':special,
                'nonsprite':game.get_nonsprite_status() if hasattr(game,'get_nonsprite_status') else [],
                'engine_version':importlib.metadata.version('arcengine'),
                'source':'GT source copy + locally installed engine; not human-replay-validated'}


def action_with_snapshots(game, tracker, action):
    """Capture each actual rendered frame; restore camera even on failure."""
    if hasattr(game,'reset_logs'): game.reset_logs()
    original=game.camera.render
    snapshots=[]
    action_code=game.perform_action.__func__.__code__
    def render(sprites):
        # Some games render an internal reference/snapshot during step(). It
        # is not an observation emitted by perform_action. Leave it untouched.
        caller=inspect.currentframe().f_back
        external=caller.f_code is action_code
        del caller
        if not external:
            return original(sprites)
        # Freeze geometry before the real render: a user display may mutate
        # state, consume a frame queue, or even advance game status.
        snapshot=tracker.capture(game,frame=np.zeros((64,64),dtype=int))
        result=original(sprites)
        snapshot['frame']=np.asarray(result,dtype=int).copy().tolist()
        snapshots.append(snapshot)
        return result
    game.camera.render=render
    try:response=game.perform_action(action,raw=True)
    finally:game.camera.render=original
    if len(response.frame)!=len(snapshots): raise ValueError('subframe count mismatch')
    for expected,snapshot in zip(response.frame,snapshots):
        if not np.array_equal(expected,snapshot['frame']): raise ValueError('snapshot/frame mismatch')
    return response,snapshots
