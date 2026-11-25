"""Explicit evaluator-only replay compatibility; no shared game edits.

Keep GT semantic names/logs/hooks. Reorder original level instances rather than
replace sprites or recolor recorded images. Legacy profiles remain reproducible.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import textwrap
from types import MethodType

import numpy as np

from ..paths import DATA, ENVIRONMENT_FILES, GAME_SOURCES
PROFILE = 'reviewed_v1'
REPLAY_PROFILE = 'replay_v2'
RECOVERY_PROFILE = 'replay_v3'
# MechEval: replay_v3 without the recording-server compatibility of tn36 and s5i5 (legacy layer order, renderer and
# pick order), for logs played on the current offline engine (arcengine 0.9.3 with the official game files).
CURRENT_PROFILE = 'replay_v3_current'
PROFILES = ('raw_gt', PROFILE, REPLAY_PROFILE, RECOVERY_PROFILE, CURRENT_PROFILE)
ORDER_GAMES = {'r11l', 'ka59', 're86', 'su15'}
REFERENCES = {
    'r11l': ('495a7899', '95d58ebaa8c758e905821bb4f5881d47166c8f5aec158bfea7585902c01040d0'),
    'ka59': ('38d34dbb', '3e8341f4c317ceb9992ca54e8ad453ee4e8a04547b27a907f4a560a9a30d857e'),
    're86': ('8af5384d', 'cf3c1520e17f0b70e7a3cea0e680601a591befce306b4dd1926c735a7d3b65e8'),
    'su15': ('1944f8ab', 'a5f91f7c963d6ca6447dae0ab21342b48a3f511601c40dfa8e972bdc59b4651e'),
    's5i5': ('18d95033', '341f85bac1fb5ca7a3315a367405c3c993a57ce4dd0ff01cc8aaa72bf6c6ed34'),
    'tn36': ('ef4dde99', '43c30052cdeb230017eb1ec23877050d96297756b86071417868ec61138c9fb7'),
    'm0r0': ('492f87ba', 'fc3954236f712759c2a5baaada09bb3244e217e022086dc9ec8f6e6fe101d797'),
    'sk48': ('d8078629', 'b6ca05eed7ba81fe5a6a26c58410fa7daddf2039d3d1c42f600aaf9f6f0111e3'),
}


def metadata_diff(response, row):
    actual = {'levels_completed': int(response.levels_completed),
              'state': str(getattr(response.state, 'value', response.state)),
              'available_actions': sorted(int(x) for x in response.available_actions)}
    expected = {k: sorted(row[k]) if k == 'available_actions' else row[k]
                for k in actual if k in row}
    return {k: {'actual': actual[k], 'recorded': v} for k, v in expected.items() if actual[k] != v}


def sprite_key(sprite):
    pixels = np.asarray(sprite.render(), dtype=np.int16)
    return (sprite.x, sprite.y, pixels.shape, hashlib.sha256(pixels.tobytes()).hexdigest(), sprite.layer)


@lru_cache(None)
def reference_module(game_id):
    revision, digest = REFERENCES[game_id]
    path = ENVIRONMENT_FILES / f'{game_id}/{revision}/{game_id}.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Reference changed; re-audit compatibility: {path}')
    spec = importlib.util.spec_from_file_location('compat_reference_' + game_id, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(None)
def reference_orders(game_id):
    module = reference_module(game_id)
    # Include L3 initialization only to check the returned L2->L3 boundary.
    return tuple(tuple(sprite_key(s) for s in level.get_sprites()) for level in module.levels[:3])


def reorder_level(level, keys):
    groups = defaultdict(list)
    for sprite in level.get_sprites():
        groups[sprite_key(sprite)].append(sprite)
    ordered = []
    for key in keys:
        if len(groups[key]) != 1:
            raise ValueError('Missing/ambiguous original sprite identity; no fuzzy order fallback')
        ordered.append(groups[key].pop())
    if any(groups.values()):
        raise ValueError('Extra GT sprites; compatibility needs re-audit')
    level._sprites[:] = ordered
    level._need_sort = True
    level._sorted_sprites = None


def apply_profile(game_id, game, profile='raw_gt'):
    if profile not in PROFILES:
        raise ValueError('Unknown environment profile: ' + profile)
    info = {'profile': profile, 'changes': [], 'semantic_scope': 'L1/L2',
            's5i5_policy': 'keep_GT; recorded_visual_conflict_pending',
            'm0r0_policy': 'keep_GT; level_rule_conflict_pending'}
    game._evaluation_environment = info
    replay_game = profile in {REPLAY_PROFILE, RECOVERY_PROFILE} and game_id in {'tn36', 's5i5'}
    recovery_game = profile in {RECOVERY_PROFILE, CURRENT_PROFILE} and game_id in {'m0r0', 'sk48'}
    if profile == 'raw_gt' or (game_id not in ORDER_GAMES | {'tu93'} and not replay_game and not recovery_game):
        return game
    source = GAME_SOURCES / f'gamesofts/modify/{game_id}/{game_id}.py'
    digest = json.loads((DATA / 'interaction_source_manifest.json').read_text())[game_id]
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('GT source changed; re-audit compatibility: ' + str(source))
    info['gt_source_sha256'] = digest
    if recovery_game:
        reference_module(game_id)
        info['reference_revision'], info['reference_sha256'] = REFERENCES[game_id]
        if game_id == 'm0r0':
            original_step = game.step.__func__
            lines, start = inspect.getsourcelines(original_step)
            # Instrumented collision predicates live in _move_orb, not step.
            # Fail closed if later hooks move into the recompiled method.
            from .interaction_rules import RULES
            if any(r.game == game_id and start <= r.line < start + len(lines) for r in RULES):
                raise ValueError('m0r0 step now contains interaction hooks; re-audit patch')
            source_code = textwrap.dedent(''.join(lines))
            target = 'if horizontal_adj or vertical_adj:'
            if source_code.count(target) != 1:
                raise ValueError('m0r0 predicate changed')
            namespace = dict(original_step.__globals__)
            exec(compile(source_code.replace(target, 'if horizontal_adj:'),
                         '<replay_v3 m0r0 horizontal-only>', 'exec'), namespace)
            game.step = MethodType(namespace['step'], game)
            info['m0r0_policy'] = 'original horizontal-only matching; GT labels retained'
            info['changes'] = ['remove GT-added vertical adjacency match']
        else:
            original = game.on_set_level

            def on_set(level):
                # Only L3 entry is affected; L1/L2 samples remain unchanged.
                if game.level_index == 2:
                    sprites = game._clean_levels[2].get_sprites_by_name('skewer_yellow')
                    if len(sprites) != 1:
                        raise ValueError('sk48 L3 yellow skewer identity changed')
                    # GT added 12 gray pixels absent from the pinned original
                    # sprite asset. Restore the asset, not the recorded image.
                    sprites[0].color_remap(4, 5)
                    ref = reference_module(game_id).sprites['zkekdulqku']
                    if not np.array_equal(sprites[0].pixels, ref.pixels):
                        raise ValueError('sk48 original sprite asset no longer matches')
                original(level)
            game.on_set_level = on_set
            info['changes'] = ['restore original L3 yellow skewer asset (12 gray pixels -> black); boundary verification only']
        return game
    if replay_game:
        reference = reference_module(game_id)  # hash check before compatibility
        info['reference_revision'], info['reference_sha256'] = REFERENCES[game_id]
        original = game.on_set_level
        if game_id == 'tn36':
            info['changes'] = ['fixed-tablet background layer 2 -> original 4 after level clone']

            def on_set(level):
                original(level)
                # tn36 replaces current_level inside on_set_level. The argument
                # now refers to the discarded level; patch the actual clone.
                current = game.current_level
                for sprite in current.get_sprites_by_name('etitgfbxdb'):
                    sprite.set_layer(4)
                current._need_sort = True
                current._sorted_sprites = None
        else:
            info['s5i5_policy'] = 'recorded original renderer and legacy pick ordering; GT labels retained'
            info['changes'] = ['target_dot layer 10 -> original 1',
                               'original initial sprite order and original camera compositing',
                               'legacy in-place descending layer sort at get_sprite_at']
            orders = reference_orders(game_id)
            game.camera._raw_render = MethodType(reference.gdivpwtbrb._raw_render, game.camera)

            def on_set(level):
                for sprite in level.get_sprites_by_tag('target_dot'):
                    sprite.set_layer(1)
                if game.level_index < len(orders):
                    reorder_level(level, orders[game.level_index])
                original(level)
                current = game.current_level
                # Use the class method to avoid stacking wrappers after RESET.
                pick = type(current).get_sprite_at.__get__(current)

                def legacy_pick(x, y, tag=None, ignore_collidable=False):
                    current._sprites.sort(key=lambda s: s.layer, reverse=True)
                    current._need_sort = True
                    return pick(x, y, tag, ignore_collidable)
                current.get_sprite_at = legacy_pick
        game.on_set_level = on_set
        return game
    original = game.on_set_level
    orders = reference_orders(game_id) if game_id in ORDER_GAMES else None
    if orders:
        info['changes'] = ['original initial sprite order before on_set_level (L1/L2 + boundary L3)']
        info['reference_revision'], info['reference_sha256'] = REFERENCES[game_id]
    else:
        info['changes'] = ['L1 camera height 38 -> 39']

    def on_set(level):
        if orders and game.level_index < len(orders):
            reorder_level(level, orders[game.level_index])
        original(level)
        if game_id == 'tu93' and game.level_index == 0:
            game.camera._height = 39
    game.on_set_level = on_set
    return game
