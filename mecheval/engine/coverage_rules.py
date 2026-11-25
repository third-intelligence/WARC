"""Rules that assign the logged events of a game program to mechanics.

The games with a table in data/event_maps resolve their events through it (event_map.resolve_gt); the other
games use the rules of prompt_category below. An event that no rule matches is left unassigned.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from ..paths import DATA
PAT = re.compile(r'^\[([^]]+)\]:\[([^]]+)\]:\[([^]]*)\]:(.*)$')
from . import event_map as common


def prompt_category(game, verb, target, detail, text):
    pair = verb, target
    if game == 'ar25':
        if target == 'piece' and verb in {'move', 'appear', 'disappear'}:
            if '(reflection)' in text:
                return 'reflection'
            if verb == 'move':
                return 'piece'
        if pair == ('move', 'mirror_axis'): return 'mirror_axis'
        if verb == 'bright' and target in {'piece', 'mirror_axis'}: return 'selection'
        if pair == ('change', 'steps'): return 'step_bar'
    elif game == 'cd82':
        # The prompt explicitly calls its target map ASSUMED. Do not silently
        # equate actual ActiveBasket/canvas/steps aliases to those examples.
        return {'energy': 'energy_bar', 'timer': 'energy_bar', 'bar': 'energy_bar',
                'container': 'container_tilt', 'reservoir': 'reservoir_recolor',
                'selector': 'selection_marker', 'marker': 'selection_marker',
                'stack': 'stack_fill'}.get(target)
    elif game == 'ft09':
        return {('change', 'color_tile'): 'color_tile',
                ('change', 'click_budget'): 'click_budget',
                ('bright', 'border_frame'): 'border_flash'}.get(pair)
    elif game == 'lp85':
        if verb == 'move' and target in {'tile', 'goal_tile_yellow', 'goal_tile_orange'}: return 'tiles'
        if pair == ('change', 'steps'): return 'life_bar'
        if pair == ('change', 'level_progress'): return 'level_pips'
    elif game == 'ls20':
        if pair == ('move', 'player'): return 'player'
        if pair == ('rotate', 'hud_piece'): return 'piece_rotation'
        if re.fullmatch(r'lock_\d+', target) and verb in {'bright', 'match', 'unlock', 'unmatch'}: return 'lock_match'
        if verb == 'bright' and target in {'goal_highlight_border', 'goal_highlight_frame'}: return 'hud_flash'
        if pair == ('change', 'steps'): return 'steps_bar'
    elif game == 'm0r0':
        if pair == ('move', 'orb_normal_left'): return 'player'
        if pair == ('move', 'orb_normal_right'): return 'mirror_player'
        if pair == ('change', 'steps'):
            if 'top' in text.lower(): return 'life_bar'
            if 'bottom' in text.lower(): return 'mirror_life_bar'
    elif game == 'tr87':
        if pair == ('change', 'moves'): return 'bottom_timer_bar'
        if pair == ('move', 'selector_top1'): return 'selector_bracket_pair'
        if verb == 'change' and target.startswith('symbol_set') and 'shape(' in detail: return 'slot1_glyph_cycle'
    elif game == 'tu93':
        if target == 'player' and verb in {'move', 'rotate', 'change', 'disappear'}: return 'player_block_on_path'
        if pair == ('change', 'steps'): return 'bottom_energy_bar'
        if target.startswith(('orange_ball', 'teal_ball', 'magenta_ball')) and verb in {'move', 'change'}: return 'enemy_ball'
    elif game == 'vc33':
        if pair == ('change', 'steps'): return 'top_timer_bar'
        if target.startswith(('joint_', 'body_', 'connector_', 'slot_')) and verb in {'move', 'extend', 'shrink'}: return 'yellow_bird_blob'
    elif game == 'wa30':
        if target.startswith('selector_at_') and verb in {'move', 'rotate'}: return 'moving_green_block'
        if pair == ('change', 'steps'): return 'bottom_timer_bar'
        if target.startswith('wall_block_at_') and verb == 'change': return 'yellow_button_toggle'
    return None
