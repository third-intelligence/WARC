"""Audited execution sites in the pinned GT source, NOT collision queries.

Line numbers are guarded by the hashes in data/interaction_source_manifest.json before instrumentation.
body/else observers run only when the original branch actually executes.
Expressions read locals/attributes; they never call game geometry or dynamics.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    game: str
    line: int
    branch: str  # body / orelse / before
    kind: str
    reason: str
    actor: str = 'None'
    other: str = 'None'
    guard: str = 'True'
    attempt: str = 'None'  # world-space actor bbox at attempted location


RULES = []


def add(game, line, branch, kind, reason, actor='None', other='None', **kw):
    RULES.append(Rule(game, line, branch, kind, reason, actor, other, **kw))


# Boundary rejection is not an input error. Mirror overlap checks in ar25 are
# NOT used by step() to reject movement; do not label those helpers collisions.
for line in (2248, 2258):
    add('ar25', line, 'body', 'collision', 'boundary', 'self.llludejph',
        attempt='(yptggrfrgo, wymrczshbz, crzzncewfw, nqnagkqeqy)')
add('ar25', 2224, 'body', 'constraint', 'fixed_selection', 'self.llludejph')
add('ar25', 2237, 'body', 'constraint', 'mirror_axis_direction', 'self.llludejph',
    guard='iyaddaovv != 0')
add('ar25', 2239, 'body', 'constraint', 'mirror_axis_direction', 'self.llludejph',
    guard='pshzrdxfu != 0')
add('ar25', 2315, 'orelse', 'invalid_action', 'off_grid')
add('ar25', 2348, 'orelse', 'invalid_action', 'no_click_target')
# Selection topology constraints, not a physical wall.
add('cd82', 1089, 'body', 'constraint', 'unselectable_center')
add('cd82', 1092, 'orelse', 'constraint', 'selection_limit')
add('cd82', 1099, 'body', 'invalid_action', 'off_grid')
add('cd82', 1103, 'body', 'invalid_action', 'no_click_target')
add('dc22', 3026, 'body', 'collision', 'solid_object', 'sprite', 'other')
add('dc22', 3564, 'body', 'collision', 'bridge_overlap', 'ftjesbxxhm',
    attempt='(jtmidhkkhs, lpfvopvoze, ftjesbxxhm.width, ftjesbxxhm.height)')
add('ka59', 1750, 'body', 'collision', 'wall_or_pit', 'player', 'blocker')
add('ka59', 1759, 'body', 'collision', 'movable_object', 'player', 'collisions')
add('ka59', 1803, 'body', 'collision', 'wall', 'sprite', 'blocked')
# A chain push is a contact even when the push succeeds.
add('ka59', 1811, 'before', 'collision', 'chain_push', 'sprite', 'chained')
add('lp85', 22005, 'body', 'invalid_action', 'no_button')
add('ft09', 2332, 'orelse', 'invalid_action', 'off_grid')
add('ft09', 2347, 'body', 'invalid_action', 'no_click_target')
add('ls20', 2402, 'body', 'collision', 'wall', 'self.gudziatsk', 'mvcsnkcqz',
    attempt='(juldcpkjse, ullicjtklz, self.gisrhqpee, self.tbwnoxqgc)')
add('ls20', 2407, 'body', 'collision', 'locked_gate', 'self.gudziatsk', 'mvcsnkcqz',
    attempt='(juldcpkjse, ullicjtklz, self.gisrhqpee, self.tbwnoxqgc)')
add('ls20', 2041, 'before', 'collision', 'push', 'self.sprite', 'aqxtoxeino')
add('m0r0', 1608, 'body', 'collision', 'boundary', 'spr',
    attempt='(new_x, new_y, spr.width, spr.height)')
add('m0r0', 1614, 'body', 'collision', 'wall', 'spr', 'wall_spr')
add('m0r0', 1619, 'body', 'collision', 'obstacle', 'spr', 'obs_spr')
add('m0r0', 1625, 'body', 'collision', 'barrier', 'spr', 'barrier_spr')
# Observe the committed rejection, not speculative would_collide_with_wall.
add('r11l', 2412, 'orelse', 'collision', 'wall', 'self.selected_leg',
    attempt='(target_x, target_y, self.selected_leg.width, self.selected_leg.height)')
add('r11l', 2241, 'body', 'collision', 'wall_body_response', 'hit_body', 'wall_spr2')
# Selecting an arm is not an invalid action or a collision even without motion.
add('r11l', 2393, 'before', 'selection', 'arm_selection', 'clicked_leg')
add('re86', 2324, 'body', 'collision', 'boundary', 'sprite')
add('re86', 2330, 'body', 'collision', 'wall_deformation', 'sprite', 'wall_spr')
add('re86', 2497, 'body', 'contact', 'recolor', 'sprite', 'target')
for line in (3267, 3293):
    add('s5i5', line, 'body', 'collision', 'arm_overlap_rollback', 'self.saved_state')
add('s5i5', 3243, 'orelse', 'invalid_action', 'off_grid')
add('s5i5', 3277, 'orelse', 'invalid_action', 'no_click_target')
add('sb26', 1052, 'body', 'invalid_action', 'unsupported_action')
add('sb26', 1261, 'orelse', 'invalid_action', 'no_click_target')
# sc25 movement uses the engine's try_move_sprite; wrapped separately.
add('sc25', 2389, 'before', 'effect', 'resize_cast', 'self.cflwkfoggh')
add('sc25', 2400, 'body', 'collision', 'resize_wall_or_boundary', 'cflwkfoggh')
add('sc25', 2357, 'body', 'collision', 'projectile_hit', 'ncqygezsiu', 'jgjgrozmil')
add('sc25', 3420, 'before', 'invalid_action', 'invalid_click')
add('sk48', 1892, 'body', 'collision', 'wall_or_boundary', 'self.mwfajkguqx[active][-1]',
    attempt='(self.mwfajkguqx[active][-1].x + dx, self.mwfajkguqx[active][-1].y + dy, self.mwfajkguqx[active][-1].width, self.mwfajkguqx[active][-1].height)')
add('sk48', 1920, 'body', 'collision', 'blocked_chain', 'seg')
add('sk48', 1909, 'body', 'constraint', 'minimum_length', 'active')
add('sk48', 1917, 'orelse', 'constraint', 'no_rail', 'active')
add('sk48', 1326, 'body', 'invalid_action', 'unsupported_action')
add('sk48', 1415, 'orelse', 'invalid_action', 'no_click_target')
add('su15', 2408, 'before', 'collision', 'enemy_damage', 'vxxrprjnav', 'vgtpakybrc')
for line in (2383, 2386, 2388, 2390, 2393):
    add('su15', line, 'body', 'collision', 'boundary_clamp', 'vxxrprjnav')
for line in (2526, 2529, 2531, 2533, 2536):
    add('su15', line, 'body', 'collision', 'boundary_clamp', 'vgtpakybrc')
for line in (2644, 2660):
    add('tn36', line, 'body', 'collision', 'wall', 'self.block', 'wall')
for line in (2649, 2664, 2731):
    add('tn36', line, 'body', 'collision', 'lethal_barrier', 'self.block', 'barrier')
add('tn36', 2629, 'body', 'contact', 'matching_scale_barrier', 'self.block', 'barrier')
for line in (2639, 2656):
    add('tn36', line, 'body', 'constraint', 'dead_actor', 'self.block')
for line in (1754, 1772, 1790, 1808):
    add('tu93', line, 'orelse', 'collision', 'wall_or_boundary', 'player_spr',
        attempt='(grid_spr.x + xjsopnydkp, grid_spr.y + i, player_spr.width, player_spr.height)')
for line in (1641, 1646, 1651, 1656):
    add('tu93', line, 'body', 'collision', 'wall_or_boundary_bounce', 'teal_spr',
        attempt='(grid_spr.x + xjsopnydkp, grid_spr.y + i, teal_spr.width, teal_spr.height)')
for line in (1592, 1612, 1627, 1673):
    add('tu93', line, 'before', 'collision', 'ball_contact', 'sprite', 'hit_sprites', guard='bool(hit_sprites)')
add('vc33', 2619, 'body', 'constraint', 'pipe_limit', '[slot_a, slot_b]')
add('vc33', 2764, 'orelse', 'invalid_action', 'off_grid')
add('vc33', 2772, 'orelse', 'invalid_action', 'no_click_target')
add('vc33', 2784, 'orelse', 'constraint', 'bar_not_connected', 'hit')
add('wa30', 1543, 'orelse', 'collision', 'occupied_hazard_or_boundary', '[sprite, tgt]',
    attempt='(x, y, sprite.width, sprite.height)')
add('wa30', 1554, 'orelse', 'collision', 'occupied_hazard_or_boundary', 'sprite',
    attempt='(x, y, sprite.width, sprite.height)')

GAMES = 'ar25 cd82 dc22 ft09 ka59 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 su15 tn36 tr87 tu93 vc33 wa30'.split()
