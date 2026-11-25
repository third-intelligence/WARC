"""The mechanic categories that the hooked program of each game can log, before the review of review_mapping.py.

Only the names are used: they define which categories the occurrence detector tracks in a game. The text next to
a name describes it where the final mechanic list (data/mechanics.json) has a description.
"""
DESCRIPTIONS = {
'ar25': '''mirror_axis|mirror axis
piece|piece
reflection|reflection
selection|Selecting another piece or mirror axis moves the highlight from the old target to the new one.
step_bar|The bar of remaining steps shortens at each action.''',
'cd82': '''container_tilt|An action changes the orientation or the position of the container (a bucket).
energy_bar|The bar of remaining steps shortens at each action.
reservoir_recolor|Clicking a colour changes the colour inside the container (or the crane) to it.
selection_marker|The underline that marks the selected colour moves to the new selection.
stack_fill|Pouring from the container fills the target area with its colour.''',
'dc22': '''bridge|The red and pink pieces (the bridge) rotate about the dark centre.
budget_bar|The bar of remaining steps shortens at each action.
carried_object|carried object
color_tile_visibility|Pressing the blue button switches the tiles on the course between a checkered and a plain blue pattern.
course_gimmick|course gimmick
crane|crane
ghost|ghost
movable_floor|movable floor
player|The player moves with the arrow keys.''',
'ft09': '''border_flash|border flash
click_budget|The bar of remaining clicks on the bottom edge shortens at each action.
color_tile|The clicked tile changes colour.''',
'ka59': '''bomb|bomb
budget_bar|The bar of remaining steps shortens at each action.
cross_object|cross object
edge_indicator|When the player block touches a wall, a hole or another block, a thin white band appears on that side or leaves it.
explosion_blast|explosion blast
player|The selected block (white centre) moves with the arrow keys. When it bumps into a block that slides on, that block's move is pushed_block.
player_center_selection|Switching the controlled block changes the centre colour of the old and the new block.''',
'lp85': '''level_pips|level pips
life_bar|The bar of remaining steps shortens at each action.
tiles|tiles''',
'ls20': '''hud_flash|hud flash
lock_match|lock match
piece_rotation|When the player steps on a rotation cell, the key piece in the corner rotates, on the board and in its HUD copy.
player|The 5 x 5 player block moves one tile (5 cells) with the arrow keys.
steps_bar|The bar of remaining steps shortens at each action.''',
'm0r0': '''life_bar|The bar of remaining steps shortens at each action.
mirror_life_bar|The mirrored bar on the bottom edge shortens at each action.
mirror_player|Moving the player moves the mirrored player symmetrically.
player|The directly controlled player moves with the arrow keys.''',
'r11l': '''arm|arm
body|body
budget_bar|The bar of remaining steps shortens at each action.
connection_line|connection line
goal_flash|When the body of the robot reaches a goal, a white 7 x 7 frame appears over the goal until the body leaves.
paint|paint''',
're86': '''budget_bar|The bar of remaining steps shortens at each action.
marker_visibility|marker visibility
piece|The selected piece moves with the arrow keys; its shape can change.
piece_selection|Selecting another piece switches the selection marker.''',
's5i5': '''budget_bar|The bar of remaining steps shortens at each action.
chained_arm|chained arm
extending_arm|A click rotates the telescopic arm and extends or retracts it.
fixed_arm|fixed arm
moving_dot|moving dot''',
'sb26': '''background_flash|background flash
bracket|bracket
budget_bar|The bar of remaining steps shortens at each action.
checkbox_selection|The selection box (the cursor square) appears, moves or changes state.
nested_tile|nested tile
outline_frame|outline frame
reference_cursor|reference cursor
target_box|target box
tile|The clicked tile changes its appearance or colour, including moving into a slot.''',
'sc25': '''budget_bar|The bar of remaining steps shortens at each action.
door|door
energy_pack|energy pack
fireball|fireball
player|player
spell_slot_grid|The grid of spell slots changes.
switch|switch
teleport_indicator|teleport indicator''',
'sk48': '''block_flash|block flash
budget_bar|The bar of remaining steps shortens at each action.
color_block|Pushed by the skewer, the colour blocks move, stretch or change shape.
match_indicator|The indicator that shows whether the blocks match the target switches.
skewer|skewer
skewer_selection|skewer selection''',
'su15': '''block|block
block_flash|block flash
budget_bar|The bar of remaining steps shortens at each action.
goal_9x9|goal 9x9
moving_ufo|moving ufo
plus_cursor|plus cursor
reference_display|reference display
vacuum|vacuum''',
'tn36': '''barrier|barrier
budget_bar|The bar of remaining steps shortens at each action.
ghost_overlay|ghost overlay
instruction_button|The clicked instruction button switches its state.
player|player
reference|In level 2, selecting a program changes the reference pose (the example display), which then stays.
run_button|run button
selector_button|The clicked program selection button switches its state.
target_flash|target flash''',
'tr87': '''bottom_timer_bar|The bar of remaining steps shortens at each action.
selector_bracket_pair|The upper and the lower selection bracket move together.
slot1_glyph_cycle|The symbol in the selected slot is replaced in place by another.''',
'tu93': '''bottom_energy_bar|The bar of remaining steps shortens at each action.
enemy_ball|The enemy ball moves on its own; the dots on its face change colour, or it bursts.
player_block_on_path|The player block on the path moves and turns with the arrow keys.''',
'vc33': '''top_timer_bar|The bar on the top edge shortens at each action; it can also refill.
yellow_bird_blob|yellow bird blob''',
'wa30': '''bottom_timer_bar|The bar of remaining steps shortens at each action.
moving_green_block|The green block moves and turns with the arrow keys and carries a grabbed object with it.
yellow_button_toggle|yellow button toggle''',
}


def descriptions():
    return {game: dict(line.split('|', 1) for line in text.splitlines())
            for game, text in DESCRIPTIONS.items()}
