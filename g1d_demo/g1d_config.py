"""Shared constants: poses, layout, slots, materials.

Pure data - torch tensors only, no Isaac imports.
"""

import math
import os
import torch

REPO = "/workspaces/isaaclab_arena"
G1D_USD = os.path.join(REPO, "g1d_demo", "assets", "g1d", "g1d.usda")
WAREHOUSE = os.path.join(REPO, "g1d_demo", "assets", "warehouse")

# GUI render decimation: physics 60 Hz, one rendered frame every
# RENDER_EVERY steps (headless never renders)
RENDER_EVERY = 6

WHEEL_JOINTS = ["Left_Wheel_Joint", "Right_Wheel_Joint"]
WHEEL_RADIUS = 0.0845
HALF_TRACK = 0.2031
DRIVE_SPEED = 0.40
HOME_ROOT_Z = 0.113  # WHEEL_JOINT_Z + WHEEL_RADIUS + hover

RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_ARM = [nm.replace("right_", "left_") for nm in RIGHT_ARM]
RIGHT_FINGERS = [
    f"right_hand_{nm}_joint" for nm in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
]
ARM_HOME = torch.tensor([0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0])
# GPU-calibrated top pinch: thumb hull at (2.22, 0, 0.91) and middle
# hull at (2.33, 0, 0.97) - a centered over-the-top squeeze, so the
# tote just rests on a plain board (no posts, no pockets)
# elbow 1.05 (was 0.9): probe-verified on the 595.99.02
# landscape - a more vertical touchdown halves the descent
# knock (16.5 -> 8.3 cm) and the close genuinely stalls
# deep fallback only: the snap radius normally fires at/before
# DOCK - it exists so a bad drape still attaches. Probe-calibrated
# (g1d_probe.py): jaw z ~0.92 at this pose, ~5 cm over the tote
ARM_GRASP = torch.tensor([-0.50, 0.0, 0.0, 0.80, 0.0, -0.90, 0.0])
# travel pose: the arm rides UP from the very start (user spec) -
# shoulder -1.45 keeps the hand at z ~1.2: at -1.25 the fingertips
# cleared the tote tops (0.93) by only 7-15 cm and a yawed pass
# swept tote_d off its post (found on the floor at (1.82, 0.93))
ARM_HIGH = torch.tensor([-1.45, 0.0, 0.0, 0.9, 0.0, -0.90, 0.0])
# over-slot gateway, PALM DOWN: wrist roll 0 / pitch -0.90 is the
# probe-flat jaw (thumb z == middle z at pitch -0.90). Elbow 0.85
# holds the jaw center at z ~1.02 - 15 cm above the tote center,
# so the snap fires from HERE; the glued tote bottom (0.89)
# clears the post top (0.805) by 8 cm - no press-down at all
ARM_HOVER = torch.tensor([-0.70, 0.0, 0.0, 0.85, 0.0, -0.90, 0.0])
# REDUCED descent: a small SHOULDER dip (not an elbow drop - the
# old 1.35->1.05 elbow ladder was an ASCENDING arc that plowed
# the jaw up through the snapped tote = the pickup jitter).
# Shoulder -0.55 lowers the jaw to z ~0.95, still 4 cm clear of
# the post with the glued tote; abort-on-attach stops mid-dip
ARM_DOCK = torch.tensor([-0.55, 0.0, 0.0, 0.85, 0.0, -0.90, 0.0])
# carry pose: arm leveled horizontal (平举), palm still down
ARM_LEVEL = torch.tensor([0.0, 0.1, 0.0, 0.25, 0.0, -0.90, 0.0])
# the LEFT arm stays chest-folded always: at arm zeros it points
# straight FORWARD at station height and rams the shelf face
LEFT_ARM_TUCK = torch.tensor([0.3, -0.1, 0.0, 1.8, 0.0, -1.0, 0.0])
FINGER_OPEN = torch.zeros(7)
FINGER_CLOSE = torch.tensor([0.9, 0.8, 0.8, 0.9, 0.8, 0.9, 0.8])
FINGER_CRADLE = 0.35  # fraction of FINGER_CLOSE for the carry wrap

# --- kinematic snap (vacuum) grasp -------------------------------------
# when the jaw center comes within SNAP_RADIUS of the tote, the
# tote teleports to SNAP_OFFSET in the right_hand_middle_1_link
# frame and is rigidly written there every physics step (no
# physical pinch at all - offset probe-calibrated on the hand rig).
# 0.25: generous on purpose - the drive-in drape and the extreme
# slots leave the jaw 0.15-0.20 m off the tote line, and the base
# physically cannot align closer (station-face nose grind). 0.30
# so the snap fires AT THE HOVER GATEWAY for every slot (dz 0.15
# + drape dy up to 0.25): the dip ladder in the demo then never
# runs - the box jumps to the palm straight from the gateway
# height, with no press-down before or after the snap (at 0.25
# the off-center slots only attached at dip depth, and the arm
# sat IN the pressed pose at the moment of the teleport)
SNAP_RADIUS = 0.36
# x along the middle-finger link: 0.30 floated the box a hand-
# length past the jaw, 0.08 still left it ahead of the fingertips
# (user: one box edge closer to the body) - -0.02 pulls it a full
# 0.12 m back so it rides against the palm, roughly one edge from
# the base nose; z -0.09 seats it just BELOW the fingertip line so
# the cradled fingers rest on the box instead of through it
SNAP_OFFSET = (-0.02, 0.0, -0.09)
# the snap point is ANCHORED to the palm link (not the finger):
# at attach the world target = the seed point above, shifted up /
# left / back in the BASE frame - finger curls and arm rotations
# then no longer displace the box relative to the palm. The hand/
# finger COLLIDERS are stripped at startup (strip_hand_colliders),
# so ANY height is contact-free - historically 0.06-0.16 straddled
# the fingertip hulls and the contact fight tumbled the box and
# jammed the base. LEFT 0.072 / BACK 0.108 per user tuning
# (0.5 + 0.1 edge left, 0.9 edge rearward)
SNAP_SHIFT_UP = 0.10
SNAP_SHIFT_LEFT = 0.072
SNAP_SHIFT_BACK = 0.108


# --- hall footprint ----------------------------------------------------
# The whole enclosure (wall runs, floor tiles, rack rows) derives
# from these; the wall runs assert on spans that are not a multiple
# of the 3 m panel, so whole/half HALL_SCALE values tile cleanly.
# 5.0 (45x30 m) fits the 12-table grid whose farthest divider sits
# at x -19.5 (table edge -20.4 < wall -22.5).
HALL_SCALE = 5.0
HALL_X = 4.8 * HALL_SCALE  # +-24: 48 m span tiles in 6/4/5/3 m runs
HALL_Y = 3.0 * HALL_SCALE  # +-15: 30 m span tiles in 6/3 m runs
RACK_Y = 2.4 * HALL_SCALE  # rack rows in the far north/south bands

# --- 12-table grid (mir_isaacLab divider pattern) ----------------------
# divider_x straight from the mir layout; its 2.5 m row separation
# is WIDENED to 4.0 m (rows at +-2.0) because our base is wider
# than mir's: with 0.9 m-deep tables the 2.5 m rows left a 0.7 m
# center corridor - the 0.56 m base could not even spin (0.86 m
# sweep) and every transit stalled nose-on into a table. At 4.0 m
# the corridor is 2.2 m and the robot has a +-0.8 m lane.
DIVIDER_X = (-8.5, -11.0, -13.75, -15.25, -18.0, -19.5)
ROW_DY = 2.0
TABLE_POS = tuple((float(x), float(dy)) for x in DIVIDER_X for dy in (-ROW_DY, ROW_DY))
TABLE_TOP = 0.78  # working surface height (arm poses calibrated ~0.78)
TABLE_SIZE = (0.9, 1.8, 0.06)  # LONG AXIS Y: the corridor-facing
# inner edge is a 0.9 m face; same-row gaps stay 1.6 m
# two slots per table on the INNER (corridor-facing) edge: dx along
# the table length, dy = distance from the table center TOWARD the
# corridor (sign applied per row in table_slot). Robot and objects
# both stay on the inner side only (user spec) - the base never
# enters the row bands, which also kills the nose-on table wedge
# the east-end standoffs kept hitting
SLOT_DXY = ((-0.22, 0.55), (0.22, 0.55))
SLOT_STANDOFF = 0.70  # base parks this far corridor-side of the slot
# (jaw reach 0.48 + margins: base edge keeps ~0.2 m clear of the
# tabletop face; the 0.48 east-end standoffs jammed the nose on it)
CORRIDOR_LANE = 0.3  # transit lane |y| (corridor floor is |y| < 1.1)
# half the tables stocked, the other half are delivery targets
# -> 12 pick+place tasks per cycle
STOCKED_TABLES = (0, 2, 4, 6, 8, 10)
# DRESSING tables (east workplace zone): no tasks, just goods on
# top - makes the hall read like a working floor (user spec)
DRESSING_TABLES = ((6.0, 3.0), (10.0, 3.2), (6.0, -3.0), (12.0, -3.4))
TOTE_SIZE = 0.12  # cuboid tote edge


def table_slot(table_idx: int, slot_idx: int):
    """World (x, y) of a slot on table `table_idx`. The slot dy is
    stored as a distance TOWARD the corridor; its sign follows the
    row so both rows get corridor-facing slots."""
    tx, ty = TABLE_POS[table_idx]
    dx, dy = SLOT_DXY[slot_idx]
    return (tx + dx, ty - math.copysign(dy, ty))


# --- object catalog ----------------------------------------------------
# name -> (usd path or None, shape spec, GRAB_UP, mass kg).
# shape spec: None (usd prop) | ("cube", (dx,dy,dz), rgb)
#            | ("sphere", diameter, rgb) | ("cyl", radius, height, rgb)
# GRAB_UP = how far above the SEED point the palm grabs (crate =
# its HANDLE high so the basket hangs below the palm)


def _prop(name):
    return os.path.join(WAREHOUSE, "Props", name + ".usd")


ARTVIP = os.path.join(REPO, "g1d_demo", "assets", "artvip")


def _av(*parts):
    return os.path.join(ARTVIP, *parts)


RED = (0.85, 0.15, 0.12)
GREEN = (0.2, 0.75, 0.25)
ORANGE = (0.95, 0.55, 0.1)
BLUE = (0.15, 0.4, 0.85)
WHITE = (0.92, 0.9, 0.86)

OBJECT_CATALOG = {
    # warehouse kit props
    "crate": (_prop("SM_CratePlastic_A_01"), None, 0.34, 0.5),
    "card_a": (_prop("SM_CardBoxA_01"), None, 0.22, 0.4),
    "card_b": (_prop("SM_CardBoxB_01"), None, 0.18, 0.3),
    "card_c": (_prop("SM_CardBoxC_01"), None, 0.26, 0.5),
    "card_d": (_prop("SM_CardBoxD_01"), None, 0.30, 0.6),
    # generated shapes (UniBot-V1 challenge task props)
    "cup_r": (None, ("cyl", 0.035, 0.10, RED), 0.09, 0.08),
    "cup_b": (None, ("cyl", 0.035, 0.10, BLUE), 0.09, 0.08),
    "apple": (None, ("sphere", 0.08, RED), 0.07, 0.10),
    "orange": (None, ("sphere", 0.075, ORANGE), 0.07, 0.09),
    "lime": (None, ("sphere", 0.065, GREEN), 0.06, 0.07),
    "book_r": (None, ("cube", (0.19, 0.13, 0.035), RED), 0.05, 0.30),
    "book_g": (None, ("cube", (0.19, 0.13, 0.035), GREEN), 0.05, 0.30),
    "towel_w": (None, ("cube", (0.28, 0.28, 0.025), WHITE), 0.05, 0.15),
    "battery": (None, ("cyl", 0.014, 0.05, (0.3, 0.3, 0.32)), 0.05, 0.03),
    "tote_y": (None, ("cube", (TOTE_SIZE, TOTE_SIZE, TOTE_SIZE), (0.95, 0.80, 0.20)), 0.10, 0.15),
    "tote_g": (None, ("cube", (TOTE_SIZE, TOTE_SIZE, TOTE_SIZE), GREEN), 0.10, 0.15),
    # ArtVIP (huggingface X-Humanoid/ArtVIP) articulated digital
    # twins - rigidified whole (single box collider) for the snap
    # grasp; the basket is the true "grab the handle" object
    "av_basket": (_av("basket_1", "model_basket_1.usda"), None, 0.25, 0.6),
    "av_toolbox": (_av("toolbox", "model.usda"), None, 0.15, 0.8),
    "av_trash": (_av("trash_can", "model.usd"), None, 0.35, 1.0),
    "av_pump": (_av("pump_bottle", "model.usd"), None, 0.18, 0.3),
    "av_carton": (_av("cardboard_box", "model.usda"), None, 0.15, 0.4),
    # SMALL ArtVIP props (a few cm - grasp whole, no tipping, real
    # textures): clock, stapler, notebook, matchbox
    "av_clock": (_av("clock001", "model_clock_001.usd"), None, 0.08, 0.15),
    "av_stapler": (_av("stapler", "model_stapler_1.usd"), None, 0.05, 0.10),
    "av_notebook": (_av("notebook", "model_notebook_0.usd"), None, 0.04, 0.20),
    "av_match": (_av("match", "model_match.usd"), None, 0.04, 0.05),
}
# The 6 stocked tables are THEMED after UniBot-V1 challenge tasks
# (ref: hnrobert-unibot-world-challenge ref/tasks-detail); each entry
# = (theme label, slot0 object, slot1 object)
# all pickable objects come from the KIT ASSET LIBRARY (user
# spec): the kit props carry MDL PBR textures so every picked
# object renders textured - the generated primitives render flat
# small ArtVIP props only (user spec: NOT the big boxes - the
# 0.5 m kit boxes tipped over at every approach and their drops
# bounced ~1 m off the slots)
# REVERTED to the small generated primitives: the small ArtVIP
# props (stapler/match) rigidify with correct bboxes but their
# physics bodies IGNORE write_root_pose_to_sim entirely (3 retry
# rounds, stuck at the stage origin - the RigidObject view resolves
# to a different body than the prim we write), and the trash_can
# accepts the teleport but its BoxCol falls through. Both stay in
# the catalog; blocked on body resolution (see memory notes)
TABLE_THEMES = (
    ("T01 挂杯子 HangCup", "cup_r", "cup_b"),
    ("T04 水果分类 FruitSort", "apple", "orange"),
    ("T08 整理书架 Bookshelf", "book_r", "book_g"),
    ("T09 工具箱收纳 ToolStorage", "crate", "card_d"),
    ("T02 水果收集 FruitCollect", "crate", "lime"),
    ("T16 叠毛巾 TowelFold", "towel_w", "tote_y"),
)
OBJECT_ORDER = tuple(obj for _theme, o0, o1 in TABLE_THEMES for obj in (o0, o1))
OBJECT_NAMES = tuple(f"obj_{i:02d}" for i in range(len(OBJECT_ORDER)))

# per-type surface finish (metallic, roughness) for the generated
# primitives - flat toy colors read as fake; metals shine, paper
# and fabric stay matte (kit USD props already carry MDL PBR)
OBJECT_FINISH = {
    "cup_r": (0.85, 0.25),  # enamel mug
    "cup_b": (0.85, 0.25),
    "apple": (0.0, 0.30),  # glossy fruit skin
    "orange": (0.0, 0.30),
    "lime": (0.0, 0.30),
    "book_r": (0.0, 0.80),  # matte paper cover
    "book_g": (0.0, 0.80),
    "towel_w": (0.0, 0.95),  # fabric
    "battery": (0.75, 0.35),  # steel shell
    "tote_y": (0.0, 0.50),  # plastic bin
    "tote_g": (0.0, 0.50),
}
OBJECT_FINISH_DEFAULT = (0.0, 0.5)

# per-type TEXTURE for the generated primitives (png, roughness,
# metallic) - g1d_make_object_textures.py output; a plain color
# renders flat, a texture reads as a real material (user spec)
OBJECT_TEXTURE = {
    "cup_r": ("metal.png", 0.25, 0.85),
    "cup_b": ("metal.png", 0.25, 0.85),
    "apple": ("apple_red.png", 0.30, 0.0),
    "orange": ("orange.png", 0.30, 0.0),
    "lime": ("lime_green.png", 0.30, 0.0),
    "book_r": ("paper.png", 0.80, 0.0),
    "book_g": ("paper.png", 0.80, 0.0),
    "towel_w": ("fabric.png", 0.95, 0.0),
    "battery": ("metal.png", 0.35, 0.75),
    "tote_y": ("plastic_yellow.png", 0.50, 0.0),
    "tote_g": ("plastic_blue.png", 0.50, 0.0),
}
# parallel rack rows flanking both sides of the work aisle: the
# hall is 30 m wide (walls +-15), tables end at +-2.9, so three
# rows per side at +-5/+-8/+-12 tile cleanly with walk gaps
RACK_ROWS = (5.0, 8.0, 12.0)
