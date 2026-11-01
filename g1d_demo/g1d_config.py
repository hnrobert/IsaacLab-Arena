"""Shared constants: poses, layout, slots, materials.

Pure data - torch tensors only, no Isaac imports.
"""

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
SNAP_RADIUS = 0.30
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
TABLE_SIZE = (0.9, 1.8, 0.06)  # LONG AXIS Y: east face is narrow, the
# same-row gaps between tables are then 2.5-0.9 = 1.6 m wide and the
# base can stand/spin in them
# two slots per table, fully ON the tabletop near the east edge:
# (dx, dy) from table center. dx 0.15 keeps every object supported
# (the 0.42 edge slots tipped objects off onto the floor); the base
# parks with its nose between the table legs UNDER the top and the
# jaw (~0.48 ahead) lands exactly on the slot line
SLOT_DXY = ((0.15, 0.40), (0.15, -0.40))
SLOT_STANDOFF = 0.65  # base parks this far EAST of the slot x: the
# nose (0.30 half-length) stays CLEAR of the tabletop edge (slot
# dx 0.15 + table edge 0.45 -> 0.20 m margin) - the G1D body is
# too tall to poke under the top like an AGV (the 0.48 standoff
# rammed the slab edge and shoved objects off)
# diagonal entry/exit runway: approach legs START this far east of
# the slot. 4.2 (not 3.1): the FAR-row slots (|y| 2.4) need up to
# 2.7 m of lateral from the center lane - at the 0.7 rad correction
# cap that needs >3.2 m of runway plus the standoff approach
SLOT_RUNWAY = 4.2
CORRIDOR_LANE = 0.3  # transit lane |y| (corridor floor is |y| < 0.8)
# half the tables stocked, the other half are delivery targets
# -> 12 pick+place tasks per cycle
STOCKED_TABLES = (0, 2, 4, 6, 8, 10)
TOTE_SIZE = 0.12  # cuboid tote edge


def table_slot(table_idx: int, slot_idx: int):
    """World (x, y) of a slot on table `table_idx`."""
    tx, ty = TABLE_POS[table_idx]
    dx, dy = SLOT_DXY[slot_idx]
    return (tx + dx, ty + dy)


# --- object catalog ----------------------------------------------------
# name -> (usd path or None for a cuboid tote, tote color, GRAB_UP,
# mass kg). GRAB_UP = how far above the SEED point the palm grabs:
# for the crate it is the HANDLE (the basket hangs below the palm),
# for card boxes roughly the box top, for totes the tuned value.


def _prop(name):
    return os.path.join(WAREHOUSE, "Props", name + ".usd")


OBJECT_CATALOG = {
    "crate": (_prop("SM_CratePlastic_A_01"), None, 0.34, 0.5),
    "card_a": (_prop("SM_CardBoxA_01"), None, 0.22, 0.4),
    "card_b": (_prop("SM_CardBoxB_01"), None, 0.18, 0.3),
    "card_c": (_prop("SM_CardBoxC_01"), None, 0.26, 0.5),
    "card_d": (_prop("SM_CardBoxD_01"), None, 0.30, 0.6),
    "tote_y": (None, (0.95, 0.80, 0.20), 0.10, 0.15),
    "tote_g": (None, (0.2, 0.8, 0.3), 0.10, 0.15),
}
# which object sits on each of the 12 stocked slots (in task order)
OBJECT_ORDER = (
    "crate",
    "card_a",
    "tote_y",
    "card_b",
    "card_c",
    "tote_g",
    "card_d",
    "crate",
    "card_a",
    "card_b",
    "tote_y",
    "card_c",
)
OBJECT_NAMES = tuple(f"obj_{i:02d}" for i in range(len(OBJECT_ORDER)))
