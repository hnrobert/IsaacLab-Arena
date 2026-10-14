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
# physically cannot align closer (station-face nose grind); the
# hover gateway still clears vertically, so it cannot fire early
SNAP_RADIUS = 0.25
# x along the middle-finger link: 0.30 floated the box a hand-
# length past the jaw, 0.08 still left it ahead of the fingertips
# (user: one box edge closer to the body) - -0.02 pulls it a full
# 0.12 m back so it rides against the palm, roughly one edge from
# the base nose; z -0.09 seats it just BELOW the fingertip line so
# the cradled fingers rest on the box instead of through it
SNAP_OFFSET = (-0.02, 0.0, -0.09)
# the snap point is ANCHORED to the palm link (not the finger):
# at attach the world target = the seed point above, shifted
# +0.12 up (one box edge) and +0.06 toward the robot's LEFT (the
# palm-open direction) - finger curls and arm rotations then no
# longer displace the box relative to the palm
SNAP_SHIFT_UP = 0.12
SNAP_SHIFT_LEFT = 0.06

# --- aisle layout ------------------------------------------------------
STATION_X = 2.45  # replenishment station center (near edge 2.20)
STATION_TOP = 0.80
PAD_POS = (-1.0, 0.0, 0.005)
TOTE_SIZE = 0.12  # fills the 11.7 cm thumb-to-middle gap at wrist -0.5
TOTE_Z = 0.805 + TOTE_SIZE / 2 + 0.005  # post top + half tote
# (posts lift the totes 4 cm into free air: fingers close on
# the tote body only - no board-contact grasp chaos)
STANDOFF_CLEAR = 1.84  # physics wall: AGV nose hits the 2.20 edge

# FIVE pick slots across the 0.9 m board (y +-0.45): every cycle
# services all five totes in sequence - multiple picks per visit.
# (no red/orange anywhere: the beam + posts render their DEFAULT
# material, and tote_a stays clear of the old livery orange)
PICK_TARGETS = (
    ("tote_a", 2.27, 0.00, (0.95, 0.80, 0.20)),
    ("tote_b", 2.27, 0.24, (0.1, 0.6, 0.9)),
    ("tote_c", 2.27, -0.24, (0.2, 0.8, 0.3)),
    ("tote_d", 2.27, 0.48, (0.8, 0.2, 0.6)),
    ("tote_e", 2.27, -0.48, (0.3, 0.7, 0.8)),
)
TOTE_NAMES = tuple(e[0] for e in PICK_TARGETS)

# delivered totes are tidied onto these pad slots by the physical
# drop - the delivery aims the HELD box at the slot, so they sit
# outside the base's swept path on the return leg
PAD_SLOTS = {
    "tote_a": (-0.90, 0.55),
    "tote_b": (-1.10, 0.55),
    "tote_c": (-1.00, -0.55),
    "tote_d": (-1.20, -0.55),
    "tote_e": (-0.80, -0.55),
}
PAD_SPAN = ((-1.45, -0.55), (-0.70, 0.70))  # pad x/y extents

STATION_BODY = (0.55, 0.5, 0.42)  # wood-tone cabinet
PAD_COLOR = (0.1, 0.35, 0.8)
