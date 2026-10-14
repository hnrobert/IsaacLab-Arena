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
# deep fallback only: the snap radius normally fires well before
# this pose is reached - it exists so a bad drape still attaches
ARM_GRASP = torch.tensor([-0.7, 0.0, 0.0, 1.05, -0.8, -0.5, 0.0])
# travel pose: the arm rides UP from the very start (user spec) -
# over the station lip, nothing to clip in the aisle
ARM_HIGH = torch.tensor([-1.25, 0.0, 0.0, 0.9, -0.8, -0.8, 0.0])
# over-slot gateway: arm swung out above the tote, hand high - the
# descent from here is a small elbow drop, not a shoulder swing
ARM_HOVER = torch.tensor([-0.70, 0.0, 0.0, 1.35, -0.8, -0.5, 0.0])
# REDUCED descent target: elbow dips 1.35 -> 1.18 only - the tote
# snaps to the hand from proximity, so the hand never needs to
# reach the tote, just come near it
ARM_DOCK = torch.tensor([-0.70, 0.0, 0.0, 1.18, -0.8, -0.5, 0.0])
# carry pose: arm leveled horizontal (平举) for the delivery walk
ARM_LEVEL = torch.tensor([0.0, 0.1, 0.0, 0.25, 0.0, -0.5, 0.0])
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
# 0.22: generous on purpose - the drive-in drape hangs the jaw
# 5-10 cm past off-center slots and a 0.13 radius never fired
# there (totes b-e all skipped); the hover gateway still clears
# it vertically (>0.24 above the tote top), so it cannot fire
# early on the swing-out
SNAP_RADIUS = 0.22
# x along the middle-finger link: 0.30 floated the box a hand-
# length past the jaw (user: too far from the palm) - 0.08 parks
# it right at the jaw opening, just ahead of the fingertip line;
# z -0.07 seats it slightly below the fingers, in the cradle
SNAP_OFFSET = (0.08, 0.0, -0.07)

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
