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
DRIVE_SPEED = 0.30
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
    f"right_hand_{nm}_joint"
    for nm in ("thumb_0", "thumb_1", "thumb_2",
               "middle_0", "middle_1", "index_0", "index_1")
]
ARM_HOME = torch.tensor([0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0])
# GPU-calibrated top pinch: thumb hull at (2.22, 0, 0.91) and middle
# hull at (2.33, 0, 0.97) - a centered over-the-top squeeze, so the
# tote just rests on a plain board (no posts, no pockets)
ARM_GRASP = torch.tensor([-0.7, 0.0, 0.0, 0.9, -0.8, -0.5, 0.0])
# approach gateway: swinging straight from the tuck clips the station
# lip, so the arm first goes UP, then pitches down onto the slot
ARM_HIGH = torch.tensor([-1.25, 0.0, 0.0, 0.9, -0.8, -0.8, 0.0])
# travel pose (chest fold): at arm zeros the G1's arms point straight
# FORWARD at station height - the uncommanded LEFT arm alone rams the
# shelf face 0.43 m ahead of the base. Both arms fold for every drive.
ARM_TUCK = torch.tensor([0.3, 0.1, 0.0, 1.8, 0.0, -1.0, 0.0])
LEFT_ARM_TUCK = torch.tensor([0.3, -0.1, 0.0, 1.8, 0.0, -1.0, 0.0])
ARM_CARRY = torch.tensor([0.0, 0.1, 0.0, 1.2, 0.0, -0.7, 0.0])
ARM_RELEASE = torch.tensor([-0.6, 0.0, 0.0, 0.8, 0.0, -0.4, 0.0])
FINGER_OPEN = torch.zeros(7)
FINGER_CLOSE = torch.tensor([0.9, 0.8, 0.8, 0.9, 0.8, 0.9, 0.8])

# --- aisle layout ------------------------------------------------------
STATION_X = 2.45          # replenishment station center (near edge 2.20)
STATION_TOP = 0.80
PAD_POS = (-1.0, 0.0, 0.005)
TOTE_SIZE = 0.12          # fills the 11.7 cm thumb-to-middle gap at wrist -0.5
TOTE_Z = 0.77 + TOTE_SIZE / 2 + 0.005  # board top + half tote
STANDOFF_CLEAR = 1.84     # physics wall: AGV nose hits the 2.20 edge

# FIVE pick slots across the 0.9 m board (y +-0.45): every cycle
# services all five totes in sequence - multiple picks per visit.
PICK_TARGETS = (
    ("tote_a", 2.27, 0.00, (0.9, 0.45, 0.1)),
    ("tote_b", 2.27, 0.16, (0.1, 0.6, 0.9)),
    ("tote_c", 2.27, -0.16, (0.2, 0.8, 0.3)),
    ("tote_d", 2.27, 0.32, (0.8, 0.2, 0.6)),
    ("tote_e", 2.27, -0.32, (0.3, 0.7, 0.8)),
)
TOTE_NAMES = tuple(e[0] for e in PICK_TARGETS)

# delivered totes are tidied onto these pad slots after the release.
# They MUST sit outside the base's swept width: drive_to would
# otherwise bulldoze them and shunt the base sideways
PAD_SLOTS = {
    "tote_a": (-0.90, 0.55),
    "tote_b": (-1.10, 0.55),
    "tote_c": (-1.00, -0.55),
    "tote_d": (-1.20, -0.55),
    "tote_e": (-0.80, -0.55),
}
PAD_STOP_X = -0.85

RACK_BEAM = (0.95, 0.45, 0.1)     # orange beams - Mir livery accent
STATION_BODY = (0.55, 0.5, 0.42)  # wood-tone cabinet
PAD_COLOR = (0.1, 0.35, 0.8)
