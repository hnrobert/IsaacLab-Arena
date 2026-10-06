"""G1D warehouse pick-and-place demo inside IsaacLab-Arena's Isaac Lab.

A Mir-style digital-twin warehouse built from the Simple_Warehouse
asset kit: perimeter walls, pallet racking flanking the aisle, floor
decals, pallets and card boxes. The wheeled G1D shuttles along the
aisle and services every tote on the replenishment station in a loop.

The grasp is CLOSED-LOOP: simulation ground truth (hand link poses +
tote pose) feeds alignment trims before the fingers close. Feedback
signals, all printed per attempt:

  * finger pad geometry vs the tote (pre-close line)
  * finger stall (a hit stops the curl short of the command)
  * tote world z after the lift (ground-truth success check)

Run inside the compose container with the GUI on the host desktop:

    ./run.sh            # tmux 'g1d_arena', GUI on DISPLAY=:1
    ./run.sh headless   # smoke run, no window
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# IsaacLab 3.0: a GUI window requires the 'kit' visualizer explicitly -
# headless=False alone no longer opens one
app_launcher = AppLauncher(
    headless=args_cli.headless,
    **({} if args_cli.headless else {"visualizer": "kit"}),
)
simulation_app = app_launcher.app

"""Now the sim is alive."""

import carb
import math
import numpy as np
import omni.usd
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

import os

REPO = "/workspaces/isaaclab_arena"
G1D_USD = os.path.join(REPO, "g1d_demo", "assets", "g1d", "g1d.usda")
WAREHOUSE = os.path.join(REPO, "g1d_demo", "assets", "warehouse")


def n(t):
    """Device buffer (torch OR warp array) -> numpy."""
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return t.numpy()  # warp array (Fabric/GPU pipeline)


WHEEL_JOINTS = ["Left_Wheel_Joint", "Right_Wheel_Joint"]
WHEEL_RADIUS = 0.0845
HALF_TRACK = 0.2031
DRIVE_SPEED = 0.25
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

# ONE proven pick slot on the centerline (the jaw zone is only ~4 cm
# wide); b/c are dressing on the racking. The slot refills each cycle
# with the next tote in the rotation.
PICK_TARGETS = (
    ("tote_a", 2.27, 0.0, (0.9, 0.45, 0.1)),
    ("tote_b", 2.27, 0.0, (0.1, 0.6, 0.9)),
    ("tote_c", 2.27, 0.0, (0.2, 0.8, 0.3)),
)

# delivered totes are tidied onto these pad slots after the release.
# They MUST sit outside the base's swept width: drive_to would
# otherwise bulldoze them and shunt the base sideways
PAD_SLOTS = {
    "tote_a": (-0.90, 0.55),
    "tote_b": (-1.10, 0.55),
    "tote_c": (-1.00, -0.55),
}
PAD_STOP_X = -0.85

RACK_BEAM = (0.95, 0.45, 0.1)     # orange beams - Mir livery accent
STATION_BODY = (0.55, 0.5, 0.42)  # wood-tone cabinet
PAD_COLOR = (0.1, 0.35, 0.8)


def prop(name):
    """Absolute container path of a Simple_Warehouse prop."""
    return os.path.join(WAREHOUSE, "Props", name + ".usd")


def make_tote_cfg(name: str, pos, color):
    """One small rigid tote spawned at `pos` (x, y, z)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=(TOTE_SIZE, TOTE_SIZE, TOTE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


_PROP_ROT = {}


def usd_prop_cfg(usd_path, pos, rot_z=0.0):
    """A static warehouse prop spawned from the Simple_Warehouse kit.

    UsdFileCfg has no orientation kwarg here, so rotations are applied
    post-spawn by rotate_props() (USD surgery on the live stage).
    """
    key = f"{os.path.basename(usd_path)[:-4]}_{abs(hash(pos)) % 100000}"
    if rot_z:
        _PROP_ROT[key] = rot_z
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + key,
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def rotate_props(env_ns="/World/envs/env_0"):
    """Apply the recorded prop yaw rotations on the live stage."""
    if not _PROP_ROT:
        return
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    for key, deg in _PROP_ROT.items():
        prim = stage.GetPrimAtPath(f"{env_ns}/{key}")
        if not prim:
            continue
        xfa = UsdGeom.Xformable(prim)
        t_op = xfa.GetTranslateOp()
        t = t_op.Get() if t_op else Gf.Vec3d(0.0, 0.0, 0.0)
        xfa.ClearXformOpOrder()
        xfa.AddTranslateOp().Set(t)
        xfa.AddRotateZOp().Set(deg)
    print(f"rotated {len(_PROP_ROT)} props", flush=True)


def build_warehouse_props():
    """All static Simple_Warehouse dressing, as spawn cfgs."""
    props = []
    # concrete-look floor tiles down the aisle (visual; the ground
    # plane below provides collision)
    for x in (-3.0, 1.5):
        props.append(usd_prop_cfg(prop("SM_floor02"), (x, 0.0, 0.002)))
    # aisle lane stripes
    for x in (-2.0, 2.0):
        for y in (-0.62, 0.62):
            props.append(usd_prop_cfg(
                prop("SM_FloorDecal_StripeFull_4m"), (x, y, 0.004)))
    # perimeter walls: rotate the 3 m panels so the long axis runs x
    for y in (-3.2, 3.2):
        for x in (-3.0, 0.0, 3.0):
            props.append(usd_prop_cfg(
                prop("SM_WallA_3M"), (x, y, 0.0), rot_z=90.0))
    # end walls (default orientation: long axis along y)
    for x in (-4.6, 4.6):
        props.append(usd_prop_cfg(prop("SM_WallA_3M"), (x, 0.0, 0.0)))
    # pallet racking flanking the aisle: 4 m shelf runs at two levels,
    # upright frames at the ends, goods piled on the beams
    for y_r in (1.6, -1.6):
        for x_c in (-1.6, 2.0):
            for z_lvl in (0.35, 1.45):
                props.append(usd_prop_cfg(
                    prop("SM_RackShelf_01"), (x_c, y_r, z_lvl)))
            for x_e in (x_c - 2.0, x_c + 2.0):
                props.append(usd_prop_cfg(
                    prop("SM_RackFrame_03"), (x_e, y_r, 1.5)))
            props.append(usd_prop_cfg(
                prop("SM_RackPile_03"), (x_c - 1.2, y_r, 0.40)))
            props.append(usd_prop_cfg(
                prop("SM_RackPile_04"), (x_c + 0.9, y_r, 0.42)))
    # floor goods near the aisle ends
    props.append(usd_prop_cfg(prop("SM_PaletteA_02"), (3.3, -1.2, 0.0)))
    props.append(usd_prop_cfg(
        prop("SM_CardBoxA_01"), (3.3, -1.2, 0.34)))
    props.append(usd_prop_cfg(
        prop("SM_CardBoxB_01"), (-3.3, 1.2, 0.0)))
    props.append(usd_prop_cfg(
        prop("SM_CratePlastic_A_01"), (-3.3, -1.2, 0.0)))
    # wall-mounted gear + traffic furniture
    props.append(usd_prop_cfg(prop("SM_FuseBox_01"), (4.45, 1.9, 0.0)))
    props.append(usd_prop_cfg(prop("S_TrafficCone"), (-0.4, -0.85, 0.0)))
    props.append(usd_prop_cfg(prop("S_TrafficCone"), (-1.6, 0.85, 0.0)))
    props.append(usd_prop_cfg(
        prop("S_AisleSign"), (1.2, -1.1, 0.0), rot_z=180.0))
    return props


@configclass
class G1DSceneCfg(InteractiveSceneCfg):
    """Warehouse aisle: kit props, station, totes, G1D."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    # --- replenishment station (the pick source) ---
    station = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Station",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.9, STATION_TOP),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=STATION_BODY
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(STATION_X, 0.0, STATION_TOP / 2),
        ),
    )
    station_beam = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StationBeam",
        spawn=sim_utils.CuboidCfg(
            size=(0.52, 0.92, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=RACK_BEAM
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(STATION_X, 0.0, STATION_TOP - 0.02),
        ),
    )
    # cantilevered pick board: the finger pads only reach x ~2.19 and
    # the station face starts at 2.20, so the totes MUST sit on an
    # overhang above the AGV nose (classic Mir edge pick point)
    pick_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PickBoard",
        spawn=sim_utils.CuboidCfg(
            size=(0.24, 0.5, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=STATION_BODY),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, 0.0, 0.75)))
    # small retaining pockets on the pick board: the graded close
    # still nudges the tote; back + side walls keep it on its slot
    pocketBB = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PocketBB",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.16, 0.08), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RACK_BEAM)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.345, 0.0, 0.81)))
    pocketYB = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PocketYB",
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.02, 0.08), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RACK_BEAM)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, 0.072, 0.81)))
    pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pad",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.4, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=PAD_COLOR
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PAD_POS),
    )

    # --- pick targets on the pick board ---
    tote_a = make_tote_cfg("tote_a", (2.27, 0.0, TOTE_Z),
                           PICK_TARGETS[0][3])
    # b/c rest on the racking as dressing
    tote_b = make_tote_cfg("tote_b", (-2.8, 1.6, 0.85),
                           PICK_TARGETS[1][3])
    tote_c = make_tote_cfg("tote_c", (-0.6, -1.6, 0.85),
                           PICK_TARGETS[2][3])

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=G1D_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, HOME_ROOT_Z)),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=WHEEL_JOINTS,
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=150.0,
            ),
            "body": ImplicitActuatorCfg(
                # exclude the wheels: an overlapping ".*" would give
                # them the 800-stiffness position drive and brake the
                # base
                joint_names_expr=[
                    "^(?!Left_Wheel_Joint$|Right_Wheel_Joint$).*$"
                ],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )


# attach the warehouse dressing as numbered fields (configclass turns
# every class attribute into an entity, so inject them dynamically)
for _i, _cfg in enumerate(build_warehouse_props()):
    setattr(G1DSceneCfg, f"wh_prop_{_i:02d}", _cfg)


def select_physx_variant(robot_prim_path: str) -> None:
    """Pick the 'physx' variant of the asset's Physics variantSet."""
    stage = omni.usd.get_context().get_stage()
    variant = stage.GetPrimAtPath(robot_prim_path).GetVariantSet("Physics")
    if variant.GetVariantNames():
        variant.SetVariantSelection("physx")
        print("physics variant: physx selected", flush=True)


def disable_agv_shell_collision() -> bool:
    """Deactivate the AGV shell collision (steals ground contact)."""
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() == "AGV_link_1":
            prim.SetActive(False)
            print(f"shell collision off: {prim.GetPath()}", flush=True)
            return True
    return False


def add_grip_material(env_ns="/World/envs/env_0") -> None:
    """High-friction physics material on fingertips + totes.

    The pinch wraps the tote but default friction (~0.5) strips it
    during any base motion; rubber-ish 1.6/1.4 makes the wrap hold.
    """
    from pxr import UsdShade, UsdPhysics as Phys

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(
        stage, env_ns + "/GripMaterial")
    Phys.MaterialAPI.Apply(mat.GetPrim())
    pa = Phys.MaterialAPI(mat.GetPrim())
    pa.CreateStaticFrictionAttr(1.6)
    pa.CreateDynamicFrictionAttr(1.4)
    pa.CreateRestitutionAttr(0.0)
    bound = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        bind = (nm in ("tote_a", "tote_b", "tote_c")
                or (nm in ("right_hand_thumb_2_link_1",
                           "right_hand_middle_1_link_1",
                           "right_hand_index_1_link_1")))
        if bind:
            b = UsdShade.MaterialBindingAPI.Apply(prim)
            b.Bind(mat, UsdShade.Tokens.weakerThanDescendants,
                   "physics")
            bound += 1
    print(f"grip material bound to {bound} prims", flush=True)


def yaw_of(quat) -> float:
    """Yaw (rad) from a single xyzw quaternion.

    NOTE: the cross term is w*z (not x*w!) - the wrong pairing reads a
    constant ~0 for every pure-yaw rotation and silently kills ALL
    heading feedback.
    """
    x, y, z, w = (
        quat.tolist() if hasattr(quat, "tolist") else list(quat)
    )
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class G1DDriver:
    """Closed-loop driver over the articulation tensor API."""

    HAND_LINKS = ("right_hand_thumb_2_link",
                  "right_hand_index_1_link",
                  "right_hand_middle_1_link")

    def __init__(self, robot: Articulation, sim: SimulationContext,
                 totes=None):
        self.robot = robot
        self.sim = sim
        self.device = robot.device
        names = self.robot.joint_names
        self.idx = {nm: i for i, nm in enumerate(names)}
        self.arm_ids = torch.tensor(
            [self.idx[nm] for nm in RIGHT_ARM], device=robot.device)
        self.left_arm_ids = torch.tensor(
            [self.idx[nm] for nm in LEFT_ARM], device=robot.device)
        self.finger_ids = torch.tensor(
            [self.idx[nm] for nm in RIGHT_FINGERS], device=robot.device)
        self.left_wheel = self.idx[WHEEL_JOINTS[0]]
        self.right_wheel = self.idx[WHEEL_JOINTS[1]]
        self.n_joints = len(names)
        # rigid-object data buffers ONLY refresh on update(); skipping
        # it made every tote print a stale snapshot (phantom "empty")
        self.totes = list(totes) if totes else []
        self._last_right_pose = ARM_HOME.tolist()

    # ---- outputs ----

    def _zeros(self) -> torch.Tensor:
        return torch.zeros((1, self.n_joints), device=self.device)

    def _baseline(self) -> torch.Tensor:
        """Position-command baseline: LEFT arm always chest-folded."""
        cmd = self._zeros()
        cmd[0, self.left_arm_ids] = LEFT_ARM_TUCK.to(self.device)
        return cmd

    def arm(self, pose) -> None:
        cmd = self._baseline()
        cmd[0, self.arm_ids] = torch.as_tensor(
            pose, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(cmd)
        self._last_right_pose = list(pose)

    def fingers(self, pose) -> None:
        cmd = self._baseline()
        cmd[0, self.finger_ids] = pose.to(self.device)
        cmd[0, self.arm_ids] = torch.as_tensor(
            self._last_right_pose, dtype=torch.float32,
            device=self.device)
        self.robot.set_joint_position_target(cmd)

    def wheels(self, left: float, right: float) -> None:
        """+v = +x forward; LEFT command negated (mirrored URDF axes)."""
        cmd = self._zeros()
        cmd[0, self.left_wheel] = -left
        cmd[0, self.right_wheel] = right
        self.robot.set_joint_velocity_target(cmd)

    # ---- state ----

    def pose(self):
        pos = n(self.robot.data.root_pos_w)[0]
        yaw = yaw_of(n(self.robot.data.root_quat_w)[0])
        return float(pos[0]), float(pos[1]), yaw

    def finger_max(self) -> float:
        pos = n(self.robot.data.joint_pos)[0]
        return max(
            abs(pos[self.idx[nm]].item()) for nm in RIGHT_FINGERS
        )

    def hand_center(self) -> np.ndarray:
        """Grasp center: thumb tip vs index/middle pair midpoint."""
        pos = n(self.robot.data.body_pos_w)[0]
        names = self.robot.data.body_names
        thumb = pos[names.index(self.HAND_LINKS[0])]
        pair = (pos[names.index(self.HAND_LINKS[1])]
                + pos[names.index(self.HAND_LINKS[2])]) / 2.0
        return (thumb + pair) / 2.0

    # ---- primitives ----

    def _refresh(self) -> None:
        self.robot.update(self.sim.get_physics_dt())
        for t in self.totes:
            t.update(self.sim.get_physics_dt())

    def run_seconds(self, seconds: float, every_step=None) -> None:
        steps = int(seconds / self.sim.get_physics_dt())
        for _ in range(steps):
            if every_step is not None:
                every_step()
            self.robot.write_data_to_sim()
            self.sim.step()
            self._refresh()

    def hold_arm(self, pose, seconds: float,
                 fingers: list = None) -> None:
        """Hold one right-arm pose (left arm stays chest-folded).

        fingers=None commands the jaw OPEN: during descents the
        fingers MUST be actively driven open - freezing them at
        measured positions lets contact half-close the jaw mid-
        descent and the arm plows the tote 30 cm off its slot
        (d13/d14). Pass an explicit list for other targets.
        """
        cmd = self._baseline()
        cmd[0, self.arm_ids] = torch.as_tensor(
            pose, dtype=torch.float32, device=self.device)
        if fingers is not None:
            cmd[0, self.finger_ids] = torch.as_tensor(
                fingers, dtype=torch.float32, device=self.device)
        self._last_right_pose = list(pose)
        self.run_seconds(
            seconds,
            every_step=lambda: self.robot.set_joint_position_target(cmd),
        )

    def move_arm_slow(self, pose, step=0.05, hold=0.15,
                      timeout=40.0):
        """Ramped arm move: 0.05 rad steps from the CURRENT joints.

        Step commands with the 800-stiffness drive snap the arm in
        ~0.2 s (too fast to look at and too fast for contact), so
        every grasp-phase transition goes through this.
        """
        pose = list(pose)
        deadline = timeout / self.sim.get_physics_dt()
        waited = 0
        while waited < deadline:
            cur = [n(self.robot.data.joint_pos)[0][
                self.idx[nm]].item() for nm in RIGHT_ARM]
            deltas = [t - c for t, c in zip(pose, cur)]
            if max(abs(d) for d in deltas) < 0.03:
                self.hold_arm(pose, 0.1)
                return True
            nxt = [c + max(-step, min(step, d))
                   for c, d in zip(cur, deltas)]
            self.hold_arm(nxt, hold)
            waited += int(hold / self.sim.get_physics_dt())
        return False

    def arm_to(self, pose, tolerance=0.3, timeout=12.0) -> bool:
        deadline = timeout / self.sim.get_physics_dt()
        waited = 0
        while waited < deadline:
            self.arm(pose)
            self.robot.write_data_to_sim()
            self.sim.step()
            self._refresh()
            waited += 1
            current = n(self.robot.data.joint_pos)[0]
            err = max(
                abs(current[self.idx[nm]].item() - t)
                for nm, t in zip(RIGHT_ARM, list(pose))
            )
            if err < tolerance:
                return True
        return False

    def fingers_to(self, pose, tolerance=0.4, timeout=8.0) -> bool:
        deadline = timeout / self.sim.get_physics_dt()
        waited = 0
        while waited < deadline:
            self.fingers(pose)
            self.robot.write_data_to_sim()
            self.sim.step()
            self._refresh()
            waited += 1
            current = n(self.robot.data.joint_pos)[0]
            plist = (pose.detach().cpu().tolist()
                     if torch.is_tensor(pose) else list(pose))
            err = max(
                abs(current[self.idx[nm]].item() - t)
                for nm, t in zip(RIGHT_FINGERS, plist)
            )
            if err < tolerance:
                return True
        return False

    def holding(self) -> bool:
        return 0.30 < self.finger_max() < 0.78

    # ---- base motion ----

    def shunt_lateral(self, line_y: float = 0.0,
                      timeout: float = 15.0, tol: float = 0.08) -> bool:
        """Translate sideways onto the y=line_y rail (endgame align)."""
        _, y, _ = self.pose()
        dy = y - line_y
        if abs(dy) < tol:
            return True
        target_yaw = -0.6 if dy > 0 else 0.6
        if not self.spin_to(target_yaw, timeout=10.0):
            return False
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        while waited < deadline:
            _, y, _ = self.pose()
            if abs(y - line_y) < 0.04:
                break
            self.wheels(0.12, 0.12)
            self.robot.write_data_to_sim()
            self.sim.step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return self.spin_to(0.0, timeout=10.0)

    def drive_to(
        self, target_x: float, target_y: float = 0.0,
        timeout: float = 60.0, tol: float = 0.06,
        speed: float = DRIVE_SPEED, spin_first: bool = True,
    ) -> bool:
        """Cruise along the aisle toward (target_x, target_y), 2D-steered.

        spin_first=False skips the in-place align spin - rotating with
        a loaded arm strips the grip, so carrying drives never spin.
        """
        if spin_first and not self.spin_to(0.0, timeout=20.0):
            return False
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        shunts = 0
        ref_x, ref_t = None, 0
        while waited < deadline:
            x, y, yaw = self.pose()
            dx = target_x - x
            dy_line = y - target_y
            if ref_x is None:
                ref_x = x
            # stall detector: grinding against an edge (seen: nose
            # jammed on the station wall at dx 0.077 forever) -> free
            # the line with a shunt, or accept a loose arrival.
            # Compare over an 8 s WINDOW - at crawl speed the base
            # moves ~1 mm/step, so per-step diffs always look stuck.
            if (waited - ref_t) >= int(8.0 / dt):
                if abs(x - ref_x) < 0.01:
                    if shunts < 2:
                        shunts += 1
                        print(f"drive stalled x={x:.3f} y={y:+.3f} "
                              f"-> shunt", flush=True)
                        self.shunt_lateral(target_y)
                    else:
                        self.wheels(0.0, 0.0)
                        print(f"arrived x={x:.3f} y={y:.3f} "
                              f"(stalled-loose) "
                              f"(target {target_x:.3f})", flush=True)
                        return True
                    ref_x, ref_t = x, waited
                else:
                    ref_x, ref_t = x, waited
            if abs(dx) < tol and abs(yaw) < 0.25:
                if abs(dy_line) >= 0.08:
                    if shunts < 2:
                        shunts += 1
                        self.shunt_lateral(target_y)
                        continue
                    self.wheels(0.0, 0.0)
                    print(f"arrived x={x:.3f} y={y:.3f} (loose) "
                          f"(target {target_x:.3f})", flush=True)
                    return True
                self.wheels(0.0, 0.0)
                print(f"arrived x={x:.3f} y={y:.3f} "
                      f"(target {target_x:.3f})", flush=True)
                return True
            v = max(-speed, min(speed, 0.7 * dx))
            if abs(v) < 0.06:
                v = 0.06 if dx > 0 else -0.06
            # calm line tracker: reversing legs drift sideways (casters)
            desired = max(-0.35, min(0.35, math.atan2(-dy_line, 2.5)))
            if dx < 0:
                desired = -desired
            err_yaw = (desired - yaw + 3.14159265) % (2 * 3.14159265) \
                - 3.14159265
            turn = max(-0.35, min(0.35, 1.2 * err_yaw))
            self.wheels(
                (v - turn * HALF_TRACK) / WHEEL_RADIUS,
                (v + turn * HALF_TRACK) / WHEEL_RADIUS,
            )
            if waited % int(2.0 / dt) == 0:
                print(f"drive: x={x:.3f} y={y:+.3f} dx={dx:+.3f} "
                      f"yaw={yaw:+.3f}", flush=True)
            self.robot.write_data_to_sim()
            self.sim.step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return False

    def back_straight(self, dist: float, speed: float = 0.05,
                      arm_hold=None, timeout: float = 25.0) -> float:
        """Straight-line reverse with a gentle yaw lock.

        Equal wheel velocities alone drift LATERALLY ~0.11 m over a
        0.35 m reverse (casters), and any strong steering correction
        swings the arm tip sideways on a ~0.5 m lever and strips the
        pinch. The tiny clamped yaw lock keeps the heading (and the
        arm) pointed without jerking the load.
        Optionally holds a full joint command (arm + fingers) every
        step. Returns the distance actually reversed.
        """
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        x0, _, yaw0 = self.pose()
        while waited < deadline:
            if (x0 - self.pose()[0]) >= dist:
                break
            _, _, yaw = self.pose()
            err = ((yaw - yaw0 + 3.14159265) % (2 * 3.14159265)
                   - 3.14159265)
            turn = max(-0.08, min(0.08, 0.8 * err))
            v = -speed
            self.wheels(
                (v - turn * HALF_TRACK) / WHEEL_RADIUS,
                (v + turn * HALF_TRACK) / WHEEL_RADIUS,
            )
            if arm_hold is not None:
                self.robot.set_joint_position_target(arm_hold)
            self.robot.write_data_to_sim()
            self.sim.step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return x0 - self.pose()[0]

    def spin_to(self, target_yaw: float, timeout: float = 20.0) -> bool:
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        while waited < deadline:
            _, _, yaw = self.pose()
            err = (yaw - target_yaw + 3.14159265) % (2 * 3.14159265) \
                - 3.14159265
            if abs(err) < 0.15:
                self.wheels(0.0, 0.0)
                return True
            rate = max(0.6, min(2.0, 1.5 * abs(err)))
            # +w in wheels() yaws the base DOWN; err>0 means yaw must
            # decrease toward the target (probe-verified sign)
            w = -rate if err > 0 else rate
            self.wheels(
                -w * HALF_TRACK / WHEEL_RADIUS,
                w * HALF_TRACK / WHEEL_RADIUS,
            )
            self.robot.write_data_to_sim()
            self.sim.step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return False


def reset_tote(tote, pos) -> None:
    """Teleport a tote back to its slot (instant in PhysX)."""
    try:
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        tote.write_root_pose_to_sim(
            torch.tensor([[pos[0], pos[1], pos[2],
                           1.0, 0.0, 0.0, 0.0]],
                         dtype=torch.float32, device=dev)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"tote reset failed: {exc}", flush=True)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"sim device: {device}", flush=True)
    sim = SimulationContext(SimulationCfg(device=device, dt=1 / 60))
    scene_cfg = G1DSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    rotate_props()
    add_grip_material()
    select_physx_variant("/World/envs/env_0/Robot")
    if not disable_agv_shell_collision():
        carb.log_warn("AGV_link_1 not found - shell collision stays on")

    sim.reset()
    print(f"joints: {len(scene['robot'].joint_names)}", flush=True)

    driver = G1DDriver(
        scene["robot"], sim,
        totes=[scene[nm] for nm in ("tote_a", "tote_b", "tote_c")])
    x0, y0, yaw0 = driver.pose()
    print(f"root after reset: x={x0:.3f} y={y0:.3f} yaw={yaw0:.3f}",
          flush=True)
    for _ in range(120):  # 2 s settle
        sim.step()
        scene["robot"].update(sim.get_physics_dt())
        for nm in ("tote_a", "tote_b", "tote_c"):
            scene[nm].update(sim.get_physics_dt())

    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[4.8, 3.6, 2.6], target=[1.2, 0.0, 0.8],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"camera setup skipped: {exc}", flush=True)

    # arm warm-up: settle the TUCK->HIGH->GRASP chain once so the
    # first real approach matches every later one
    driver.move_arm_slow(ARM_HIGH.tolist())
    driver.move_arm_slow(ARM_GRASP.tolist())
    driver.hold_arm(ARM_GRASP.tolist(), 1.0)
    driver.move_arm_slow(ARM_HIGH.tolist())
    driver.move_arm_slow(ARM_TUCK.tolist())
    driver.hold_arm(ARM_TUCK.tolist(), 0.5)

    slots = {nm: (x, y, TOTE_Z) for nm, x, y, _ in PICK_TARGETS}
    grasps_ok = 0
    grasps_total = 0
    carry_streak = 0
    carry_total = 0
    cycle = 0
    while simulation_app.is_running():
        cycle += 1
        print(f"=== cycle {cycle} ===", flush=True)

        order = (("tote_a",), ("tote_b",), ("tote_c",))[(cycle - 1) % 3]
        for name in order:
            tx, ty = PICK_TARGETS[0][1], PICK_TARGETS[0][2]
            tote = scene[name]
            slot = (tx, ty, TOTE_Z)
            reset_tote(tote, slot)
            driver.run_seconds(0.5)
            # tuck TIGHT before driving: a half-swung arm is exactly
            # the 0.45 m pole that rams the station face
            driver.move_arm_slow(ARM_TUCK.tolist())
            driver.hold_arm(ARM_TUCK.tolist(), 0.4)
            driver.fingers_to(FINGER_OPEN)

            standoff = min(tx - 0.40, STANDOFF_CLEAR)
            # two-phase approach: the jaw zone is only ~4 cm wide, so
            # align the line BEFORE the wall zone (the nose jams on
            # the station edge near the standoff and a jammed shunt
            # cannot recover y) - then creep in already centered.
            pre_wall = standoff - 0.25
            if not driver.drive_to(pre_wall, target_y=ty, tol=0.05):
                print(f"{name}: drive to pre-wall FAILED", flush=True)
                continue
            driver.shunt_lateral(ty, tol=0.03)
            if not driver.drive_to(standoff, target_y=ty, tol=0.02):
                print(f"{name}: drive to station FAILED", flush=True)
                continue

            if cycle == 1 or carry_streak == 0:
                # station warm-up: rehearse the whole descent chain
                # once ON the station apron, then re-seat the tote.
                # Warm-up'd attempts carried 4/4 (d7-d10); without it
                # the drape roll is a coin flip (~2/6). Running it
                # after every failure resets the streak attempts to
                # the deterministic configuration.
                driver.fingers_to(FINGER_OPEN)
                driver.move_arm_slow(ARM_HIGH.tolist())
                driver.move_arm_slow(ARM_GRASP.tolist(), step=0.05,
                                     hold=0.2)
                driver.hold_arm(ARM_GRASP.tolist(), 1.0)
                driver.move_arm_slow(ARM_HIGH.tolist())
                reset_tote(tote, slot)
                driver.run_seconds(0.4)

            # slow gateway up, then a fine-stepped descent
            driver.move_arm_slow(ARM_HIGH.tolist())
            driver.hold_arm(ARM_HIGH.tolist(), 0.5)
            driver.move_arm_slow(ARM_GRASP.tolist(), step=0.05,
                                 hold=0.2)
            driver.hold_arm(ARM_GRASP.tolist(), 1.0)
            pose = ARM_GRASP.tolist()

            # (lateral alignment: base y at arrival +-0.03 via the
            # two-phase approach; arm-side trims and base shunts both
            # re-roll the springy contact drape and made things
            # worse - the station warm-up above is the reliable
            # normalizer)
            # depth trim: seat the thumb tip 3 cm into the tote's
            # vertical band via pitch (z coupling is the strong axis);
            # iterate - the drape offsets exceed one clipped step
            # (seen: tips hovering 3-10 cm over the tote top)
            for _z in range(2):
                pos_v = n(driver.robot.data.body_pos_w)[0]
                names_v = driver.robot.data.body_names
                th_v = pos_v[names_v.index("right_hand_thumb_2_link")]
                dz = (TOTE_Z + 0.01) - th_v[2]
                if abs(dz) <= 0.02:
                    break
                dpt = float(np.clip(dz / 0.378, -0.20, 0.20))
                driver.hold_arm([pose[0] - dpt] + pose[1:], 0.6)
                pose[0] -= dpt

            # base-yaw gate: arm-vs-station contact during the
            # descent yaws the free-castering base (+0.51 rad seen),
            # shifts the whole jaw off the tote and can hook the hand
            # under the board edge - the reverse then jams with the
            # wheels spinning (d17 attempt 5). If the heading is
            # compromised, lift, re-align, and re-descend.
            for _yaw_try in range(2):
                _, _, byaw = driver.pose()
                if abs(byaw) < 0.20:
                    break
                print(f"  yaw gate: base yaw {byaw:+.2f} -> redo "
                      f"descent", flush=True)
                driver.move_arm_slow(ARM_HIGH.tolist(), step=0.06)
                driver.spin_to(0.0, timeout=10.0)
                driver.drive_to(standoff, target_y=ty, tol=0.03)
                driver.move_arm_slow(ARM_HIGH.tolist())
                driver.move_arm_slow(ARM_GRASP.tolist(), step=0.05,
                                     hold=0.2)
                driver.hold_arm(ARM_GRASP.tolist(), 1.0)
                pose = ARM_GRASP.tolist()

            # pre-close geometry: jaw span vs tote width, one line
            tp = n(tote.data.root_pos_w)[0]
            pos_b = n(driver.robot.data.body_pos_w)[0]
            names_b = driver.robot.data.body_names
            th = pos_b[names_b.index("right_hand_thumb_2_link")]
            pr = (pos_b[names_b.index("right_hand_index_1_link")]
                  + pos_b[names_b.index("right_hand_middle_1_link")]) / 2.0
            jp_d = n(driver.robot.data.joint_pos)[0]
            arm_act = [round(jp_d[driver.idx[nm]].item(), 2)
                       for nm in RIGHT_ARM]
            bx, by, byaw = driver.pose()
            print(f"  diag: base=({bx:.2f},{by:+.2f},{byaw:+.2f}) "
                  f"arm={arm_act}", flush=True)
            for _nm in ("right_hand_thumb_2_link",
                        "right_hand_middle_1_link",
                        "right_hand_index_1_link"):
                _q = pos_b[names_b.index(_nm)]
                print(f"  tip {_nm.replace('right_hand_', '')}: "
                      f"({_q[0]:.3f},{_q[1]:+.3f},{_q[2]:.3f})",
                      flush=True)
            print(
                f"pre-close {name}: thumb_y={th[1]:+.3f} "
                f"pair_y={pr[1]:+.3f} tote_y={tp[1]:+.3f} "
                f"span={abs(th[1] - pr[1]) * 100:.0f}cm",
                flush=True,
            )

            # graded slow close with a REAL-contact gate: a stall in
            # 0.30-0.78 alone is NOT proof - a FREE close reaches
            # ~0.37 at the 0.4 fraction and reads "HOLDING" with the
            # tote untouched (probe-verified phantom). A catch is
            # real only if the tote MOVED during the close (lifted,
            # nudged, tilted) or the stall sits well past the free-
            # close level (0.5+). No real catch at 0.4 = re-settle
            # the drape and re-descend once.
            caught = False
            tote_pre = n(tote.data.root_pos_w)[0][:3]
            for settle in range(2):
                if settle:
                    driver.move_arm_slow(ARM_HIGH.tolist(), step=0.06)
                    driver.hold_arm(ARM_HIGH.tolist(), 0.4)
                    driver.move_arm_slow(pose, step=0.05, hold=0.2)
                    driver.hold_arm(pose, 0.8)
                    tote_pre = n(tote.data.root_pos_w)[0][:3]
                for frac in (0.4, 0.6, 0.8):
                    close_cmd = driver._baseline()
                    close_cmd[0, driver.finger_ids] = (
                        FINGER_CLOSE * frac).to(driver.device)
                    close_cmd[0, driver.arm_ids] = torch.as_tensor(
                        pose, dtype=torch.float32, device=driver.device)
                    driver.run_seconds(
                        1.2,
                        every_step=lambda: (
                            driver.robot.set_joint_position_target(close_cmd)),
                    )
                    fm = driver.finger_max()
                    tp_c = n(tote.data.root_pos_w)[0][:3]
                    moved = float(np.linalg.norm(
                        tp_c[:2] - tote_pre[:2])) * 100
                    dz_c = abs(tp_c[2] - tote_pre[2]) * 100
                    pos_g = n(driver.robot.data.body_pos_w)[0]
                    nm_g = driver.robot.data.body_names
                    th_g = pos_g[nm_g.index("right_hand_thumb_2_link")]
                    pr_g = ((pos_g[nm_g.index("right_hand_index_1_link")]
                             + pos_g[nm_g.index(
                                 "right_hand_middle_1_link")]) / 2.0)
                    near = (abs(th_g[1] - tp_c[1]) < 0.08
                            and abs(pr_g[1] - tp_c[1]) < 0.10)
                    got = (0.30 < fm < 0.85
                           and (moved >= 1.0 or dz_c >= 0.5
                                or (fm >= 0.50 and near)))
                    print(f"  close {frac:.2f}: finger_max={fm:.2f} "
                          f"tote_moved={moved:.1f}cm dz={dz_c:.1f}cm "
                          f"near={int(near)} "
                          f"{'HOLDING' if got else '-'}", flush=True)
                    if got:
                        caught = True
                        break
                    if fm > 0.78:
                        # jaw closed on AIR (no stall): deeper targets
                        # just make a fist. Abandon the ladder and
                        # re-settle instead.
                        break
                if caught:
                    break
            driver._last_right_pose = list(pose)
            grasps_total += 1

            # firmer grip, then a slow lift to unweight the tote from
            # the board before any base motion
            grip_frac = 0.6 if caught else 0.7
            grip_cmd = driver._baseline()
            grip_cmd[0, driver.finger_ids] = (
                FINGER_CLOSE * grip_frac).to(driver.device)
            grip_pose = list(pose)
            grip_cmd[0, driver.arm_ids] = torch.as_tensor(
                grip_pose, dtype=torch.float32, device=driver.device)
            driver.run_seconds(
                1.0,
                every_step=lambda: (
                    driver.robot.set_joint_position_target(grip_cmd)),
            )
            if caught:
                # straight from catch to reverse: no seat press (it
                # can wedge the arm into a strut base->tote->board
                # and the wheels just spin - d16), no wiggle, no
                # lift. The catch state (tote hanging ~3 cm in the
                # jaw) is exactly what carried in d7-d10.
                pass

            # THE criterion: ONE continuous 0.35 reverse (equal-wheel,
            # yaw-locked) holding arm+fingers - the d7-d10 carry
            # mechanism. Legged reverse-with-re-seat tested WORSE:
            # stop-start dynamics break the static friction hold and
            # the re-seats never re-engage.
            tx0 = n(tote.data.root_pos_w)[0]
            bx0 = driver.pose()[0]
            carry_hold = driver._baseline()
            carry_hold[0, driver.arm_ids] = torch.as_tensor(
                pose, dtype=torch.float32, device=driver.device)
            carry_hold[0, driver.finger_ids] = (
                FINGER_CLOSE * grip_frac).to(driver.device)
            driver.back_straight(0.35, speed=0.05,
                                 arm_hold=carry_hold)
            tx1 = n(tote.data.root_pos_w)[0]
            bx1 = driver.pose()[0]
            carried = (tx0[0] - tx1[0]) > 0.15
            carry_total += 1
            carry_streak = carry_streak + 1 if carried else 0
            print(
                f"CARRY TEST {name}: base {bx0:.2f}->{bx1:.2f} "
                f"tote {tx0[0]:.2f}->{tx1[0]:.2f} "
                f"-> {'CARRIED' if carried else 'dropped'} "
                f"[streak {carry_streak}/5, {carry_total} tries]",
                flush=True,
            )
            got = carried

            # clear the aisle BEFORE returning: the drag leaves the
            # tote in the base's own path (~1.6-2.0 m) and the return
            # drive bulldozes it, shunting the base off-line - the
            # next approach then arrives y-off and only jaw-edges the
            # tote. Carried -> pad slot (delivered); dropped -> back
            # to the station slot for the next attempt.
            driver.fingers_to(FINGER_OPEN, tolerance=0.5, timeout=3.0)
            if carried:
                slot_x, slot_y = PAD_SLOTS[name]
                reset_tote(tote, (slot_x, slot_y,
                                  TOTE_SIZE / 2 + 0.005))
                grasps_ok += 1
                print(f"drag-delivered {name} (reset to pad)",
                      flush=True)
            else:
                reset_tote(tote, slot)
                print(f"re-seated {name} on the station", flush=True)
            driver.run_seconds(0.3)
            driver.arm_to(ARM_HIGH, tolerance=0.25, timeout=10.0)
            driver.arm_to(ARM_TUCK, tolerance=0.2, timeout=10.0)
            driver.hold_arm(ARM_TUCK.tolist(), 0.3)
            driver.drive_to(0.0)

        print(f"=== cycle {cycle}: {grasps_ok}/{grasps_total} grasps "
              f"lifted ===", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
