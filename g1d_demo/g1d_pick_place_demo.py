"""G1D warehouse pick-and-place demo inside IsaacLab-Arena's Isaac Lab.

A standalone Isaac Lab scene (no Arena RL stack): the wheeled Unitree G1D
(from the Wheeled-G1-Demo USD asset) drives from its home pose to a pickup
table, grasps a small tote with its three-finger hand, shuttles to a drop
pad, releases, and loops. Choreography mirrors the verified Wheeled-G1-Demo
driver: differential drive on the mirrored wheel axes, command-and-wait arm
poses, finger-stall grasp, staged lift.

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

# --- geometry (mirrors Wheeled-G1-Demo config/) ---
TABLE_POS = (2.45, 0.0, 0.365)  # cuboid CENTER (0.73 tall -> top at 0.73)
TOTE_START = (2.30, 0.0, 0.765 + 0.01)  # 0.07 cube, half over edge
PAD_POS = (-1.0, 0.0, 0.005)
STANDOFF_X = 1.80

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
RIGHT_FINGERS = [
    f"right_hand_{n}_joint"
    for n in ("thumb_0", "thumb_1", "thumb_2",
              "middle_0", "middle_1", "index_0", "index_1")
]
ARM_HOME = torch.tensor([0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0])
ARM_REACH = torch.tensor([-0.9, 0.1, 0.0, 1.5, 0.0, -0.8, 0.0])
ARM_CARRY = torch.tensor([-0.4, 0.1, 0.0, 0.7, 0.0, -0.5, 0.0])
ARM_RELEASE = torch.tensor([-0.6, 0.0, 0.0, 0.8, 0.0, -0.4, 0.0])
FINGER_OPEN = torch.zeros(7)
FINGER_CLOSE = torch.tensor([0.9, 0.8, 0.8, 0.9, 0.8, 0.9, 0.8])

RACK_COLOR = (0.35, 0.38, 0.42)
TABLE_COLOR = (0.55, 0.5, 0.42)
TOTE_COLOR = (0.9, 0.45, 0.1)
PAD_COLOR = (0.1, 0.35, 0.8)


@configclass
class G1DSceneCfg(InteractiveSceneCfg):
    """Warehouse scene: ground, light, racks, table, tote, pad, G1D."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    # static props (collision-only cuboids; no rigid body = immovable)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.73),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TABLE_COLOR
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )
    pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pad",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=PAD_COLOR
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PAD_POS),
    )
    racks = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Racks",
        spawn=sim_utils.CuboidCfg(
            size=(4.5, 0.5, 1.6),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=RACK_COLOR
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.4, 0.8)),
    )
    racks2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Racks2",
        spawn=sim_utils.CuboidCfg(
            size=(4.5, 0.5, 1.6),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=RACK_COLOR
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.4, 0.8)),
    )

    tote = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tote",
        spawn=sim_utils.CuboidCfg(
            size=(0.07, 0.07, 0.07),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TOTE_COLOR
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TOTE_START),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=G1D_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, HOME_ROOT_Z)),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=WHEEL_JOINTS,
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=150.0,
            ),
            "body": ImplicitActuatorCfg(
                # exclude the wheels: an overlapping ".*" would give them
                # the 800-stiffness position drive and brake the base
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


def self_robot_bodies(robot: Articulation) -> dict:
    """Return body name -> world position for reach tuning."""
    names = robot.data.body_names
    pos = robot.data.body_pos_w.numpy()[0]
    return {n: pos[i] for i, n in enumerate(names)}


def select_physx_variant(robot_prim_path: str) -> None:
    """Pick the 'physx' variant of the asset's Physics variantSet.

    The imported G1D ships physics payloads in a variantSet with NO
    default selection - without it no joints or articulation root are
    composed at all.
    """
    stage = omni.usd.get_context().get_stage()
    variant = stage.GetPrimAtPath(robot_prim_path).GetVariantSet("Physics")
    if variant.GetVariantNames():
        variant.SetVariantSelection("physx")
        print("physics variant: physx selected", flush=True)


def relocate_articulation_root(robot_prim_path: str) -> None:
    """Move ArticulationRootAPI from the nested AGV_link up to the Robot prim.

    Isaac Lab resolves the articulation at the prim it manages; the
    imported G1D asset carries the root API deep inside (Geometry/
    AGV_link), so resolution at /Robot fails.
    """
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and                 str(prim.GetPath()) != robot_prim_path:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"root API removed from {prim.GetPath()}", flush=True)
    if not root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        print(f"root API applied to {robot_prim_path}", flush=True)


def disable_agv_shell_collision() -> bool:
    """Deactivate the AGV shell collision instance (steals ground contact).

    Same surgery as the Wheeled-G1-Demo scene: the imported asset splits
    each link into visual + collision instances, and the shell's collision
    (AGV_link_1) reaches lower than the wheels.
    """
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() == "AGV_link_1":
            prim.SetActive(False)
            print(f"shell collision off: {prim.GetPath()}", flush=True)
            return True
    return False


def yaw_of(quat) -> float:
    """Yaw (rad) from a single xyzw quaternion (this IsaacLab's order)."""
    import math

    x, y, z, w = (
        quat.tolist() if hasattr(quat, "tolist") else list(quat)
    )
    return math.atan2(2.0 * (x * w + y * z), 1.0 - 2.0 * (y * y + z * z))


class G1DDriver:
    """Closed-loop driver over the articulation tensor API."""

    def __init__(self, robot: Articulation, sim: SimulationContext):
        self.robot = robot
        self.sim = sim
        names = self.robot.joint_names
        self.idx = {n: i for i, n in enumerate(names)}
        self.arm_ids = torch.tensor(
            [self.idx[n] for n in RIGHT_ARM], device=robot.device
        )
        self.finger_ids = torch.tensor(
            [self.idx[n] for n in RIGHT_FINGERS], device=robot.device
        )
        self.left_wheel = self.idx[WHEEL_JOINTS[0]]
        self.right_wheel = self.idx[WHEEL_JOINTS[1]]
        self.n_joints = len(names)

    # ---- outputs ----

    def _zeros(self) -> torch.Tensor:
        return torch.zeros((1, self.n_joints), device=self.robot.device)

    def arm(self, pose: torch.Tensor) -> None:
        cmd = self._zeros()
        cmd[0, self.arm_ids] = pose.to(self.robot.device)
        self.robot.set_joint_position_target(cmd)

    def fingers(self, pose: torch.Tensor) -> None:
        cmd = self._zeros()
        cmd[0, self.finger_ids] = pose.to(self.robot.device)
        self.robot.set_joint_position_target(cmd)

    def wheels(self, left: float, right: float) -> None:
        """+v = +x forward; LEFT command negated (mirrored URDF axes)."""
        cmd = self._zeros()
        cmd[0, self.left_wheel] = -left
        cmd[0, self.right_wheel] = right
        self.robot.set_joint_velocity_target(cmd)

    # ---- state ----

    def pose(self):
        pos = torch.as_tensor(
            self.robot.data.root_pos_w.numpy()[0])
        yaw = yaw_of(torch.as_tensor(
            self.robot.data.root_quat_w.numpy()[0]))
        return pos[0].item(), pos[1].item(), yaw

    def finger_max(self) -> float:
        pos = torch.as_tensor(
            self.robot.data.joint_pos.numpy()[0])
        return max(
            abs(pos[self.idx[n]].item()) for n in RIGHT_FINGERS
        )

    # ---- primitives (each: write targets every step, step sim, watch clock) ----

    def run_seconds(self, seconds: float, every_step=None) -> None:
        steps = int(seconds / self.sim.get_physics_dt())
        for _ in range(steps):
            if every_step is not None:
                every_step()
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.sim.get_physics_dt())

    def wait_pose(
        self, pose: torch.Tensor, ids: torch.Tensor, names,
        tolerance: float, timeout: float,
    ) -> bool:
        deadline = timeout / self.sim.get_physics_dt()
        waited = 0
        while waited < deadline:
            cmd = self._zeros()
            cmd[0, ids] = pose.to(self.robot.device)
            if ids is self.arm_ids:
                self.robot.set_joint_position_target(cmd)
            else:
                self.robot.set_joint_position_target(cmd)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.sim.get_physics_dt())
            waited += 1
            current = torch.as_tensor(
                self.robot.data.joint_pos.numpy()[0])
            err = max(
                abs(current[self.idx[n]].item() - t)
                for n, t in zip(names, pose.tolist())
            )
            if err < tolerance:
                return True
        return False

    def arm_to(self, pose, tolerance=0.3, timeout=12.0) -> bool:
        return self.wait_pose(
            pose, self.arm_ids, RIGHT_ARM, tolerance, timeout
        )

    def fingers_to(self, pose, tolerance=0.4, timeout=8.0) -> bool:
        return self.wait_pose(
            pose, self.finger_ids, RIGHT_FINGERS, tolerance, timeout
        )

    def holding(self) -> bool:
        return 0.25 < self.finger_max() < 0.8

    def drive_to(self, target_x: float, timeout: float = 60.0) -> bool:
        """Spin-to-heading then cruise with a yaw-hold differential."""
        if not self.spin_to(0.0, timeout=20.0):
            return False
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        last_dx = None
        while waited < deadline:
            x, _, yaw = self.pose()
            dx = target_x - x
            if abs(dx) < 0.06 and abs(yaw) < 0.3:
                self.wheels(0.0, 0.0)
                print(f"arrived x={x:.3f} (target {target_x:.3f})", flush=True)
                return True
            v = max(-DRIVE_SPEED, min(DRIVE_SPEED, 0.7 * dx))
            if abs(v) < 0.06:
                v = 0.06 if dx > 0 else -0.06
            turn = max(-0.5, min(0.5, -1.0 * yaw))
            self.wheels(
                (v - turn * HALF_TRACK) / WHEEL_RADIUS,
                (v + turn * HALF_TRACK) / WHEEL_RADIUS,
            )
            if waited % int(2.0 / dt) == 0:
                print(f"drive: x={x:.3f} dx={dx:+.3f} yaw={yaw:+.3f}", flush=True)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.sim.get_physics_dt())
            waited += 1
        self.wheels(0.0, 0.0)
        return False

    def spin_to(self, target_yaw: float, timeout: float = 20.0) -> bool:
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        while waited < deadline:
            _, _, yaw = self.pose()
            err = (yaw - target_yaw + 3.14159265) % (2 * 3.14159265) - 3.14159265
            if abs(err) < 0.15:
                self.wheels(0.0, 0.0)
                return True
            rate = max(0.6, min(2.0, 1.5 * abs(err)))
            w = rate if err > 0 else -rate
            self.wheels(
                -w * HALF_TRACK / WHEEL_RADIUS,
                w * HALF_TRACK / WHEEL_RADIUS,
            )
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.sim.get_physics_dt())
            waited += 1
        self.wheels(0.0, 0.0)
        return False


def main() -> None:
    sim = SimulationContext(SimulationCfg(device="cpu", dt=1 / 60))
    scene_cfg = G1DSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    select_physx_variant("/World/envs/env_0/Robot")
    if not disable_agv_shell_collision():
        carb.log_warn("AGV_link_1 not found - shell collision stays on")

    sim.reset()
    print(f"joints: {len(scene['robot'].joint_names)}", flush=True)
    print(f"idx sample: {scene['robot'].joint_names[:6]}", flush=True)

    driver = G1DDriver(scene["robot"], sim)
    x0, y0, yaw0 = driver.pose()
    print(f"root after reset: x={x0:.3f} y={y0:.3f} yaw={yaw0:.3f}", flush=True)
    import time as _t
    for _ in range(120):  # 2 s settle
        sim.step()
        scene["robot"].update(sim.get_physics_dt())
        scene["tote"].update(sim.get_physics_dt())
    tp = scene["tote"].data.root_pos_w.numpy()[0]
    print(f"tote after settle: {tp.round(3)}", flush=True)

    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[4.0, 3.5, 2.2], target=[0.8, 0.0, 0.8],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"camera setup skipped: {exc}", flush=True)

    cycle = 0
    while simulation_app.is_running():
        cycle += 1
        print(f"=== cycle {cycle} ===", flush=True)

        driver.arm_to(ARM_HOME, tolerance=0.35)
        driver.fingers_to(FINGER_OPEN)
        if not driver.drive_to(STANDOFF_X):
            print("drive to table FAILED", flush=True)
            continue
        driver.arm_to(ARM_REACH, tolerance=0.25, timeout=20.0)
        if cycle == 1:
            jp = torch.as_tensor(
                driver.robot.data.joint_pos.numpy()[0])
            for n in ("right_shoulder_pitch_joint", "right_elbow_joint",
                      "right_wrist_pitch_joint"):
                i = driver.idx[n]
                print(
                    f"{n}: actual={jp[i].item():+.3f} "
                    f"cmd={ARM_REACH[RIGHT_ARM.index(n)].item():+.3f}",
                    flush=True,
                )
            bodies = self_robot_bodies(driver.robot)
            palm = bodies.get("right_hand_palm_link")
            if palm is not None:
                print(f"palm after reach: {palm.round(3)}", flush=True)
        driver.fingers_to(FINGER_CLOSE, tolerance=0.45)
        got = driver.holding()
        print(f"grasp check: {'HOLDING' if got else 'empty'}", flush=True)
        # staged lift (single jumps stall against the table)
        for frac in (0.34, 0.67, 1.0):
            stage = ARM_REACH + (ARM_CARRY - ARM_REACH) * frac
            driver.arm_to(stage, tolerance=0.4, timeout=8.0)
        if not driver.drive_to(PAD_POS[0]):
            print("drive to pad FAILED", flush=True)
            continue
        driver.arm_to(ARM_RELEASE, tolerance=0.4)
        driver.fingers_to(FINGER_OPEN)
        print("released on the pad", flush=True)
        driver.drive_to(0.0)
        print(f"=== cycle {cycle} complete ===", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
