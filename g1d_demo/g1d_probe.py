"""Static arm-pose probe: command poses, print fingertip geometry.

Calibrates the palm-down wrist angles and the hover/dock elbow
heights for the snap grasp - robot only, no warehouse dressing
(fast headless startup):

    docker exec isaaclab_arena-latest bash -c \
      "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
       g1d_demo/g1d_probe.py --headless"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Now the sim is alive."""

import torch

import isaaclab.sim as sim_utils
from g1d_config import G1D_USD, HOME_ROOT_Z, LEFT_ARM, LEFT_ARM_TUCK, RIGHT_ARM, WHEEL_JOINTS
from g1d_driver import n
from g1d_usd import select_physx_variant
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

# tote reference heights (station slot): center 0.87, top 0.93,
# post top 0.805 - the glued tote must clear the post on attach
TOTE_Z = 0.87


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    """Bare stage: ground, light, robot."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
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
                joint_names_expr=["^(?!Left_Wheel_Joint$|Right_Wheel_Joint$).*$"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )


LINKS = ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(SimulationCfg(device=device, dt=1 / 60))
    scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=4.0))
    select_physx_variant("/World/envs/env_0/Robot")
    sim.reset()
    robot = scene["robot"]
    names = robot.joint_names
    idx = {nm: i for i, nm in enumerate(names)}
    arm_ids = torch.tensor([idx[nm] for nm in RIGHT_ARM], device=device)
    left_ids = torch.tensor([idx[nm] for nm in LEFT_ARM], device=device)

    def hold(pose, seconds=1.2):
        cmd = torch.zeros((1, len(names)), device=device)
        cmd[0, left_ids] = LEFT_ARM_TUCK.to(device)
        cmd[0, arm_ids] = torch.tensor(pose, dtype=torch.float32, device=device)
        for _ in range(int(seconds * 60)):
            robot.set_joint_position_target(cmd)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim.get_physics_dt())

    def tips():
        pos = n(robot.data.body_pos_w)[0]
        bnames = robot.data.body_names
        return {nm: pos[bnames.index(nm)] for nm in LINKS}

    hold([0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0], 1.0)  # settle

    print("== wrist_pitch sweep: flat jaw = thumb z == middle z ==", flush=True)
    print("(shoulder -0.70, elbow 1.30, roll 0)", flush=True)
    for p in (-0.35, -0.65, -0.80, -0.90, -1.00):
        hold([-0.70, 0.0, 0.0, 1.30, 0.0, p, 0.0])
        t = tips()
        th, md = t[LINKS[0]], t[LINKS[2]]
        cz = (t[LINKS[0]][2] + t[LINKS[1]][2] + t[LINKS[2]][2]) / 3.0
        print(
            f"  pitch {p:+.2f}: thumb_z={th[2]:.3f} "
            f"mid_z={md[2]:.3f} dz={md[2] - th[2]:+.3f} "
            f"jaw_c_z={cz:.3f} gap_y={abs(th[1] - md[1]) * 100:.1f}cm",
            flush=True,
        )

    print("== elbow sweep at pitch -0.85 (palm down): attach height ==", flush=True)
    print(f"(tote center {TOTE_Z}, snap radius 0.22)", flush=True)
    for e in (1.30, 1.15, 1.05, 0.95, 0.85, 0.75):
        hold([-0.70, 0.0, 0.0, e, 0.0, -0.85, 0.0])
        t = tips()
        cz = (t[LINKS[0]][2] + t[LINKS[1]][2] + t[LINKS[2]][2]) / 3.0
        cx = (t[LINKS[0]][0] + t[LINKS[1]][0] + t[LINKS[2]][0]) / 3.0
        print(
            f"  elbow {e:.2f}: jaw_c=({cx:.3f},{cz:.3f}) "
            f"dz_to_tote={cz - TOTE_Z:+.3f} "
            f"glued_bottom={(cz - 0.07 - 0.06):.3f} vs post 0.805",
            flush=True,
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
