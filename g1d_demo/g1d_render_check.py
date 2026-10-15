"""One-off visual check: render the warehouse hall dressing offscreen.

Verifies the MDL materials compile and show their textures (not the
red error fallback), and that the scaled hall enclosure (full wall
runs, closed corners) looks right:

    docker exec isaaclab_arena-latest bash -c \
      "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
       g1d_demo/g1d_render_check.py --headless"
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True, offscreen_render=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Now the sim is alive."""

import isaaclab.sim as sim_utils
import torch
from g1d_scene import build_warehouse_props, rotate_props, usd_prop_cfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass


def lookat_quat_xyzw(pos, target, up=(0.0, 0.0, 1.0)):
    """xyzw quaternion orienting an OpenGL-convention camera (looks
    down local -Z, +Y up) at `pos` towards `target`."""
    f = [t - p for t, p in zip(target, pos)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    z = [-c for c in f]  # camera local +Z is AWAY from the target
    x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]
    n = math.sqrt(sum(c * c for c in x))
    x = [c / n for c in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    # cam->world matrix has the camera axes as COLUMNS
    m = [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
    tr = m[0] + m[4] + m[8]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, i, j, k = 0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w, i, j, k = (m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w, i, j, k = (m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s
    else:
        s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w, i, j, k = (m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s
    return (i, j, k, w)


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(
            pos=(15.0, 11.0, 12.0),
            rot=lookat_quat_xyzw((15.0, 11.0, 12.0), (0.0, 0.0, 0.5)),
            convention="opengl",
        ),
    )
    cam2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera2",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
        offset=CameraCfg.OffsetCfg(
            pos=(-7.0, 0.0, 1.6),
            rot=lookat_quat_xyzw((-7.0, 0.0, 1.6), (9.0, 0.0, 1.0)),
            convention="opengl",
        ),
    )


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60, device="cpu"))
    # the full hall dressing, attached like G1DSceneCfg does
    for i, cfg in enumerate(build_warehouse_props()):
        setattr(CheckSceneCfg, f"wh_prop_{i:02d}", cfg)
    scene = InteractiveScene(CheckSceneCfg(num_envs=1, env_spacing=0.0))
    rotate_props()  # apply the recorded wall/prop yaw rotations
    sim.reset()
    for _ in range(12):  # RTX warmup frames
        sim.render()
    for name, tag in (("cam", "hall"), ("cam2", "aisle")):
        cam = scene[name]
        cam.update(dt=sim.get_physics_dt())
        img = cam.data.output["rgb"][0].cpu().numpy()
        out = f"/workspaces/isaaclab_arena/g1d_demo/_render_check_{tag}.png"
        from PIL import Image

        Image.fromarray(img[:, :, :3]).save(out)
        print(f"saved {out} mean {img[:, :, :3].reshape(-1, 3).mean(axis=0)}", flush=True)


main()
simulation_app.close()
