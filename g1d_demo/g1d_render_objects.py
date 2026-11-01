"""Quick visual check: render ONLY the pickable objects with their
bound textures (no hall dressing - fast even on CPU).

    docker exec isaaclab_arena-latest bash -c \
      "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
       g1d_demo/g1d_render_objects.py"
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True, offscreen_render=True, enable_cameras=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from g1d_config import OBJECT_NAMES, OBJECT_ORDER, STOCKED_TABLES, table_slot  # noqa: E402
from g1d_scene import build_tables, make_object_cfg  # noqa: E402
from g1d_usd import bind_object_materials, rigidify_objects  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


def lookat_quat_xyzw(pos, target, up=(0.0, 0.0, 1.0)):
    f = [t - p for t, p in zip(target, pos)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    z = [-c for c in f]
    x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]
    n = math.sqrt(sum(c * c for c in x))
    x = [c / n for c in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
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
class ObjSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
        offset=CameraCfg.OffsetCfg(
            # over the first stocked table's slots, looking down
            pos=(-11.2, -2.4, 2.0),
            rot=lookat_quat_xyzw((-11.2, -2.4, 2.0), (-11.5, -1.0, 0.75), up=(0.0, 1.0, 0.0)),
            convention="opengl",
        ),
    )


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60, device="cpu"))
    for i, cfg in enumerate(build_tables()):
        setattr(ObjSceneCfg, f"table_{i:02d}", cfg)
    for i, nm in enumerate(OBJECT_NAMES):
        setattr(ObjSceneCfg, nm, make_object_cfg(nm, OBJECT_ORDER[i], table_slot(STOCKED_TABLES[i // 2], i % 2)))
    scene = InteractiveScene(ObjSceneCfg(num_envs=1, env_spacing=0.0))
    rigidify_objects()  # USD props need rigid bodies before reset
    bind_object_materials()
    sim.reset()
    for _ in range(10):
        sim.render()
    cam = scene["cam"]
    cam.update(dt=sim.get_physics_dt())
    img = cam.data.output["rgb"][0].cpu().numpy()
    from PIL import Image

    out = "/workspaces/isaaclab_arena/g1d_demo/_render_objects.png"
    Image.fromarray(img[:, :, :3]).save(out)
    print(f"saved {out}", flush=True)


main()
simulation_app.close()
