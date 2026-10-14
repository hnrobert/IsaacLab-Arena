"""Scene construction: warehouse kit dressing, station, totes, robot."""

import os

import isaaclab.sim as sim_utils
import omni.usd
from g1d_config import (
    G1D_USD,
    HOME_ROOT_Z,
    PAD_COLOR,
    PAD_POS,
    PICK_TARGETS,
    STATION_BODY,
    STATION_TOP,
    STATION_X,
    TOTE_SIZE,
    TOTE_Z,
    WAREHOUSE,
    WHEEL_JOINTS,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
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
            props.append(usd_prop_cfg(prop("SM_FloorDecal_StripeFull_4m"), (x, y, 0.004)))
    # perimeter walls: rotate the 3 m panels so the long axis runs x
    for y in (-3.2, 3.2):
        for x in (-3.0, 0.0, 3.0):
            props.append(usd_prop_cfg(prop("SM_WallA_3M"), (x, y, 0.0), rot_z=90.0))
    # end walls (default orientation: long axis along y)
    for x in (-4.6, 4.6):
        props.append(usd_prop_cfg(prop("SM_WallA_3M"), (x, 0.0, 0.0)))
    # pallet racking flanking the aisle: 4 m shelf runs at two levels,
    # upright frames at the ends, goods piled on the beams
    for y_r in (1.6, -1.6):
        for x_c in (-1.6, 2.0):
            for z_lvl in (0.35, 1.45):
                props.append(usd_prop_cfg(prop("SM_RackShelf_01"), (x_c, y_r, z_lvl)))
            for x_e in (x_c - 2.0, x_c + 2.0):
                props.append(usd_prop_cfg(prop("SM_RackFrame_03"), (x_e, y_r, 1.5)))
            props.append(usd_prop_cfg(prop("SM_RackPile_03"), (x_c - 1.2, y_r, 0.40)))
            props.append(usd_prop_cfg(prop("SM_RackPile_04"), (x_c + 0.9, y_r, 0.42)))
    # floor goods near the aisle ends
    props.append(usd_prop_cfg(prop("SM_PaletteA_02"), (3.3, -1.2, 0.0)))
    props.append(usd_prop_cfg(prop("SM_CardBoxA_01"), (3.3, -1.2, 0.34)))
    props.append(usd_prop_cfg(prop("SM_CardBoxB_01"), (-3.3, 1.2, 0.0)))
    props.append(usd_prop_cfg(prop("SM_CratePlastic_A_01"), (-3.3, -1.2, 0.0)))
    # wall-mounted gear + traffic furniture (cones OUT of the
    # +-0.55 delivery lanes - at +-0.85 they left ~2 cm to the
    # base corners and one run ground to a halt against one)
    props.append(usd_prop_cfg(prop("SM_FuseBox_01"), (4.45, 1.9, 0.0)))
    props.append(usd_prop_cfg(prop("S_TrafficCone"), (-0.4, -1.4, 0.0)))
    props.append(usd_prop_cfg(prop("S_TrafficCone"), (-1.6, 1.4, 0.0)))
    props.append(usd_prop_cfg(prop("S_AisleSign"), (1.2, -1.1, 0.0), rot_z=180.0))
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=STATION_BODY),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(STATION_X, 0.0, STATION_TOP / 2),
        ),
    )
    station_beam = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StationBeam",
        spawn=sim_utils.CuboidCfg(
            size=(0.52, 1.12, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
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
            size=(0.24, 1.10, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=STATION_BODY),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, 0.0, 0.75)),
    )
    # (retaining pockets REMOVED: single-slot-era walls whose lips
    # sit in the jaw descent band and sweep totes off the off-center
    # slots; knock-reset handles nudges now)
    pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pad",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.4, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PAD_COLOR),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=PAD_POS),
    )

    # presenting posts: each tote sits on a small post, body in
    # free air ~4 cm above the board - the fingers close on the tote
    # only (no board contact = no knock/tip/lip chaos). Default
    # material - no color livery (user: no red in the scene)
    post_a = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PostA",
        spawn=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.035), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, 0.00, 0.7875)),
    )
    post_b = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PostB",
        spawn=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.035), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, 0.24, 0.7875)),
    )
    post_c = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PostC",
        spawn=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.035), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, -0.24, 0.7875)),
    )
    post_d = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PostD",
        spawn=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.035), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, 0.48, 0.7875)),
    )
    post_e = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PostE",
        spawn=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.035), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.27, -0.48, 0.7875)),
    )

    # --- pick targets on the station board ---
    # every tote spawns ON its station slot (the rack dressing
    # positions are gone - all five are pick targets now)
    tote_a = make_tote_cfg("tote_a", (2.27, 0.00, TOTE_Z), PICK_TARGETS[0][3])
    tote_b = make_tote_cfg("tote_b", (2.27, 0.24, TOTE_Z), PICK_TARGETS[1][3])
    tote_c = make_tote_cfg("tote_c", (2.27, -0.24, TOTE_Z), PICK_TARGETS[2][3])
    tote_d = make_tote_cfg("tote_d", (2.27, 0.48, TOTE_Z), PICK_TARGETS[3][3])
    tote_e = make_tote_cfg("tote_e", (2.27, -0.48, TOTE_Z), PICK_TARGETS[4][3])

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
                # exclude the wheels: an overlapping ".*" would give
                # them the 800-stiffness position drive and brake the
                # base. Also excludes the wrists - they get their own
                # stiff group below.
                joint_names_expr=[
                    "^(?!Left_Wheel_Joint$|Right_Wheel_Joint$"
                    "|left_wrist_(roll|pitch|yaw)_joint$"
                    "|right_wrist_(roll|pitch|yaw)_joint$).*$"
                ],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
            # dedicated WRIST group: under descent contact the wrist
            # yaw deflected 0.8 rad (jaw rotates 47 deg, tips shift
            # 24 cm lateral -> every phantom close round). 3x the
            # stiffness should hold the jaw square on the tote.
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=["^(left|right)_wrist_(roll|pitch|yaw)_joint$"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=2400.0,
                damping=60.0,
            ),
        },
    )


# attach the warehouse dressing as numbered fields (configclass turns
# every class attribute into an entity, so inject them dynamically)
for _i, _cfg in enumerate(build_warehouse_props()):
    setattr(G1DSceneCfg, f"wh_prop_{_i:02d}", _cfg)
