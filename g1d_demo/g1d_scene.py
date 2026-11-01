"""Scene construction: warehouse hall dressing, 12-table grid, objects, robot."""

import os

import isaaclab.sim as sim_utils
import omni.usd
from g1d_config import (
    G1D_USD,
    HALL_X,
    HALL_Y,
    HOME_ROOT_Z,
    OBJECT_CATALOG,
    OBJECT_NAMES,
    OBJECT_ORDER,
    RACK_Y,
    STOCKED_TABLES,
    TABLE_POS,
    TABLE_SIZE,
    TABLE_TOP,
    TOTE_SIZE,
    WAREHOUSE,
    WHEEL_JOINTS,
    table_slot,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

TABLE_COLOR = (0.55, 0.5, 0.42)  # wood-tone top, matches the old station
LEG_COLOR = (0.35, 0.33, 0.3)


def prop(name):
    """Absolute container path of a Simple_Warehouse prop."""
    return os.path.join(WAREHOUSE, "Props", name + ".usd")


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


def _centers(half, step):
    """Panel-tile centers covering -half..+half with `step`-spaced runs."""
    n = 2 * half / step
    assert abs(n - round(n)) < 1e-6, f"{half} m span not a multiple of the {step} m panel"
    return [-half + step * (i + 0.5) for i in range(int(round(n)))]


def build_tables():
    """12 work tables: a top slab plus four legs each, as spawn cfgs."""
    cfgs = []
    tx_len, ty_len, t_thick = TABLE_SIZE
    leg_h = TABLE_TOP - t_thick
    for ti, (cx, cy) in enumerate(TABLE_POS):
        cfgs.append(
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Table{ti:02d}",
                spawn=sim_utils.CuboidCfg(
                    size=(tx_len, ty_len, t_thick),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, TABLE_TOP - t_thick / 2)),
            )
        )
        # legs at the four corners (USD prim names cannot contain
        # dots, so they are numbered, not coordinate-named)
        for li, (lx, ly) in enumerate([
            (-tx_len / 2 + 0.09, -ty_len / 2 + 0.09),
            (tx_len / 2 - 0.09, -ty_len / 2 + 0.09),
            (-tx_len / 2 + 0.09, ty_len / 2 - 0.09),
            (tx_len / 2 - 0.09, ty_len / 2 - 0.09),
        ]):
            cfgs.append(
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Table{ti:02d}_L{li}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.07, 0.07, leg_h),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=LEG_COLOR),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(cx + lx, cy + ly, leg_h / 2)),
                )
            )
    return cfgs


def make_object_cfg(name: str, obj_type: str, pos):
    """One pickable object at `pos` - a kit USD prop or a cuboid tote."""
    usd_path, color, _grab, mass = OBJECT_CATALOG[obj_type]
    if usd_path:
        spawn = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    else:
        spawn = sim_utils.CuboidCfg(
            size=(TOTE_SIZE, TOTE_SIZE, TOTE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=spawn,
        # spawn 0.25 m above the table and let it settle - the kit
        # prop origins/heights vary, so a drop-in beats guessing z
        init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], TABLE_TOP + 0.25)),
    )


def build_warehouse_props():
    """All static Simple_Warehouse dressing, as spawn cfgs."""
    props = []
    # concrete-look 6 m floor tiles slab over the whole hall (visual;
    # the ground plane below provides collision)
    for x in _centers(HALL_X, 6.0):
        for y in _centers(HALL_Y, 6.0):
            props.append(usd_prop_cfg(prop("SM_floor02"), (x, y, 0.002)))
    # center-corridor lane stripes (4 m decals; span tiles in 4 m)
    for x in _centers(HALL_X - 4.0, 4.0):
        for y in (-0.62, 0.62):
            props.append(usd_prop_cfg(prop("SM_FloorDecal_StripeFull_4m"), (x, y, 0.004)))
    # perimeter: full 3 m panel runs on all four sides, corners
    # cross-overlapping solid
    for y in (HALL_Y, -HALL_Y):
        for x in _centers(HALL_X, 3.0):
            props.append(usd_prop_cfg(prop("SM_WallA_3M"), (x, y, 0.0), rot_z=90.0))
    for x in (HALL_X, -HALL_X):
        for y in _centers(HALL_Y, 3.0):
            props.append(usd_prop_cfg(prop("SM_WallA_3M"), (x, y, 0.0)))
    # pallet racking in the far north/south bands (outside the table
    # rows): uprights sit at their FOOT (origin z 0..3), shelves ride
    # the beams (origin mid-slab). 5 m bay pitch tiles the 40 m span.
    for y_r in (RACK_Y, -RACK_Y):
        for x_c in _centers(HALL_X - 4.0, 5.0):
            for z_lvl in (0.35, 1.45):
                props.append(usd_prop_cfg(prop("SM_RackShelf_01"), (x_c, y_r, z_lvl)))
            for x_e in (x_c - 2.0, x_c + 2.0):
                props.append(usd_prop_cfg(prop("SM_RackFrame_03"), (x_e, y_r, 0.0)))
            props.append(usd_prop_cfg(prop("SM_RackPile_03"), (x_c - 1.2, y_r, 0.40)))
            props.append(usd_prop_cfg(prop("SM_RackPile_04"), (x_c + 0.9, y_r, 0.42)))
    # traffic furniture OUT of the center corridor (the transit lane
    # is |y| < 0.8; tables occupy 0.8..1.7) - beside the table rows
    props.append(usd_prop_cfg(prop("S_TrafficCone"), (-2.5, 2.8, 0.0)))
    props.append(usd_prop_cfg(prop("S_TrafficCone"), (-6.8, -2.8, 0.0)))
    # S_AisleSign's origin is CENTERED (local z -0.39..0.39) - lift
    # it half its height or it sits half-buried
    props.append(usd_prop_cfg(prop("S_AisleSign"), (-4.5, 3.2, 0.39), rot_z=180.0))
    return props


@configclass
class G1DSceneCfg(InteractiveSceneCfg):
    """Warehouse hall: kit dressing, 12-table grid, objects, G1D."""

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
            # dedicated WRIST group: 3x stiffness holds the jaw
            # square under descent contact
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=["^(left|right)_wrist_(roll|pitch|yaw)_joint$"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=2400.0,
                damping=60.0,
            ),
        },
    )


# the 12 pickable objects on the stocked tables (task order), the
# tables, and the warehouse dressing attach as numbered fields
# (configclass turns every class attribute into an entity, so
# inject them dynamically)
_si = 0
for _ti in STOCKED_TABLES:
    for _slot in (0, 1):
        setattr(
            G1DSceneCfg,
            OBJECT_NAMES[_si],
            make_object_cfg(OBJECT_NAMES[_si], OBJECT_ORDER[_si], table_slot(_ti, _slot)),
        )
        _si += 1

for _i, _cfg in enumerate(build_tables()):
    setattr(G1DSceneCfg, f"table_{_i:02d}", _cfg)

for _i, _cfg in enumerate(build_warehouse_props()):
    setattr(G1DSceneCfg, f"wh_prop_{_i:02d}", _cfg)
