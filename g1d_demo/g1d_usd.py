"""Live-stage USD surgery: physics variant, colliders, materials."""

import omni.usd
from g1d_config import TOTE_NAMES


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


def arm_collision_hand_only() -> int:
    """Strip collision from every non-hand robot collider.

    Policy (user): colliders ONLY on the hands - plus wheels and
    casters, which driving needs. The shoulder cylinders were strut
    sources: arm-vs-station contact wedged base->arm->board and
    deadlocked the wheels (d16), and contact shove-induced base yaw
    (+0.5 rad) hooked the hand under the board (d17).
    """
    from pxr import UsdPhysics as Phys

    stage = omni.usd.get_context().get_stage()
    stripped = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Robot/" not in path or not prim.HasAPI(Phys.CollisionAPI):
            continue
        nm = prim.GetName()
        parent = prim.GetParent().GetName()
        keep = (
            "Wheel_Link" in parent  # wheel cylinders
            or nm.startswith("caster_")  # AGV ground support
            or "_link_1" in parent  # fingertip hulls
            or (nm == "box" and "hand" in path)  # thumb pad boxes
        )
        if keep:
            continue
        Phys.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        stripped += 1
        print(f"collision off: {path}", flush=True)
    return stripped


def add_grip_material(env_ns="/World/envs/env_0") -> None:
    """High-friction physics material on fingertips + totes.

    The pinch wraps the tote but default friction (~0.5) strips it
    during any base motion; rubber-ish 1.6/1.4 makes the wrap hold.
    """
    from pxr import UsdPhysics as Phys
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, env_ns + "/GripMaterial")
    Phys.MaterialAPI.Apply(mat.GetPrim())
    pa = Phys.MaterialAPI(mat.GetPrim())
    pa.CreateStaticFrictionAttr(1.6)
    pa.CreateDynamicFrictionAttr(1.4)
    pa.CreateRestitutionAttr(0.0)
    bound = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        bind = nm in TOTE_NAMES or (
            nm in ("right_hand_thumb_2_link_1", "right_hand_middle_1_link_1", "right_hand_index_1_link_1")
        )
        if bind:
            b = UsdShade.MaterialBindingAPI.Apply(prim)
            b.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            bound += 1
    print(f"grip material bound to {bound} prims", flush=True)
