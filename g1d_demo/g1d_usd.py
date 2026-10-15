"""Live-stage USD surgery: physics variant, colliders, materials."""

import os

import omni.usd
from g1d_config import TOTE_NAMES, WAREHOUSE


def select_physx_variant(robot_prim_path: str) -> None:
    """Pick the 'physx' variant of the asset's Physics variantSet."""
    stage = omni.usd.get_context().get_stage()
    variant = stage.GetPrimAtPath(robot_prim_path).GetVariantSet("Physics")
    if variant.GetVariantNames():
        variant.SetVariantSelection("physx")
        print("physics variant: physx selected", flush=True)


def strip_arm_colliders() -> int:
    """Remove every ARM collider (shoulder..fingertips) - the
    DEFAULT for the snap demo.

    The grasp is a kinematic snap and the carry a rigid pose
    write, so arm colliders have no functional role - and the
    held box (its own collider stays ON so the drop is physical)
    wedging the WRIST/FOREARM hulls shoved the arm the whole
    carry: commanded lifts sagged into a fold (sh -0.65 -> -0.33,
    hand z 0.99 -> 0.73 over 10 s) - the visible "press-down
    after pickup". Wheels, casters and the AGV body keep their
    colliders. (Tried toggling the TOTE collider at runtime
    instead: disabling works but the GPU pipeline never
    re-registers it - the released box fell through the ground.)
    """
    from pxr import UsdPhysics as Phys

    stage = omni.usd.get_context().get_stage()
    parts = ("hand", "arm", "wrist", "forearm", "shoulder", "elbow")
    stripped = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/Robot/" not in path or not prim.HasAPI(Phys.CollisionAPI):
            continue
        if any(p in path.lower() for p in parts):
            Phys.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            stripped += 1
    return stripped


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


def bind_placeholder_materials(env_ns="/World/envs/env_0"):
    """Rebind the cone + sign props to UsdPreviewSurface networks.

    Their MDL materials (M_TrafficCone/M_AisleSign) bake the texture
    paths into the material BODY (no texture parameters), and the RTX
    real-time renderer resolves those './Textures/...' defaults
    differently from the offscreen path - the props rendered pitch
    black in the GUI. PreviewSurface + USD-anchored PNGs renders the
    same everywhere (every functional cuboid in the scene uses it).
    Textures: g1d_make_textures.py output.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    tex = os.path.join(WAREHOUSE, "Materials", "Textures")

    def preview_material(stage, path, diffuse_png, roughness):
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        tex_node = UsdShade.Shader.Define(stage, path + "/DiffuseTex")
        tex_node.CreateIdAttr("UsdUVTexture")
        tex_node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_png)
        tex_node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex_node.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_node.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        reader = UsdShade.Shader.Define(stage, path + "/UVReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex_node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_node.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    specs = {
        # the sign's Text png IS the final look (black board, white
        # arrow); the cone's D png has the orange body + white band
        "S_TrafficCone": (os.path.join(tex, "T_TrafficCone_D.png"), 0.35),
        "S_AisleSign": (os.path.join(tex, "AisleSign_Text_01.png"), 0.5),
    }
    stage = omni.usd.get_context().get_stage()
    materials = {}
    for prefix, (png, rough) in specs.items():
        materials[prefix] = preview_material(stage, f"{env_ns}/PreviewMaterials/{prefix}Mat", png, rough)
    bound = 0
    for prim in stage.Traverse():
        nm = prim.GetName()
        for prefix, mat in materials.items():
            if nm.startswith(prefix) and prim.GetTypeName() == "Mesh":
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants)
                bound += 1
    print(f"placeholder materials bound to {bound} meshes", flush=True)


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
