"""Live-stage USD surgery: physics variant, colliders, materials."""

import os

import omni.usd
from g1d_config import OBJECT_CATALOG, OBJECT_NAMES, OBJECT_ORDER, WAREHOUSE


def select_physx_variant(robot_prim_path: str) -> None:
    """Pick the 'physx' variant of the asset's Physics variantSet."""
    stage = omni.usd.get_context().get_stage()
    variant = stage.GetPrimAtPath(robot_prim_path).GetVariantSet("Physics")
    if variant.GetVariantNames():
        variant.SetVariantSelection("physx")
        print("physics variant: physx selected", flush=True)


def rigidify_objects(env_ns: str = "/World/envs/env_0") -> int:
    """Make the kit-prop objects proper rigid bodies.

    The props ship collision meshes but NO rigid body on the root
    (RigidObjectCfg cannot resolve them), and their mesh/convex-hull
    colliders are tippy - the crate tipped over on every re-seat and
    slid off the table. So per kit object: RigidBodyAPI + explicit
    mass on the root, the stock mesh colliders DISABLED, and a
    bbox-sized guide Cube child carrying a clean BOX CollisionAPI -
    flat, stable, predictable. Run between scene creation and
    sim.reset(), same window as select_physx_variant(). (A raw
    "physics:approximation" write broke the collider parse entirely
    - the objects fell through the floor to z -42000.)
    """
    from pxr import Gf, Usd, UsdGeom
    from pxr import UsdPhysics as Phys

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], False)
    n = 0
    for nm in OBJECT_NAMES:
        root = stage.GetPrimAtPath(f"{env_ns}/{nm}")
        if not root:
            print(f"rigidify: no prim at {env_ns}/{nm}", flush=True)
            continue
        if not root.HasAPI(Phys.RigidBodyAPI):
            Phys.RigidBodyAPI.Apply(root)
        # de-instance: UsdFileCfg spawns kit props as INSTANCED
        # references ("/World/template/obj_xx/proto_asset_0"), and
        # write_root_pose_to_sim on an instanced body silently does
        # nothing - the re-seat teleports never took effect. A
        # private prim stack makes every runtime write work
        root.SetInstanceable(False)
        usd_path, _color, _grab, mass = OBJECT_CATALOG[_obj_type(nm)]
        if usd_path is None:  # cuboid totes already collide fine
            n += 1
            continue
        Phys.MassAPI.Apply(root)
        Phys.MassAPI(root).CreateMassAttr(mass)
        # disable every stock mesh collider under the object, and
        # measure the visual bbox for the box collider size/offset
        for p in stage.Traverse():
            path = str(p.GetPath())
            if path.startswith(f"{env_ns}/{nm}/") and p.HasAPI(Phys.CollisionAPI):
                Phys.CollisionAPI(p).CreateCollisionEnabledAttr(False)
        root_xo = UsdGeom.Xformable(root)
        root_t = Gf.Vec3d(0.0)
        t_op = root_xo.GetTranslateOp()
        if t_op:
            root_t = Gf.Vec3d(t_op.Get())
        rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
        size = rng.GetSize()
        mid = rng.GetMidpoint() - root_t  # bbox center in root space
        cube = UsdGeom.Cube.Define(stage, f"{env_ns}/{nm}/BoxCol")
        cube.CreateSizeAttr(1.0)
        cube_xo = UsdGeom.Xformable(cube.GetPrim())
        cube_xo.AddTranslateOp().Set(Gf.Vec3d(mid))
        cube_xo.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
        # INVISIBLE, not "guide" purpose: guide prims are SKIPPED by
        # the physics parser - the collider never existed and the
        # objects fell through the tabletop onto the legs below
        cube.CreateVisibilityAttr("invisible")
        Phys.CollisionAPI.Apply(cube.GetPrim())
        print(f"  {nm} box col {size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f} @{mid}", flush=True)
        n += 1
        n += 1
    print(f"rigidified {n} objects", flush=True)
    return n


def _obj_type(obj_name: str) -> str:
    """Catalog type of a spawned object prim name."""
    return OBJECT_ORDER[OBJECT_NAMES.index(obj_name)]


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
    from pxr import Sdf, UsdShade

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
        bind = nm in OBJECT_NAMES or (
            nm in ("right_hand_thumb_2_link_1", "right_hand_middle_1_link_1", "right_hand_index_1_link_1")
        )
        if bind:
            b = UsdShade.MaterialBindingAPI.Apply(prim)
            b.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            bound += 1
    print(f"grip material bound to {bound} prims", flush=True)
