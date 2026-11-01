"""Live-stage USD surgery: physics variant, colliders, materials."""

import os

import omni.usd
from g1d_config import OBJECT_CATALOG, OBJECT_NAMES, OBJECT_ORDER, OBJECT_TEXTURE, WAREHOUSE


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
        # strip every PHYSICS schema from the whole subtree FIRST:
        # ArtVIP models are articulated twins carrying multiple
        # RigidBodyAPI links - RigidObjectCfg demands exactly ONE
        # root rigid body. Only the root RigidBody (re-applied
        # below) + our BoxCol cube survive.
        for p in stage.Traverse():
            path = str(p.GetPath())
            if path != f"{env_ns}/{nm}" and not path.startswith(f"{env_ns}/{nm}/"):
                continue
            lop = p.GetMetadata("apiSchemas")  # Sdf.TokenListOp
            items = list(lop.GetAddedOrExplicitItems()) if lop else []
            keep = [s for s in items if not (str(s).startswith("Physics") or str(s).startswith("Physx"))]
            if keep != items:
                from pxr import Sdf

                op = Sdf.TokenListOp()
                op.explicitItems = keep
                p.SetMetadata("apiSchemas", op)
        Phys.RigidBodyAPI.Apply(root)
        Phys.MassAPI.Apply(root)
        Phys.MassAPI(root).CreateMassAttr(mass)
        # ArtVIP bodies initialize at the STAGE ORIGIN regardless of
        # the spawn translate (seen: every ArtVIP prop reads
        # (0,0,0) after reset) - explicitly write the slot position
        # onto the root xform so the body composes in place
        from g1d_config import STOCKED_TABLES as _ST
        from g1d_config import TABLE_TOP as _TT
        from g1d_config import table_slot as _slot

        i_obj = OBJECT_NAMES.index(nm)
        sx, sy = _slot(_ST[i_obj // 2], i_obj % 2)
        xo = UsdGeom.Xformable(root)
        t_op = xo.GetTranslateOp()
        if t_op is None:
            t_op = xo.AddTranslateOp()
        t_op.Set(Gf.Vec3d(sx, sy, _TT + 0.25))
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


def bind_object_materials(env_ns: str = "/World/envs/env_0") -> int:
    """Bind textured PreviewSurface networks to the generated
    primitives (cups/spheres/books/towels/totes).

    A plain diffuse_color renders FLAT in RTX (user report); a
    UsdUVTexture-driven PreviewSurface gives a real material look -
    same proven pattern as bind_placeholder_materials. Textures
    from g1d_make_object_textures.py; per-type assignment via
    OBJECT_TEXTURE in the config. Kit/ArtVIP USD props keep their
    own materials. The IsaacLab shape spawners create the gprim AT
    the object root (Cube/Sphere/Cylinder typed), so the material
    binds to the root prim itself.
    """
    from pxr import Sdf, UsdShade

    texdir = os.path.join(os.path.dirname(WAREHOUSE), "materials")

    def preview_material(stage, path, diffuse_png, roughness, metallic):
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
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

    stage = omni.usd.get_context().get_stage()
    # one shared material per (texture, finish) combo
    cache = {}
    bound = 0
    for nm in OBJECT_NAMES:
        obj_type = _obj_type(nm)
        spec = OBJECT_TEXTURE.get(obj_type)
        if spec is None:
            continue  # USD props carry their own materials
        png, rough, metal = spec
        key = (png, rough, metal)
        if key not in cache:
            cache[key] = preview_material(
                stage,
                f"{env_ns}/ObjectMaterials/{obj_type}",
                os.path.join(texdir, png),
                rough,
                metal,
            )
        root = stage.GetPrimAtPath(f"{env_ns}/{nm}")
        if not root:
            continue
        # The shape spawners bind their own diffuse-color material
        # DIRECTLY on the gprim, and a weakerThanDescendants root
        # bind loses to it (why the cups/spheres/books/towels stayed
        # flat). Rebind on EVERY gprim under the object, clearing
        # any existing direct binding first.
        for p in stage.Traverse():
            path = str(p.GetPath())
            if path != f"{env_ns}/{nm}" and not path.startswith(f"{env_ns}/{nm}/"):
                continue
            if p.GetTypeName() not in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"):
                continue
            rel = p.GetRelationship("material:binding")
            if rel:
                rel.ClearTargets(True)  # drop the spawner's flat color
            UsdShade.MaterialBindingAPI.Apply(p).Bind(cache[key], UsdShade.Tokens.strongerThanDescendants)
            _write_box_uv(p)
            bound += 1
    print(f"object materials bound to {bound} prims", flush=True)
    return bound


def _write_box_uv(prim) -> None:
    """Author an `st` primvar via box projection.

    The IsaacLab shape spawners create Meshes with NO uv primvar -
    the UsdPrimvarReader in the material network then returns a
    constant and the texture collapses to ONE texel (the flat
    gray/white every generated object rendered as). Triplanar box
    projection from the mesh points gives each face a varying uv,
    so grain/speckle/weave actually show.
    """
    import numpy as np

    from pxr import Sdf, UsdGeom

    mesh = UsdGeom.Mesh(prim)
    existing = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if existing.IsDefined():
        return  # authored UVs win (SimReady meshes ship proper st)
    pts = mesh.GetPointsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not fvi:
        return
    pts = np.array([[c for c in p] for p in pts])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    uvs, idx = [], []
    for v in fvi:
        q = (pts[v] - lo) / span
        # dominant axis of this vertex's offset from the mesh center
        d = np.abs(pts[v] - (lo + hi) / 2.0)
        ax = int(np.argmax(d))
        if ax == 0:
            uv = (q[1], q[2])
        elif ax == 1:
            uv = (q[0], q[2])
        else:
            uv = (q[0], q[1])
        uvs.append(uv)
        idx.append(len(uvs) - 1)
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    pv.Set(uvs)
    pv.SetIndicesAttr(idx)


def attach_parallel_gripper(driver) -> bool:
    """Swap the three-finger hand for a REAL Robotiq 2F-85 parallel
    gripper (user spec: a normal two-finger gripper).

    The full Robotiq base.usda (base + both finger assemblies, PAD
    OPEN pose baked) is referenced as a WORLD-SPACE prim at env
    level and driven from the palm pose every step by
    gripper_apply - nesting it under the articulation palm link
    flung the robot to x=-331 even with every physics schema
    stripped, so it must NOT touch the articulation. The Robotiq's
    own physics schemas are stripped; the three-finger hand
    geometry is hidden. Call AFTER sim.reset() with a settled
    robot (palm-frame axes come from the live finger geometry).
    """
    import numpy as np
    import os as _os

    from g1d_driver import n, quat_rot_inv
    from pxr import Sdf, UsdGeom

    gripper_usd = _os.path.join(_os.path.dirname(WAREHOUSE), "robotiq2f85", "base.usda")
    stage = omni.usd.get_context().get_stage()
    names = driver.robot.data.body_names
    palm_i = names.index("right_hand_palm_link")
    th_i = names.index("right_hand_thumb_2_link")
    md_i = names.index("right_hand_middle_1_link")
    pos = n(driver.robot.data.body_pos_w)[0]
    quats = n(driver.robot.data.body_quat_w)[0]
    palm_p = np.array(pos[palm_i])
    palm_q = np.array(quats[palm_i])
    fwd = np.array(pos[md_i]) - palm_p  # palm -> fingertips
    lat = np.array(pos[th_i]) - np.array(pos[md_i])  # jaw axis
    fwd = fwd / np.linalg.norm(fwd)
    lat = lat - fwd * float(np.dot(lat, fwd))
    lat = lat / np.linalg.norm(lat)
    f_loc = np.array(quat_rot_inv(palm_q.tolist(), fwd.tolist()))
    l_loc = np.array(quat_rot_inv(palm_q.tolist(), lat.tolist()))

    # hide the three-finger GEOMETRY (visual only)
    hidden = 0
    for p in stage.Traverse():
        nm = p.GetName()
        path = str(p.GetPath())
        if "/Robot/" not in path or not nm.startswith("right_hand_"):
            continue
        rest = nm[len("right_hand_") :]
        if rest.startswith(("thumb", "middle", "index")):
            UsdGeom.Imageable(p).CreateVisibilityAttr("invisible")
            hidden += 1

    if not _os.path.isfile(gripper_usd):
        print("gripper: Robotiq asset missing, hand left as-is", flush=True)
        return False
    prim = stage.DefinePrim("/World/envs/env_0/GripperHand", "Xform")
    prim.GetReferences().AddReference(gripper_usd)
    xfo = UsdGeom.Xformable(prim)
    t_op = xfo.AddTranslateOp()
    o_op = xfo.AddOrientOp()
    # strip EVERY physics schema in the referenced subtree: nested
    # bodies/colliders under nothing would still fight the solver
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not path.startswith("/World/envs/env_0/GripperHand"):
            continue
        lop = p.GetMetadata("apiSchemas")
        items = list(lop.GetAddedOrExplicitItems()) if lop else []
        keep = [s for s in items if not (str(s).startswith("Physics") or str(s).startswith("Physx"))]
        if keep != items:
            op = Sdf.TokenListOp()
            op.explicitItems = keep
            p.SetMetadata("apiSchemas", op)
    driver._gripper = {
        "palm_i": palm_i,
        "f_loc": f_loc,
        "l_loc": l_loc,
        "ops": ((t_op, o_op),),
        "world": True,  # world-space graft: pose driven per step
    }
    gripper_apply(driver)
    print(
        f"gripper grafted: Robotiq 2F-85 mounted ({hidden} finger meshes hidden)",
        flush=True,
    )
    return True


def gripper_apply(driver) -> None:
    """Pose the gripper from the palm every step (world-space graft).
    The Robotiq base frame points its fingers along local -Z with
    the palm coupling flange at the origin - mount it so -Z aligns
    with the palm-forward axis (a 180-deg X flip from the palm
    orientation)."""
    g = getattr(driver, "_gripper", None)
    if g is None:
        return
    import numpy as np

    from g1d_driver import n, quat_mul, quat_rot
    from pxr import Gf

    pos = n(driver.robot.data.body_pos_w)[0]
    quats = n(driver.robot.data.body_quat_w)[0]
    palm_p = np.array(pos[g["palm_i"]])
    palm_q = np.array(quats[g["palm_i"]])
    fwd = np.array(quat_rot(palm_q.tolist(), g["f_loc"].tolist()))
    base = palm_p + fwd * 0.05
    # gripper quat = palm quat * rot(X, pi)  (xyzw math)
    flip = (0.0, 1.0, 0.0, 0.0)
    q = quat_mul(palm_q.tolist(), flip)
    q_wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    (t_op, o_op) = g["ops"][0]
    t_op.Set(Gf.Vec3d(float(base[0]), float(base[1]), float(base[2])))
    o_op.Set(Gf.Quatf(*q_wxyz))


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
