"""G1D warehouse table-to-table pick-and-place demo.

A large warehouse hall (45x30 m) dressed from the Simple_Warehouse
kit, with TWELVE work tables arranged in the mir_isaacLab divider
pattern (six divider x positions x two rows flanking a center
corridor). Half the tables are stocked with a variety of objects
(plastic crate "basket", four card-box sizes, two totes), half are
empty delivery targets - 12 pick+place tasks per cycle.

The grasp is a KINEMATIC SNAP (vacuum): the arm swings out over the
slot, the object teleports to the palm (each object type grabs at
its own height - the crate at its HANDLE) and is rigidly carried.
Delivery drives to the target table and drops the object onto it.

Run inside the compose container with the GUI on the host desktop:

    ./run.sh            # tmux 'g1d_arena', GUI on DISPLAY=:1
    ./run.sh headless   # smoke run, no window
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hand-only-collision",
    action="store_true",
    help="strip arm colliders (cosmetic contact only - the grasp itself is a kinematic snap and never depends on it)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# IsaacLab 3.0: a GUI window requires the 'kit' visualizer explicitly -
# headless=False alone no longer opens one
app_launcher = AppLauncher(
    headless=args_cli.headless,
    **({} if args_cli.headless else {"visualizer": "kit"}),
)
simulation_app = app_launcher.app

"""Now the sim is alive - the split modules import isaaclab safely."""

import math
import torch

import carb
from g1d_config import (
    ARM_GRASP,
    ARM_HIGH,
    ARM_HOME,
    ARM_HOVER,
    ARM_LEVEL,
    CORRIDOR_LANE,
    FINGER_OPEN,
    OBJECT_CATALOG,
    OBJECT_NAMES,
    OBJECT_ORDER,
    SLOT_STANDOFF,
    STOCKED_TABLES,
    TABLE_POS,
    TABLE_THEMES,
    TABLE_TOP,
    TOTE_SIZE,
    table_slot,
)
from g1d_driver import G1DDriver, RenderStepper, n, reset_tote
from g1d_scene import G1DSceneCfg, rotate_props
from g1d_usd import (
    add_grip_material,
    arm_collision_hand_only,
    bind_object_materials,
    disable_agv_shell_collision,
    rigidify_objects,
    select_physx_variant,
    strip_arm_colliders,
)
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext


def hold_cmd(driver: G1DDriver, pose, fingers) -> torch.Tensor:
    """Full-joint command: left arm tucked, right arm at `pose`,
    fingers at `fingers` (used as the arm_hold for driving)."""
    cmd = driver._baseline()
    cmd[0, driver.arm_ids] = torch.as_tensor(pose, dtype=torch.float32, device=driver.device)
    cmd[0, driver.finger_ids] = torch.as_tensor(fingers, dtype=torch.float32, device=driver.device)
    return cmd


def build_tasks():
    """12 pick+place tasks: every stocked slot -> a round-robin slot
    on the empty tables (each empty table receives two objects)."""
    empty = [t for t in range(len(TABLE_POS)) if t not in STOCKED_TABLES]
    tasks = []
    k = 0
    for theme_i, ti in enumerate(STOCKED_TABLES):
        for slot in (0, 1):
            dst_table = empty[k % len(empty)]
            dst_slot = (k // len(empty)) % 2
            tasks.append({
                "obj": OBJECT_NAMES[k],
                "type": OBJECT_ORDER[k],
                "theme": TABLE_THEMES[theme_i][0],
                "src": table_slot(ti, slot),
                "dst": table_slot(dst_table, dst_slot),
            })
            k += 1
    return tasks


def corridor_transit(driver: G1DDriver, tx: float, lane: float = CORRIDOR_LANE) -> None:
    """Long transit along the center corridor to x=tx: westbound legs
    ride forward_line (axis -pi), eastbound legs drive_to. Both home
    y to the corridor lane; big lateral errors get a short second leg."""
    bx, by, yaw = driver.pose()
    if abs(tx - bx) < 0.3:
        return
    if tx < bx:
        driver.spin_to(math.pi, timeout=20.0)
        driver.forward_line(bx - tx, target_y=lane, speed=0.4, timeout=90.0)
    else:
        driver.spin_to(0.0, timeout=20.0)
        driver.drive_to(tx, target_y=lane, tol=0.10, timeout=90.0)


def approach_slot(driver: G1DDriver, sx: float, sy: float, arm_hold=None) -> bool:
    """Inner-side approach: the slot sits on the table's corridor-
    facing edge; the robot stays IN the corridor the whole time -
    transit along x to the slot's x, then face the table (+-pi/2)
    and creep to the standoff with cross-axis (x) homing on the
    slot. The base never enters the row bands, so the nose-on table
    wedges of the east-end standoffs are geometrically gone."""
    row_sign = 1.0 if sy >= 0 else -1.0
    standoff_y = sy - row_sign * SLOT_STANDOFF
    face = (math.pi / 2) * row_sign
    bx, by, yaw = driver.pose()
    # RETURN TO THE LANE first if too deep: corridor_transit's
    # early-return (x already close) ignores a bad y, and spinning
    # inside the row band clips the tabletop corners (seen: spin
    # wedged at -0.78 rad with the nose on the face)
    if abs(by) > 0.7:
        # NEVER spin at depth: the 0.86 m corner sweep cannot clear
        # the tabletop face from any reachable standoff (reach needs
        # base <= 0.97 from the slot line, spin safety needs >= 1.53)
        # - reverse out along the current heading instead (the robot
        # faces the table there, so backing = toward the corridor)
        driver.back_straight(abs(by) - CORRIDOR_LANE + 0.1, speed=0.20, arm_hold=arm_hold, timeout=25.0)
    # corridor transit to the slot x (lane biased toward the row so
    # the final creep is short)
    if abs(driver.pose()[0] - sx) > 0.4:
        corridor_transit(driver, sx, lane=CORRIDOR_LANE * row_sign)
    driver.spin_to(face, timeout=15.0)
    # creep to the standoff in small NO-OVERSHOOT nudges (a +0.25
    # overshoot rammed the nose into the tabletop face)
    driver.forward_line(
        max(0.35, abs(standoff_y - by) + 0.25),
        target_x=sx,
        speed=0.2,
        arm_hold=arm_hold,
        timeout=30.0,
        lat_cap=0.6,
    )
    driver.forward_line(0.15, target_x=sx, speed=0.12, arm_hold=arm_hold, timeout=15.0, lat_cap=0.6)
    driver.spin_to(face, timeout=8.0, tol=0.15)
    bx_fin, by_fin, _ = driver.pose()
    ok_xy = abs(by_fin - standoff_y) < 0.28 and abs(bx_fin - sx) < 0.30
    if not ok_xy:
        print(f"  approach ends at ({bx_fin:.2f},{by_fin:+.2f}) vs ({sx:.2f},{standoff_y:+.2f})", flush=True)
    return ok_xy


def exit_slot(driver: G1DDriver, arm_hold=None) -> None:
    """Back ALL THE WAY to the corridor lane: any spin below |y| 0.7
    clips the tabletop corners (0.86 m sweep vs 1.1 m face line), so
    the next heading change must happen lane-deep."""
    by = abs(driver.pose()[1])
    driver.back_straight(max(0.9, by - CORRIDOR_LANE + 0.15), speed=0.20, arm_hold=arm_hold, timeout=25.0)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"sim device: {device}", flush=True)
    sim = SimulationContext(SimulationCfg(device=device, dt=1 / 60))
    scene_cfg = G1DSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    rotate_props()
    add_grip_material()
    select_physx_variant("/World/envs/env_0/Robot")
    rigidify_objects()  # kit props ship no RigidBodyAPI on the root
    bind_object_materials()  # textured look for the generated primitives
    if not disable_agv_shell_collision():
        carb.log_warn("AGV_link_1 not found - shell collision stays on")
    # arm colliders OFF by default: the grasp/carry are kinematic and
    # fingertip hulls fighting the held box at low ride heights
    # tumbled it and jammed the base
    n_arm = strip_arm_colliders()
    print(f"arm colliders stripped: {n_arm}", flush=True)
    if args_cli.hand_only_collision:
        n_off = arm_collision_hand_only()
        print(f"hand-only collision: {n_off} arm colliders stripped", flush=True)

    sim.reset()
    print(f"joints: {len(scene['robot'].joint_names)}", flush=True)

    stepper = RenderStepper(headless=args_cli.headless)
    driver = G1DDriver(scene["robot"], sim, totes=[scene[nm] for nm in OBJECT_NAMES], step=stepper.step)
    x0, y0, yaw0 = driver.pose()
    print(f"root after reset: x={x0:.3f} y={y0:.3f} yaw={yaw0:.3f}", flush=True)

    # ArtVIP bodies compose at the STAGE ORIGIN regardless of any
    # spawn/surgery transform (seen: every ArtVIP prop reads (0,0)
    # after reset) - teleport each object onto its slot at runtime,
    # where the write provably takes (the re-seat teleports work)
    driver.run_seconds(0.2)  # warm the buffers first
    for _i, _nm in enumerate(OBJECT_NAMES):
        _sx, _sy = table_slot(STOCKED_TABLES[_i // 2], _i % 2)
        reset_tote(scene[_nm], (_sx, _sy, TABLE_TOP + 0.25))
    driver.run_seconds(0.3)

    # STARTUP, anti-tip order (user report: the robot TIPS OVER at
    # startup - a tall torso + raised arm on a narrow AGV, and any
    # spawn transient flips it; every later "mystery yaw torque" was
    # the tipped base flailing):
    #   1. braked settle at a LOW arm pose - wheels commanded to
    #      zero every step, arms folded near the body
    #   2. slow stepped raise to ARM_HIGH
    #   3. hard-park teleport LAST, then a braced hold (arm frozen
    #      at its measured pose + wheel brakes every step) so the
    #      teleport impulse has nothing to amplify
    braced = hold_cmd(driver, ARM_HOME.tolist(), FINGER_OPEN.tolist())

    def _brake() -> None:
        driver.robot.set_joint_position_target(braced)
        driver.wheels(0.0, 0.0)

    # TEMP: attitude telemetry through the startup (is the base
    # pitching/rolling/leaning?)
    def _att(tag):
        q = n(driver.robot.data.root_quat_w)[0]  # xyzw
        x, y, z, w = (float(v) for v in q)
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
        roll = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
        pz = float(n(driver.robot.data.root_pos_w)[0][2])
        print(f"  ATT {tag}: pitch={pitch:+.1f} roll={roll:+.1f} z={pz:.3f} pose={driver.pose()}", flush=True)

    for _s in range(6):
        driver.run_seconds(0.5, every_step=_brake)
        _att("settle")
    driver.move_arm_slow(ARM_HIGH.tolist(), step=0.05, hold=0.2, timeout=20.0)
    _att("raised")
    driver.fingers_to(FINGER_OPEN)

    # each object's SETTLED center height - used by the per-task
    # resets (kit prop origins/heights vary, so measure, don't guess)
    settle_z = {}
    for nm in OBJECT_NAMES:
        tp_s = n(scene[nm].data.root_pos_w)[0]
        settle_z[nm] = float(tp_s[2])
        print(f"  {nm} settled ({tp_s[0]:.2f},{tp_s[1]:+.2f},{tp_s[2]:.3f})", flush=True)

    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[-6.0, 5.0, 3.5],
            target=[-13.0, 0.0, 0.8],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"camera setup skipped: {exc}", flush=True)

    # NO TELEPORT PARK: write_root_pose_to_sim on the articulation
    # root is VIOLENT - the braced 1.5 s hold after it flung the
    # base 1.7 m sideways with yaw 0.85 (attitude telemetry showed
    # an upright frozen base right up to the teleport). The braced
    # settle above already parks the robot at the spawn upright, so
    # just brace a little longer and go.
    driver.run_seconds(0.5, every_step=_brake)
    print(f"ready at {driver.pose()}", flush=True)

    tasks = build_tasks()
    picks = 0
    placements = 0
    cycle = 0
    while simulation_app.is_running():
        cycle += 1
        print(f"=== cycle {cycle}: {len(tasks)} tasks ===", flush=True)

        for task in tasks:
            print(f"--- task {task['obj']} [{task['theme']}] ---", flush=True)
            t0_steps = stepper.steps
            obj = scene[task["obj"]]
            obj_type = task["type"]
            grab_up = OBJECT_CATALOG[obj_type][2]
            sx, sy = task["src"]
            dx_, dy_ = task["dst"]
            obj_h = TOTE_SIZE if OBJECT_CATALOG[obj_type][0] is None else 0.45

            driver.detach()  # nothing held between tasks
            reset_tote(obj, (sx, sy, settle_z[task["obj"]]))
            driver.run_seconds(0.2)
            driver.fingers_to(FINGER_OPEN, tolerance=0.5, timeout=3.0)

            # --- approach the source slot --------------------------
            if not approach_slot(driver, sx, sy):
                print(f"{task['obj']} ({obj_type}): approach FAILED", flush=True)
                exit_slot(driver)
                continue
            print(f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s at slot", flush=True)

            # knock recovery: the approach can shove an object off
            # its slot - re-seat it before arming the descent
            tp_chk = n(obj.data.root_pos_w)[0]
            if math.hypot(tp_chk[0] - sx, tp_chk[1] - sy) > 0.15 or abs(tp_chk[2] - settle_z[task["obj"]]) > 0.10:
                reset_tote(obj, (sx, sy, settle_z[task["obj"]]))
                # TEMP diag: does the write take effect at all?
                tp_w = n(obj.data.root_pos_w)[0]
                print(f"  RESET-WRITE readback: ({tp_w[0]:.2f},{tp_w[1]:+.2f},{tp_w[2]:.2f})", flush=True)
                driver.run_seconds(0.2)
                print("  (object re-seated after approach knock)", flush=True)

            # --- hover + proximity snap (grab at the object's own
            # anchor height - the crate grabs its HANDLE) -----------
            driver.move_arm_smooth(ARM_HOVER.tolist(), speed=0.5)
            driver.hold_arm(ARM_HOVER.tolist(), 0.2)
            driver.arm_capture(obj, grab_up=grab_up)
            for dip in (ARM_GRASP.tolist(),):
                if driver._attach is not None:
                    break
                driver.move_arm_slow(dip, step=0.05, hold=0.15, abort_attach=True)
            driver.disarm_capture()
            if driver._attach is None:
                hc = driver.hand_center()
                tp_f = n(obj.data.root_pos_w)[0]
                print(
                    f"{task['obj']} ({obj_type}): snap FAILED "
                    f"(jaw=({hc[0]:.2f},{hc[1]:+.2f},{hc[2]:.2f}) "
                    f"obj=({tp_f[0]:.2f},{tp_f[1]:+.2f},{tp_f[2]:.2f}) "
                    f"dist={math.dist(hc.tolist(), tp_f[:3].tolist()):.2f})",
                    flush=True,
                )
                driver.move_arm_smooth(ARM_HIGH.tolist(), speed=0.6)
                exit_slot(driver)
                continue
            picks += 1
            print(f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s snapped {obj_type}", flush=True)

            # --- lift (STEP-HOLDS while holding, never per-step
            # ramps: the smooth ramp loses authority once the object
            # pose write runs every step - see driver notes) --------
            pose = driver.right_arm_now()
            lift = list(pose)
            lift[0] -= 0.15
            driver.move_arm_slow(lift, step=0.05, hold=0.15)
            carry_hold = hold_cmd(driver, lift, FINGER_OPEN.tolist())

            # --- exit the gap and level the arm for the carry ------
            exit_slot(driver, arm_hold=carry_hold)
            driver.move_arm_slow(ARM_LEVEL.tolist(), step=0.08, hold=0.12, timeout=15.0)
            carry_hold = hold_cmd(driver, ARM_LEVEL.tolist(), FINGER_OPEN.tolist())

            # --- transit + approach the destination slot ----------
            corridor_transit(driver, dx_, lane=CORRIDOR_LANE * (1.0 if dy_ >= 0 else -1.0))
            if not approach_slot(driver, dx_, dy_, arm_hold=carry_hold):
                print(f"{task['obj']}: delivery approach FAILED - drop in corridor", flush=True)
                driver.detach()
                driver.run_seconds(1.0)
                driver.move_arm_smooth(ARM_HIGH.tolist(), speed=0.6)
                exit_slot(driver)
                continue
            print(f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s at dst", flush=True)

            # --- release over the table: the object drops ~0.3 m
            # onto the tabletop (arm colliders are stripped, so
            # nothing clips the fall) ------------------------------
            driver.detach()
            driver.run_seconds(1.0)
            tp = n(obj.data.root_pos_w)[0]
            on_table = abs(tp[2] - TABLE_TOP - obj_h / 2) < 0.15 and math.hypot(tp[0] - dx_, tp[1] - dy_) < 0.35
            placements += int(on_table)
            print(
                f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s dropped {obj_type} at "
                f"({tp[0]:.2f},{tp[1]:+.2f},{tp[2]:.2f}) "
                f"{'ON TABLE' if on_table else 'off!'} (slot {dx_:.2f},{dy_:+.2f})",
                flush=True,
            )

            # --- arm up, exit, next task --------------------------
            driver.fingers_to(FINGER_OPEN, tolerance=0.5, timeout=3.0)
            driver.move_arm_smooth(ARM_HIGH.tolist(), speed=0.6)
            exit_slot(driver)

        print(f"=== cycle {cycle}: {placements} placed / {picks} picked ===", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
