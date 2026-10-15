"""G1D warehouse pick-and-place demo inside IsaacLab-Arena's Isaac Lab.

A Mir-style digital-twin warehouse built from the Simple_Warehouse
asset kit: perimeter walls, pallet racking flanking the aisle, floor
decals, pallets and card boxes. The wheeled G1D shuttles along the
aisle and services every tote on the replenishment station in a loop.

Per user spec, the grasp is a KINEMATIC SNAP (vacuum): the arm rides
UP from the very start, dips only a little over the slot, and the
moment the jaw comes near the tote the tote TELEPORTS onto the hand
and is rigidly written there every step - no physical pinch at all.
The carry then levels the arm (平举), U-turns 180 deg, walks FORWARD
to the pad slot, and PLACES the tote physically: detach and let it
fall. Finally it turns back around and returns for the next tote.

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
    ARM_DOCK,
    ARM_GRASP,
    ARM_HIGH,
    ARM_HOVER,
    ARM_LEVEL,
    FINGER_OPEN,
    PAD_SLOTS,
    PAD_SPAN,
    PICK_TARGETS,
    STANDOFF_CLEAR,
    TOTE_NAMES,
    TOTE_Z,
)
from g1d_driver import G1DDriver, RenderStepper, n, reset_tote
from g1d_scene import G1DSceneCfg, rotate_props
from g1d_usd import (
    add_grip_material,
    arm_collision_hand_only,
    disable_agv_shell_collision,
    select_physx_variant,
    strip_hand_colliders,
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"sim device: {device}", flush=True)
    sim = SimulationContext(SimulationCfg(device=device, dt=1 / 60))
    scene_cfg = G1DSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    rotate_props()
    add_grip_material()
    select_physx_variant("/World/envs/env_0/Robot")
    if not disable_agv_shell_collision():
        carb.log_warn("AGV_link_1 not found - shell collision stays on")
    # hand colliders OFF by default: the grasp/carry are kinematic
    # and fingertip hulls fighting the held box at low ride
    # heights tumbled it and jammed the base
    n_hand = strip_hand_colliders()
    print(f"hand colliders stripped: {n_hand}", flush=True)
    if args_cli.hand_only_collision:
        n_off = arm_collision_hand_only()
        print(f"hand-only collision: {n_off} arm colliders stripped", flush=True)

    sim.reset()
    print(f"joints: {len(scene['robot'].joint_names)}", flush=True)

    stepper = RenderStepper(headless=args_cli.headless)
    driver = G1DDriver(scene["robot"], sim, totes=[scene[nm] for nm in TOTE_NAMES], step=stepper.step)
    x0, y0, yaw0 = driver.pose()
    print(f"root after reset: x={x0:.3f} y={y0:.3f} yaw={yaw0:.3f}", flush=True)
    for _ in range(120):  # 2 s settle
        stepper.step(sim)
        scene["robot"].update(sim.get_physics_dt())
        for nm in TOTE_NAMES:
            scene[nm].update(sim.get_physics_dt())

    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[4.8, 3.6, 2.6],
            target=[1.2, 0.0, 0.8],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"camera setup skipped: {exc}", flush=True)

    # arm UP before the first wheel turns - it stays up for every
    # approach (user spec: raised from the start). Smooth ramp: a
    # raw hold_arm here snapped the 1.45-rad shoulder swing shut
    # in ~0.3 s (one of the "finishes too fast" swings)
    driver.move_arm_smooth(ARM_HIGH.tolist(), speed=0.5, timeout=8.0)
    driver.hold_arm(ARM_HIGH.tolist(), 0.2)
    driver.fingers_to(FINGER_OPEN)

    picks = 0
    placements = 0
    cycle = 0
    while simulation_app.is_running():
        cycle += 1
        print(f"=== cycle {cycle} ===", flush=True)

        for name, tx, ty, _color in PICK_TARGETS:
            t0_steps = stepper.steps
            tote = scene[name]
            slot = (tx, ty, TOTE_Z)
            driver.detach()  # nothing held between totes
            reset_tote(tote, slot)  # back from the pad / any knock
            driver.run_seconds(0.2)
            driver.fingers_to(FINGER_OPEN, tolerance=0.5, timeout=3.0)

            # --- approach: two-phase drive, arm stays raised -----
            standoff = min(tx - 0.40, STANDOFF_CLEAR)
            pre_wall = standoff - 0.25
            # extreme slot-to-slot jumps need up to ~0.9 m of y -
            # more than one tracker leg can close - so add an early
            # runway waypoint when far off the slot line
            _, by_pre, _ = driver.pose()
            if abs(by_pre - ty) > 0.25:
                driver.drive_to(pre_wall - 0.90, target_y=ty, tol=0.08, timeout=45.0, lat_cap=0.55)
            # converge y AT THE PRE-WALL (an angled nose still
            # clears the station face here), then square up and
            # creep the last 0.25 m dead-straight - any yaw at the
            # standoff digs a nose corner into the face
            if not driver.drive_to(pre_wall, target_y=ty, tol=0.05, lat_cap=0.55):
                print(f"{name}: drive to pre-wall FAILED", flush=True)
                continue
            driver.spin_to(0.0, timeout=8.0)
            if not driver.drive_to(standoff, target_y=ty, tol=0.03):
                print(f"{name}: drive to station FAILED", flush=True)
                continue
            driver.spin_to(0.0, timeout=6.0)
            # the snap radius covers ~0.20 m of lateral - one
            # runway retry if y is still wider than that, else
            # let the tote go to the next cycle
            for _trim in range(1):
                _, by_now, _ = driver.pose()
                if abs(by_now - ty) < 0.18:
                    break
                print(f"  y-trim runway: {by_now:+.2f} -> {ty:+.2f}", flush=True)
                driver.back_straight(0.55, speed=0.12)
                driver.spin_to(0.0, timeout=8.0)
                driver.drive_to(pre_wall, target_y=ty, tol=0.05, lat_cap=0.55, timeout=30.0)
                driver.spin_to(0.0, timeout=8.0)
                driver.drive_to(standoff, target_y=ty, tol=0.03, timeout=15.0)
                driver.spin_to(0.0, timeout=6.0)
            print(f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s arrived", flush=True)

            # knock recovery: a yawed approach can sweep a tote off
            # its post (tote_d twice found ~0.4 m away on the
            # floor) - re-seat it before arming the descent
            tp_chk = n(tote.data.root_pos_w)[0]
            if abs(tp_chk[1] - ty) > 0.10 or abs(tp_chk[0] - tx) > 0.12 or abs(tp_chk[2] - TOTE_Z) > 0.05:
                reset_tote(tote, slot)
                driver.run_seconds(0.2)
                print("  (tote re-seated after approach knock)", flush=True)

            # --- REDUCED descent + proximity snap -----------------
            # swing out OVER the slot, then a small elbow dip - the
            # snap radius normally fires partway down ARM_DOCK, and
            # ARM_GRASP is only a drape fallback
            driver.move_arm_smooth(ARM_HOVER.tolist(), speed=0.5)
            driver.hold_arm(ARM_HOVER.tolist(), 0.2)
            driver.arm_capture(tote)
            # small SHOULDER dips only, and the ramp ABORTS the
            # instant the snap fires - the arm never presses onto
            # the tote/post (that press was the pickup jitter)
            for dip in (ARM_DOCK.tolist(), ARM_GRASP.tolist()):
                if driver._attach is not None:
                    break
                driver.move_arm_slow(dip, step=0.05, hold=0.15, abort_attach=True)
            driver.disarm_capture()
            if driver._attach is None:
                hc = driver.hand_center()
                tp_f = n(tote.data.root_pos_w)[0]
                print(
                    f"{name}: snap FAILED - skipped "
                    f"(jaw=({hc[0]:.2f},{hc[1]:+.2f},{hc[2]:.2f}) "
                    f"tote=({tp_f[0]:.2f},{tp_f[1]:+.2f},{tp_f[2]:.2f}) "
                    f"dist={math.dist(hc.tolist(), tp_f[:3].tolist()):.2f})",
                    flush=True,
                )
                driver.move_arm_smooth(ARM_HIGH.tolist(), speed=0.6)
                continue
            picks += 1
            print(f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s snapped", flush=True)

            # --- lift clear, fingers stay OPEN ----------------------
            # freeze the arm where the snap fired (mid-dip) and
            # rise off the post. NO finger curl while holding: the
            # box rides AT the fingertip line now, and curling the
            # fingers into the kinematic box is a contact fight
            # that deflected the arm and scattered the drops - the
            # box sits on the open palm like a tray instead
            pose = driver.right_arm_now()
            lift = list(pose)
            lift[0] -= 0.15  # rise ~12 cm before any base motion
            # gentle: at speed 0.6 the small 0.15-rad pop completed
            # in ~0.25 s - visibly abrupt (another too-fast swing)
            driver.move_arm_smooth(lift, speed=0.35)
            carry_hold = hold_cmd(driver, lift, FINGER_OPEN.tolist())

            # --- clear the station, U-turn 180 deg ----------------
            # spinning at the standoff grinds the nose on the
            # station face - back off first (probe-proven)
            driver.back_straight(0.35, speed=0.10, arm_hold=carry_hold)
            ok = driver.spin_to(math.pi, timeout=15.0, tol=0.15)
            if not ok:
                driver.back_straight(0.25, speed=0.10, arm_hold=carry_hold)
                ok = driver.spin_to(math.pi, timeout=15.0, tol=0.15)
            print(
                f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s u-turn (ok={int(ok)} yaw={driver.pose()[2]:+.2f})",
                flush=True,
            )

            # --- level the arm (平举) + forward delivery ----------
            # leveling AFTER the spin: a level arm sweeps the held
            # tote through the station board / neighboring totes
            driver.move_arm_smooth(ARM_LEVEL.tolist(), speed=0.5, timeout=15.0)
            carry_hold = hold_cmd(driver, ARM_LEVEL.tolist(), FINGER_OPEN.tolist())
            # settle the arm before measuring the box lead - the
            # level swing changes it mid-motion (0.5 -> 0.67 m and
            # the stop line overshot by just that delta)
            driver.run_seconds(
                0.5,
                every_step=lambda: (driver.robot.set_joint_position_target(carry_hold)),
            )
            slot_px, slot_py = PAD_SLOTS[name]
            for _leg in range(3):
                tp = n(tote.data.root_pos_w)[0]
                d_err = math.hypot(tp[0] - slot_px, tp[1] - slot_py)
                print(
                    f"  leg {_leg}: base={driver.pose()[0]:.2f} "
                    f"box=({tp[0]:.2f},{tp[1]:+.2f},{tp[2]:.2f}) "
                    f"slot-delta {d_err:.2f}",
                    flush=True,
                )
                if d_err < 0.10:
                    break
                bx_p, by_p, byaw_p = driver.pose()
                if byaw_p > 0.0:  # must face -x for the run
                    driver.spin_to(math.pi, timeout=8.0, tol=0.25)
                # reach = box lead ahead of the base (facing -x);
                # stop the BASE so the HELD box sits over the slot
                reach = max(0.2, bx_p - tp[0])
                driver.forward_line(bx_p - (slot_px + reach), target_y=slot_py, speed=0.25, arm_hold=carry_hold)

            # --- physical placement: release and let it fall ------
            # (tried backing the base off / retracting the arm to
            # keep the falling box clear of the fingertips - both
            # caused worse failures: the arm sweep bats the box a
            # half-metre long, the reverse parks the rear corner on
            # the fresh drop and jams every later spin. The plain
            # drop lands 4/5 on-pad; take that)
            driver.detach()
            driver.run_seconds(1.0)
            tp = n(tote.data.root_pos_w)[0]
            (x_lo, x_hi), (y_lo, y_hi) = PAD_SPAN
            on_pad = x_lo <= tp[0] <= x_hi and y_lo <= tp[1] <= y_hi
            placements += int(on_pad)
            print(
                f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s "
                f"dropped at ({tp[0]:.2f},{tp[1]:+.2f},{tp[2]:.2f}) "
                f"{'ON PAD' if on_pad else 'off pad!'} "
                f"(slot {slot_px},{slot_py:+.2f})",
                flush=True,
            )

            # --- open up, turn back, arm up, return ---------------
            driver.fingers_to(FINGER_OPEN, tolerance=0.5, timeout=3.0)
            driver.move_arm_smooth(ARM_HIGH.tolist(), speed=0.6)
            driver.spin_to(0.0, timeout=15.0)
            # CORRIDOR RESET: approaching the next tote straight
            # from the drop position threads the base between the
            # freshly dropped totes and it eventually wedges (every
            # later spin silently times out) - return to the aisle
            # centerline first, then approach with a clean runway
            driver.drive_to(0.5, target_y=0.0, spin_first=False, tol=0.06, timeout=40.0)

        print(f"=== cycle {cycle}: {placements} placed / {picks} picked ===", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
