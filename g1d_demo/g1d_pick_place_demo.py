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
import numpy as np
import torch

import carb
from g1d_config import (
    ARM_DOCK,
    ARM_GRASP,
    ARM_HIGH,
    ARM_HOVER,
    ARM_LEVEL,
    FINGER_CLOSE,
    FINGER_CRADLE,
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
from g1d_usd import add_grip_material, arm_collision_hand_only, disable_agv_shell_collision, select_physx_variant
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
    # approach (user spec: raised from the start)
    driver.hold_arm(ARM_HIGH.tolist(), 0.5)
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
            if not driver.drive_to(pre_wall, target_y=ty, tol=0.05):
                print(f"{name}: drive to pre-wall FAILED", flush=True)
                continue
            if not driver.drive_to(standoff, target_y=ty, tol=0.03):
                print(f"{name}: drive to station FAILED", flush=True)
                continue
            driver.spin_to(0.0, timeout=6.0)
            print(f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s arrived", flush=True)

            # --- REDUCED descent + proximity snap -----------------
            # swing out OVER the slot, then a small elbow dip - the
            # snap radius normally fires partway down ARM_DOCK, and
            # ARM_GRASP is only a drape fallback
            driver.move_arm_smooth(ARM_HOVER.tolist(), speed=0.6)
            driver.hold_arm(ARM_HOVER.tolist(), 0.2)
            driver.arm_capture(tote)
            pose = ARM_HOVER.tolist()
            for elbow in (float(ARM_DOCK[3]), float(ARM_GRASP[3])):
                if driver._attach is not None:
                    break
                pose[3] = elbow
                driver.move_arm_slow(pose, step=0.05, hold=0.15)
            driver.disarm_capture()
            if driver._attach is None:
                print(f"{name}: snap FAILED - skipped", flush=True)
                driver.move_arm_smooth(ARM_HIGH.tolist(), speed=0.6)
                continue
            picks += 1
            print(f"  t+{(stepper.steps - t0_steps) / 60.0:.0f}s snapped", flush=True)

            # --- wrap + visible lift ------------------------------
            cradle = (FINGER_CLOSE * FINGER_CRADLE).tolist()
            wrap = hold_cmd(driver, pose, cradle)
            driver.run_seconds(
                0.35,
                every_step=lambda: (driver.robot.set_joint_position_target(wrap)),
            )
            lift = list(pose)
            lift[0] -= 0.10  # rise 10 cm before any base motion
            driver.move_arm_smooth(lift, speed=0.5)
            carry_hold = hold_cmd(driver, lift, cradle)

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
            driver.move_arm_smooth(ARM_LEVEL.tolist(), speed=0.6, timeout=15.0)
            carry_hold = hold_cmd(driver, ARM_LEVEL.tolist(), cradle)
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
            # no explicit return leg: the next tote's approach
            # drive_to covers the aisle run

        print(f"=== cycle {cycle}: {placements} placed / {picks} picked ===", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
