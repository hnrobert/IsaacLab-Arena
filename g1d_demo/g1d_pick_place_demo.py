"""G1D warehouse pick-and-place demo inside IsaacLab-Arena's Isaac Lab.

A Mir-style digital-twin warehouse built from the Simple_Warehouse
asset kit: perimeter walls, pallet racking flanking the aisle, floor
decals, pallets and card boxes. The wheeled G1D shuttles along the
aisle and services every tote on the replenishment station in a loop.

The grasp is CLOSED-LOOP: simulation ground truth (hand link poses +
tote pose) feeds alignment trims before the fingers close. Feedback
signals, all printed per attempt:

  * finger pad geometry vs the tote (pre-close line)
  * finger stall (a hit stops the curl short of the command)
  * tote world z after the lift (ground-truth success check)

Run inside the compose container with the GUI on the host desktop:

    ./run.sh            # tmux 'g1d_arena', GUI on DISPLAY=:1
    ./run.sh headless   # smoke run, no window
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hand-only-collision", action="store_true",
    help="strip arm colliders (worse grasping: the shoulder contact "
         "is load-bearing for the grasp equilibrium - 89pct carry "
         "WITH it vs ~40pct without)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# IsaacLab 3.0: a GUI window requires the 'kit' visualizer explicitly -
# headless=False alone no longer opens one
app_launcher = AppLauncher(
    headless=args_cli.headless,
    **({} if args_cli.headless else {"visualizer": "kit"}),
)
simulation_app = app_launcher.app

"""Now the sim is alive."""

"""Now the sim is alive - the split modules import isaaclab safely."""

import carb
import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

from g1d_config import (ARM_GRASP, ARM_HIGH, ARM_RELEASE, ARM_TUCK,
                        FINGER_CLOSE, FINGER_OPEN, PAD_SLOTS,
                        PICK_TARGETS, RIGHT_ARM, STANDOFF_CLEAR,
                        TOTE_NAMES, TOTE_SIZE, TOTE_Z)
from g1d_driver import G1DDriver, RenderStepper, n, reset_tote
from g1d_scene import G1DSceneCfg, rotate_props
from g1d_usd import (add_grip_material, arm_collision_hand_only,
                     disable_agv_shell_collision,
                     select_physx_variant)


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
        print(f"hand-only collision: {n_off} arm colliders stripped",
              flush=True)

    sim.reset()
    print(f"joints: {len(scene['robot'].joint_names)}", flush=True)

    stepper = RenderStepper(headless=args_cli.headless)
    driver = G1DDriver(
        scene["robot"], sim,
        totes=[scene[nm] for nm in TOTE_NAMES],
        step=stepper.step)
    x0, y0, yaw0 = driver.pose()
    print(f"root after reset: x={x0:.3f} y={y0:.3f} yaw={yaw0:.3f}",
          flush=True)
    for _ in range(120):  # 2 s settle
        stepper.step(sim)
        scene["robot"].update(sim.get_physics_dt())
        for nm in TOTE_NAMES:
            scene[nm].update(sim.get_physics_dt())

    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[4.8, 3.6, 2.6], target=[1.2, 0.0, 0.8],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"camera setup skipped: {exc}", flush=True)

    # (home warm-up removed: the STATION warm-up below is the
    # effective drape normalizer - this chain only added ~25 s of
    # sim time before the first drive)

    slots = {nm: (x, y, TOTE_Z) for nm, x, y, _ in PICK_TARGETS}
    grasps_ok = 0
    grasps_total = 0
    carry_streak = 0
    carry_total = 0
    cycle = 0
    while simulation_app.is_running():
        cycle += 1
        print(f"=== cycle {cycle} ===", flush=True)

        for name, tx, ty, _color in PICK_TARGETS:
            tote = scene[name]
            slot = (tx, ty, TOTE_Z)
            reset_tote(tote, slot)
            driver.run_seconds(0.5)
            # tuck TIGHT before driving: a half-swung arm is exactly
            # the 0.45 m pole that rams the station face
            driver.move_arm_slow(ARM_TUCK.tolist(), step=0.08,
                                 hold=0.10)
            driver.hold_arm(ARM_TUCK.tolist(), 0.2)
            driver.fingers_to(FINGER_OPEN)

            standoff = min(tx - 0.40, STANDOFF_CLEAR)
            # two-phase approach: the jaw zone is only ~4 cm wide, so
            # align the line BEFORE the wall zone (the nose jams on
            # the station edge near the standoff and a jammed shunt
            # cannot recover y) - then creep in already centered.
            pre_wall = standoff - 0.25
            if not driver.drive_to(pre_wall, target_y=ty, tol=0.05):
                print(f"{name}: drive to pre-wall FAILED", flush=True)
                continue
            driver.shunt_lateral(ty, tol=0.03)
            if not driver.drive_to(standoff, target_y=ty, tol=0.02):
                print(f"{name}: drive to station FAILED", flush=True)
                continue

            if cycle == 1 or carry_streak == 0:
                # station warm-up: rehearse the whole descent chain
                # once ON the station apron, then re-seat the tote.
                # Warm-up'd attempts carried 4/4 (d7-d10); without it
                # the drape roll is a coin flip (~2/6). Running it
                # after every failure resets the streak attempts to
                # the deterministic configuration.
                driver.fingers_to(FINGER_OPEN)
                driver.move_arm_slow(ARM_HIGH.tolist(), step=0.08,
                                     hold=0.10)
                driver.move_arm_slow(ARM_GRASP.tolist(), step=0.05,
                                     hold=0.2)
                driver.hold_arm(ARM_GRASP.tolist(), 0.5)
                driver.move_arm_slow(ARM_HIGH.tolist(), step=0.08,
                                     hold=0.10)
                reset_tote(tote, slot)
                driver.run_seconds(0.3)

            # slow gateway up, then a fine-stepped descent
            driver.move_arm_slow(ARM_HIGH.tolist(), step=0.08,
                                 hold=0.10)
            driver.hold_arm(ARM_HIGH.tolist(), 0.3)
            driver.move_arm_slow(ARM_GRASP.tolist(), step=0.05,
                                 hold=0.2)
            driver.hold_arm(ARM_GRASP.tolist(), 1.0)
            pose = ARM_GRASP.tolist()

            # (lateral alignment: three arm-yaw trim attempts all
            # failed - under contact drape the shoulder yaw has ~no
            # authority over the jaw center (yaw 0->+0.35, offset
            # unchanged). The micro-carry loop handles bad bites by
            # physical test instead.)
            # depth trim: seat the thumb tip 3 cm into the tote's
            # vertical band via pitch (z coupling is the strong axis);
            # iterate - the drape offsets exceed one clipped step
            # (seen: tips hovering 3-10 cm over the tote top)
            for _z in range(2):
                pos_v = n(driver.robot.data.body_pos_w)[0]
                names_v = driver.robot.data.body_names
                th_v = pos_v[names_v.index("right_hand_thumb_2_link")]
                dz = (TOTE_Z + 0.01) - th_v[2]
                if abs(dz) <= 0.02:
                    break
                dpt = float(np.clip(dz / 0.378, -0.20, 0.20))
                driver.hold_arm([pose[0] - dpt] + pose[1:], 0.6)
                pose[0] -= dpt

            # base-yaw gate: arm-vs-station contact during the
            # descent yaws the free-castering base (+0.51 rad seen),
            # shifts the whole jaw off the tote and can hook the hand
            # under the board edge - the reverse then jams with the
            # wheels spinning (d17 attempt 5). If the heading is
            # compromised, lift, re-align, and re-descend.
            for _gate in range(3):
                _, _, byaw = driver.pose()
                tp_g = n(tote.data.root_pos_w)[0]
                q_g = n(tote.data.root_quat_w)[0]
                tilt_g = 2.0 * math.acos(max(-1.0, min(1.0, q_g[3])))
                knocked = (abs(tp_g[1] - ty) > 0.06
                           or abs(tp_g[0] - tx) > 0.08
                           or tilt_g > 0.25)
                if abs(byaw) < 0.20 and not knocked:
                    break
                why = (f"yaw {byaw:+.2f}" if abs(byaw) >= 0.20
                       else "tote knocked off slot")
                if knocked:
                    reset_tote(tote, slot)
                    driver.run_seconds(0.4)
                print(f"  descent gate: {why} -> redo descent",
                      flush=True)
                pos_o = n(driver.robot.data.body_pos_w)[0]
                nm_o = driver.robot.data.body_names
                th_o = pos_o[nm_o.index("right_hand_thumb_2_link")]
                pr_o = (pos_o[nm_o.index("right_hand_index_1_link")]
                        + pos_o[nm_o.index(
                            "right_hand_middle_1_link")]) / 2.0
                tp_o = n(tote.data.root_pos_w)[0]
                off = (th_o[1] + pr_o[1]) / 2.0 - tp_o[1]
                ty_eff = float(np.clip(ty - 0.8 * off,
                                       ty - 0.12, ty + 0.12))
                driver.move_arm_slow(ARM_HIGH.tolist(), step=0.06)
                driver.spin_to(0.0, timeout=10.0)
                driver.drive_to(standoff, target_y=ty_eff, tol=0.03)
                driver.move_arm_slow(ARM_HIGH.tolist())
                driver.move_arm_slow(ARM_GRASP.tolist(), step=0.05,
                                     hold=0.2)
                driver.hold_arm(ARM_GRASP.tolist(), 1.0)
                pose = ARM_GRASP.tolist()

            # pre-close geometry: jaw span vs tote width, one line
            tp = n(tote.data.root_pos_w)[0]
            pos_b = n(driver.robot.data.body_pos_w)[0]
            names_b = driver.robot.data.body_names
            th = pos_b[names_b.index("right_hand_thumb_2_link")]
            pr = (pos_b[names_b.index("right_hand_index_1_link")]
                  + pos_b[names_b.index("right_hand_middle_1_link")]) / 2.0
            jp_d = n(driver.robot.data.joint_pos)[0]
            arm_act = [round(jp_d[driver.idx[nm]].item(), 2)
                       for nm in RIGHT_ARM]
            bx, by, byaw = driver.pose()
            print(f"  diag: base=({bx:.2f},{by:+.2f},{byaw:+.2f}) "
                  f"arm={arm_act}", flush=True)
            for _nm in ("right_hand_thumb_2_link",
                        "right_hand_middle_1_link",
                        "right_hand_index_1_link"):
                _q = pos_b[names_b.index(_nm)]
                print(f"  tip {_nm.replace('right_hand_', '')}: "
                      f"({_q[0]:.3f},{_q[1]:+.3f},{_q[2]:.3f})",
                      flush=True)
            print(
                f"pre-close {name}: thumb_y={th[1]:+.3f} "
                f"pair_y={pr[1]:+.3f} tote_y={tp[1]:+.3f} "
                f"span={abs(th[1] - pr[1]) * 100:.0f}cm",
                flush=True,
            )

            # --- close + MICRO-CARRY acceptance ------------------
            # Stall values proved unreliable as catch-quality
            # predictors (a free close at 0.6 reaches 0.54-0.56 and
            # fakes depth; flat/deep thresholds rejected real catches
            # while deeper bites hooked the board edge and jammed the
            # base). Accept on physical EVIDENCE (tote moved), then
            # test the grip DIRECTLY: reverse 5 cm - the tote must
            # follow. Slip or base-jam -> re-settle and bite again.
            caught = False
            carry_hold = None
            tx0 = None
            bx0 = None
            tote_pre = n(tote.data.root_pos_w)[0][:3]
            for settle in range(4):
                if settle:
                    print(f"  re-settle {settle}/3", flush=True)
                    # a wild round can knock the tote off its slot
                    # (seen: swept 87 cm sideways) - put it back
                    tp_r = n(tote.data.root_pos_w)[0]
                    q_r = n(tote.data.root_quat_w)[0]
                    tilt_r = 2.0 * math.acos(
                        max(-1.0, min(1.0, q_r[3])))
                    if (abs(tp_r[1] - ty) > 0.06
                            or abs(tp_r[0] - tx) > 0.08
                            or tilt_r > 0.25):
                        reset_tote(tote, slot)
                        driver.run_seconds(0.4)
                        print("  (tote re-seated after knock)",
                              flush=True)
                    # vary the bite a little per round - identical
                    # re-descents replay the identical missing
                    # geometry, but too deep hooks the board edge
                    pose[5] = -0.50 - 0.04 * settle
                    # BASE-SIDE drape compensation: on the off-center
                    # slots the arm hangs 5-7 cm past the tote; the
                    # arm cannot be trimmed, but the BASE aims true -
                    # steer the standoff target by the measured jaw
                    # offset so the draped jaw lands on the tote
                    pos_o = n(driver.robot.data.body_pos_w)[0]
                    nm_o = driver.robot.data.body_names
                    th_o = pos_o[nm_o.index("right_hand_thumb_2_link")]
                    pr_o = (pos_o[nm_o.index(
                        "right_hand_index_1_link")]
                        + pos_o[nm_o.index(
                            "right_hand_middle_1_link")]) / 2.0
                    tp_o = n(tote.data.root_pos_w)[0]
                    off = (th_o[1] + pr_o[1]) / 2.0 - tp_o[1]
                    ty_eff = float(np.clip(ty - 0.8 * off,
                                           ty - 0.12, ty + 0.12))
                    if abs(off) > 0.03:
                        print(f"  base-yaw... drape off={off:+.3f} "
                              f"-> aim y {ty_eff:+.3f}", flush=True)
                    # lift, restore the standoff (a slipped round
                    # sits 5 cm back; a jammed one needs the unhook)
                    # and drop a fresh drape
                    driver.move_arm_slow(ARM_HIGH.tolist(), step=0.06)
                    driver.hold_arm(ARM_HIGH.tolist(), 0.4)
                    driver.drive_to(standoff, target_y=ty_eff,
                                    tol=0.03, timeout=30.0)
                    driver.move_arm_slow(ARM_HIGH.tolist())
                    driver.move_arm_slow(pose, step=0.05, hold=0.2)
                    driver.hold_arm(pose, 0.8)
                    # re-run the z trim: the fresh drape can again
                    # leave the tips hovering over the tote top
                    for _z in range(2):
                        pos_v = n(driver.robot.data.body_pos_w)[0]
                        names_v = driver.robot.data.body_names
                        th_v = pos_v[names_v.index(
                            "right_hand_thumb_2_link")]
                        dz = (TOTE_Z + 0.01) - th_v[2]
                        if abs(dz) <= 0.02:
                            break
                        dpt = float(np.clip(dz / 0.378, -0.20, 0.20))
                        driver.hold_arm(
                            [pose[0] - dpt] + pose[1:], 0.6)
                        pose[0] -= dpt
                    tote_pre = n(tote.data.root_pos_w)[0][:3]
                caught_round = False
                for frac in (0.4, 0.6, 0.8):
                    close_cmd = driver._baseline()
                    close_cmd[0, driver.finger_ids] = (
                        FINGER_CLOSE * frac).to(driver.device)
                    close_cmd[0, driver.arm_ids] = torch.as_tensor(
                        pose, dtype=torch.float32, device=driver.device)
                    driver.run_seconds(
                        1.2,
                        every_step=lambda: (
                            driver.robot.set_joint_position_target(close_cmd)),
                    )
                    fm = driver.finger_max()
                    tp_c = n(tote.data.root_pos_w)[0][:3]
                    moved = float(np.linalg.norm(
                        tp_c[:2] - tote_pre[:2])) * 100
                    dz_c = abs(tp_c[2] - tote_pre[2]) * 100
                    pos_g = n(driver.robot.data.body_pos_w)[0]
                    nm_g = driver.robot.data.body_names
                    th_g = pos_g[nm_g.index("right_hand_thumb_2_link")]
                    pr_g = ((pos_g[nm_g.index("right_hand_index_1_link")]
                             + pos_g[nm_g.index(
                                 "right_hand_middle_1_link")]) / 2.0)
                    near = (abs(th_g[1] - tp_c[1]) < 0.08
                            and abs(pr_g[1] - tp_c[1]) < 0.10)
                    got = (0.30 < fm < 0.85
                           and (moved >= 1.0 or dz_c >= 0.5
                                or (fm >= 0.50 and near)))
                    print(f"  close {frac:.2f}: finger_max={fm:.2f} "
                          f"tote_moved={moved:.1f}cm dz={dz_c:.1f}cm "
                          f"near={int(near)} "
                          f"{'HOLDING' if got else '-'}", flush=True)
                    if got:
                        caught_round = True
                        break
                    if fm > 0.78:
                        # jaw closed on AIR (no stall): deeper targets
                        # just make a fist. Abandon the ladder and
                        # re-settle instead.
                        break
                if not caught_round:
                    continue

                # evidence catch -> MICRO-CARRY: the criterion in
                # miniature, at 1/7 the cost of a failed carry
                micro_hold = driver._baseline()
                micro_hold[0, driver.arm_ids] = torch.as_tensor(
                    pose, dtype=torch.float32, device=driver.device)
                micro_hold[0, driver.finger_ids] = (
                    FINGER_CLOSE * 0.6).to(driver.device)
                if tx0 is None:
                    tx0 = n(tote.data.root_pos_w)[0][0]
                    bx0 = driver.pose()[0]
                bx_m0 = driver.pose()[0]
                tx_m0 = n(tote.data.root_pos_w)[0][0]
                driver.back_straight(0.05, speed=0.04,
                                     arm_hold=micro_hold)
                bx_moved = bx_m0 - driver.pose()[0]
                tx_moved = tx_m0 - n(tote.data.root_pos_w)[0][0]
                ratio = tx_moved / bx_moved if bx_moved > 1e-6 else 0.0
                print(f"  micro-carry: base {bx_moved * 100:.1f}cm "
                      f"tote {tx_moved * 100:.1f}cm "
                      f"({ratio * 100:.0f}%)", flush=True)
                if bx_moved < 0.03:
                    # base jammed - the hand is hooked; the re-settle
                    # arm-lift + drive restores it
                    print("  micro-carry JAM -> unhook and retry",
                          flush=True)
                    continue
                if tx_moved >= 0.02:
                    caught = True
                    carry_hold = micro_hold
                    break
                print("  micro-carry slipped -> retry", flush=True)
            driver._last_right_pose = list(pose)
            grasps_total += 1

            # THE criterion: continue reversing to ~0.37 m total from
            # the pre-micro position (yaw-locked, arm+fingers held) -
            # the d7-d10 carry mechanism; legged stop-start reverses
            # tested worse.
            if tx0 is None:
                tx0 = n(tote.data.root_pos_w)[0][0]
                bx0 = driver.pose()[0]
            if caught:
                # flat continuous pull - the d7-d10/d18 mechanism
                driver.back_straight(0.32, speed=0.05,
                                     arm_hold=carry_hold)
            else:
                fallback = driver._baseline()
                fallback[0, driver.arm_ids] = torch.as_tensor(
                    pose, dtype=torch.float32, device=driver.device)
                fallback[0, driver.finger_ids] = (
                    FINGER_CLOSE * 0.6).to(driver.device)
                driver.back_straight(0.35, speed=0.05,
                                     arm_hold=fallback)
            tx1 = n(tote.data.root_pos_w)[0]
            bx1 = driver.pose()[0]
            carried = (tx0 - tx1[0]) > 0.15
            carry_total += 1
            carry_streak = carry_streak + 1 if carried else 0
            print(
                f"CARRY TEST {name}: base {bx0:.2f}->{bx1:.2f} "
                f"tote {tx0:.2f}->{tx1[0]:.2f} "
                f"-> {'CARRIED' if carried else 'dropped'} "
                f"[streak {carry_streak}/5, {carry_total} tries]",
                flush=True,
            )
            got = carried

            # clear the aisle BEFORE returning: the drag leaves the
            # tote in the base's own path (~1.6-2.0 m) and the return
            # drive bulldozes it, shunting the base off-line - the
            # next approach then arrives y-off and only jaw-edges the
            # tote. Carried -> pad slot (delivered); dropped -> back
            # to the station slot for the next attempt.
            driver.fingers_to(FINGER_OPEN, tolerance=0.5, timeout=3.0)
            if carried:
                slot_x, slot_y = PAD_SLOTS[name]
                reset_tote(tote, (slot_x, slot_y,
                                  TOTE_SIZE / 2 + 0.005))
                grasps_ok += 1
                print(f"drag-delivered {name} (reset to pad)",
                      flush=True)
            else:
                reset_tote(tote, slot)
                print(f"re-seated {name} on the station", flush=True)
            driver.run_seconds(0.3)
            driver.arm_to(ARM_HIGH, tolerance=0.25, timeout=10.0)
            driver.arm_to(ARM_TUCK, tolerance=0.2, timeout=10.0)
            driver.hold_arm(ARM_TUCK.tolist(), 0.3)
            driver.drive_to(0.0)

        print(f"=== cycle {cycle}: {grasps_ok}/{grasps_total} grasps "
              f"lifted ===", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
