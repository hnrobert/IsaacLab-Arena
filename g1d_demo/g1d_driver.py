"""Closed-loop driver: stepping, motion primitives, navigation."""
import math

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from g1d_config import (ARM_HOME, DRIVE_SPEED, HALF_TRACK, LEFT_ARM,
                        LEFT_ARM_TUCK, RENDER_EVERY, RIGHT_ARM,
                        RIGHT_FINGERS, WHEEL_JOINTS, WHEEL_RADIUS)


class RenderStepper:
    """Physics stepping with decimated rendering (GUI speed).

    The warehouse kit scene is render-heavy: per-step RTX rendering
    runs the GUI at ~0.15x real time (a 25 s drive takes 3 wall-
    clock minutes). Physics keeps its 60 Hz; only every `every`-th
    step renders (~fps visual). Headless never renders.
    """

    def __init__(self, headless: bool, every: int = RENDER_EVERY):
        self.headless = headless
        self.every = max(1, every)
        self._count = 0
        self.steps = 0  # total physics steps (sim-time clock)

    def step(self, sim) -> None:
        self.steps += 1
        if self.headless:
            sim.step(render=False)
            return
        self._count += 1
        if self._count >= self.every:
            self._count = 0
            sim.step(render=True)
        else:
            sim.step(render=False)


def n(t):
    """Device buffer (torch OR warp array) -> numpy."""
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return t.numpy()  # warp array (Fabric/GPU pipeline)

def yaw_of(quat) -> float:
    """Yaw (rad) from a single xyzw quaternion.

    NOTE: the cross term is w*z (not x*w!) - the wrong pairing reads a
    constant ~0 for every pure-yaw rotation and silently kills ALL
    heading feedback.
    """
    x, y, z, w = (
        quat.tolist() if hasattr(quat, "tolist") else list(quat)
    )
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class G1DDriver:
    """Closed-loop driver over the articulation tensor API."""

    HAND_LINKS = ("right_hand_thumb_2_link",
                  "right_hand_index_1_link",
                  "right_hand_middle_1_link")

    def __init__(self, robot: Articulation, sim: SimulationContext,
                 totes=None, step=None):
        self.robot = robot
        self.sim = sim
        # render-decimated stepping (or a plain step when omitted)
        self._step = (lambda: step(self.sim)) if step is not None \
            else self.sim.step
        self.device = robot.device
        names = self.robot.joint_names
        self.idx = {nm: i for i, nm in enumerate(names)}
        self.arm_ids = torch.tensor(
            [self.idx[nm] for nm in RIGHT_ARM], device=robot.device)
        self.left_arm_ids = torch.tensor(
            [self.idx[nm] for nm in LEFT_ARM], device=robot.device)
        self.finger_ids = torch.tensor(
            [self.idx[nm] for nm in RIGHT_FINGERS], device=robot.device)
        self.left_wheel = self.idx[WHEEL_JOINTS[0]]
        self.right_wheel = self.idx[WHEEL_JOINTS[1]]
        self.n_joints = len(names)
        # rigid-object data buffers ONLY refresh on update(); skipping
        # it made every tote print a stale snapshot (phantom "empty")
        self.totes = list(totes) if totes else []
        self._last_right_pose = ARM_HOME.tolist()

    # ---- outputs ----

    def _zeros(self) -> torch.Tensor:
        return torch.zeros((1, self.n_joints), device=self.device)

    def _baseline(self) -> torch.Tensor:
        """Position-command baseline: LEFT arm always chest-folded."""
        cmd = self._zeros()
        cmd[0, self.left_arm_ids] = LEFT_ARM_TUCK.to(self.device)
        return cmd

    def arm(self, pose) -> None:
        cmd = self._baseline()
        cmd[0, self.arm_ids] = torch.as_tensor(
            pose, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(cmd)
        self._last_right_pose = list(pose)

    def fingers(self, pose) -> None:
        cmd = self._baseline()
        cmd[0, self.finger_ids] = pose.to(self.device)
        cmd[0, self.arm_ids] = torch.as_tensor(
            self._last_right_pose, dtype=torch.float32,
            device=self.device)
        self.robot.set_joint_position_target(cmd)

    def wheels(self, left: float, right: float) -> None:
        """+v = +x forward; LEFT command negated (mirrored URDF axes)."""
        cmd = self._zeros()
        cmd[0, self.left_wheel] = -left
        cmd[0, self.right_wheel] = right
        self.robot.set_joint_velocity_target(cmd)

    # ---- state ----

    def pose(self):
        pos = n(self.robot.data.root_pos_w)[0]
        yaw = yaw_of(n(self.robot.data.root_quat_w)[0])
        return float(pos[0]), float(pos[1]), yaw

    def finger_max(self) -> float:
        pos = n(self.robot.data.joint_pos)[0]
        return max(
            abs(pos[self.idx[nm]].item()) for nm in RIGHT_FINGERS
        )

    def hand_center(self) -> np.ndarray:
        """Grasp center: thumb tip vs index/middle pair midpoint."""
        pos = n(self.robot.data.body_pos_w)[0]
        names = self.robot.data.body_names
        thumb = pos[names.index(self.HAND_LINKS[0])]
        pair = (pos[names.index(self.HAND_LINKS[1])]
                + pos[names.index(self.HAND_LINKS[2])]) / 2.0
        return (thumb + pair) / 2.0

    # ---- primitives ----

    def _refresh(self) -> None:
        self.robot.update(self.sim.get_physics_dt())
        for t in self.totes:
            t.update(self.sim.get_physics_dt())

    def run_seconds(self, seconds: float, every_step=None) -> None:
        steps = int(seconds / self.sim.get_physics_dt())
        for _ in range(steps):
            if every_step is not None:
                every_step()
            self.robot.write_data_to_sim()
            self._step()
            self._refresh()

    def hold_arm(self, pose, seconds: float,
                 fingers: list = None) -> None:
        """Hold one right-arm pose (left arm stays chest-folded).

        fingers=None commands the jaw OPEN: during descents the
        fingers MUST be actively driven open - freezing them at
        measured positions lets contact half-close the jaw mid-
        descent and the arm plows the tote 30 cm off its slot
        (d13/d14). Pass an explicit list for other targets.
        """
        cmd = self._baseline()
        cmd[0, self.arm_ids] = torch.as_tensor(
            pose, dtype=torch.float32, device=self.device)
        if fingers is not None:
            cmd[0, self.finger_ids] = torch.as_tensor(
                fingers, dtype=torch.float32, device=self.device)
        self._last_right_pose = list(pose)
        self.run_seconds(
            seconds,
            every_step=lambda: self.robot.set_joint_position_target(cmd),
        )

    def move_arm_smooth(self, pose, speed=0.9, timeout=10.0):
        """Continuously-interpolated arm move: a velocity-limited
        linear ramp re-targeted EVERY physics step (60 Hz).

        The old step-and-hold ramp (0.05-0.12 rad jumps every
        0.06-0.2 s) looked stuttery and its discrete JUMPS were what
        plowed totes on contact - per-step increments of speed*dt
        (~0.008 rad) stay gentle even at higher rates, so the
        descent can be faster AND smoother.
        """
        pose = list(pose)
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        while waited < deadline:
            cur = [n(self.robot.data.joint_pos)[0][
                self.idx[nm]].item() for nm in RIGHT_ARM]
            deltas = [t - c for t, c in zip(pose, cur)]
            if max(abs(d) for d in deltas) < 0.03:
                self.hold_arm(pose, 0.1)
                return True
            cap = speed * dt
            nxt = [c + max(-cap, min(cap, d))
                   for c, d in zip(cur, deltas)]
            self.hold_arm(nxt, dt)
            waited += 1
        return False

    def move_arm_slow(self, pose, step=0.05, hold=0.15,
                      timeout=10.0):
        """Ramped arm move: 0.05 rad steps from the CURRENT joints.

        Step commands with the 800-stiffness drive snap the arm in
        ~0.2 s (too fast to look at and too fast for contact), so
        every grasp-phase transition goes through this.
        """
        pose = list(pose)
        deadline = timeout / self.sim.get_physics_dt()
        waited = 0
        while waited < deadline:
            cur = [n(self.robot.data.joint_pos)[0][
                self.idx[nm]].item() for nm in RIGHT_ARM]
            deltas = [t - c for t, c in zip(pose, cur)]
            if max(abs(d) for d in deltas) < 0.03:
                self.hold_arm(pose, 0.1)
                return True
            nxt = [c + max(-step, min(step, d))
                   for c, d in zip(cur, deltas)]
            self.hold_arm(nxt, hold)
            waited += int(hold / self.sim.get_physics_dt())
        return False

    def arm_to(self, pose, tolerance=0.3, timeout=12.0) -> bool:
        deadline = timeout / self.sim.get_physics_dt()
        waited = 0
        while waited < deadline:
            self.arm(pose)
            self.robot.write_data_to_sim()
            self._step()
            self._refresh()
            waited += 1
            current = n(self.robot.data.joint_pos)[0]
            err = max(
                abs(current[self.idx[nm]].item() - t)
                for nm, t in zip(RIGHT_ARM, list(pose))
            )
            if err < tolerance:
                return True
        return False

    def fingers_to(self, pose, tolerance=0.4, timeout=8.0) -> bool:
        deadline = timeout / self.sim.get_physics_dt()
        waited = 0
        while waited < deadline:
            self.fingers(pose)
            self.robot.write_data_to_sim()
            self._step()
            self._refresh()
            waited += 1
            current = n(self.robot.data.joint_pos)[0]
            plist = (pose.detach().cpu().tolist()
                     if torch.is_tensor(pose) else list(pose))
            err = max(
                abs(current[self.idx[nm]].item() - t)
                for nm, t in zip(RIGHT_FINGERS, plist)
            )
            if err < tolerance:
                return True
        return False

    def holding(self) -> bool:
        return 0.30 < self.finger_max() < 0.78

    # ---- base motion ----

    def shunt_lateral(self, line_y: float = 0.0,
                      timeout: float = 15.0, tol: float = 0.08) -> bool:
        """Translate sideways onto the y=line_y rail (endgame align)."""
        _, y, _ = self.pose()
        dy = y - line_y
        if abs(dy) < tol:
            return True
        target_yaw = -0.6 if dy > 0 else 0.6
        if not self.spin_to(target_yaw, timeout=10.0):
            return False
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        while waited < deadline:
            _, y, _ = self.pose()
            if abs(y - line_y) < 0.04:
                break
            self.wheels(0.12, 0.12)
            self.robot.write_data_to_sim()
            self._step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return self.spin_to(0.0, timeout=10.0)

    def drive_to(
        self, target_x: float, target_y: float = 0.0,
        timeout: float = 60.0, tol: float = 0.06,
        speed: float = DRIVE_SPEED, spin_first: bool = True,
    ) -> bool:
        """Cruise along the aisle toward (target_x, target_y), 2D-steered.

        spin_first=False skips the in-place align spin - rotating with
        a loaded arm strips the grip, so carrying drives never spin.
        """
        if spin_first and not self.spin_to(0.0, timeout=20.0):
            return False
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        shunts = 0
        ref_x, ref_t = None, 0
        while waited < deadline:
            x, y, yaw = self.pose()
            dx = target_x - x
            dy_line = y - target_y
            if ref_x is None:
                ref_x = x
            # stall detector: grinding against an edge (seen: nose
            # jammed on the station wall at dx 0.077 forever) -> free
            # the line with a shunt, or accept a loose arrival.
            # Compare over an 8 s WINDOW - at crawl speed the base
            # moves ~1 mm/step, so per-step diffs always look stuck.
            if (waited - ref_t) >= int(4.0 / dt):
                if abs(x - ref_x) < 0.01:
                    if shunts < 1:
                        shunts += 1
                        print(f"drive stalled x={x:.3f} y={y:+.3f} "
                              f"-> shunt", flush=True)
                        self.shunt_lateral(target_y)
                    else:
                        self.wheels(0.0, 0.0)
                        print(f"arrived x={x:.3f} y={y:.3f} "
                              f"(stalled-loose) "
                              f"(target {target_x:.3f})", flush=True)
                        return True
                    ref_x, ref_t = x, waited
                else:
                    ref_x, ref_t = x, waited
            if abs(dx) < tol and abs(yaw) < 0.25:
                if abs(dy_line) >= 0.08:
                    if shunts < 2:
                        shunts += 1
                        self.shunt_lateral(target_y)
                        continue
                    self.wheels(0.0, 0.0)
                    print(f"arrived x={x:.3f} y={y:.3f} (loose) "
                          f"(target {target_x:.3f})", flush=True)
                    return True
                self.wheels(0.0, 0.0)
                print(f"arrived x={x:.3f} y={y:.3f} "
                      f"(target {target_x:.3f})", flush=True)
                return True
            v = max(-speed, min(speed, 0.7 * dx))
            if abs(v) < 0.06:
                v = 0.06 if dx > 0 else -0.06
            # calm PURE-PURSUIT line tracker: reversing legs drift
            # sideways (casters). Lookahead shrinks with the
            # remaining distance so lateral offsets actually close
            # (a fixed 2.5 m lookahead on a 1.8 m leg leaves ~8 cm)
            desired = max(-0.35, min(
                0.35, math.atan2(-dy_line, max(0.35, abs(dx)))))
            if dx < 0:
                desired = -desired
            err_yaw = (desired - yaw + 3.14159265) % (2 * 3.14159265) \
                - 3.14159265
            turn = max(-0.35, min(0.35, 1.2 * err_yaw))
            self.wheels(
                (v - turn * HALF_TRACK) / WHEEL_RADIUS,
                (v + turn * HALF_TRACK) / WHEEL_RADIUS,
            )
            if waited % int(2.0 / dt) == 0:
                print(f"drive: x={x:.3f} y={y:+.3f} dx={dx:+.3f} "
                      f"yaw={yaw:+.3f}", flush=True)
            self.robot.write_data_to_sim()
            self._step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return False

    def back_straight(self, dist: float, speed: float = 0.05,
                      arm_hold=None, timeout: float = 25.0) -> float:
        """Straight-line reverse with a gentle yaw lock.

        Equal wheel velocities alone drift LATERALLY ~0.11 m over a
        0.35 m reverse (casters), and any strong steering correction
        swings the arm tip sideways on a ~0.5 m lever and strips the
        pinch. The tiny clamped yaw lock keeps the heading (and the
        arm) pointed without jerking the load.
        Optionally holds a full joint command (arm + fingers) every
        step. Returns the distance actually reversed.
        """
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        x0, _, yaw0 = self.pose()
        while waited < deadline:
            if (x0 - self.pose()[0]) >= dist:
                break
            _, _, yaw = self.pose()
            err = ((yaw - yaw0 + 3.14159265) % (2 * 3.14159265)
                   - 3.14159265)
            turn = max(-0.08, min(0.08, 0.8 * err))
            v = -speed
            self.wheels(
                (v - turn * HALF_TRACK) / WHEEL_RADIUS,
                (v + turn * HALF_TRACK) / WHEEL_RADIUS,
            )
            if arm_hold is not None:
                self.robot.set_joint_position_target(arm_hold)
            self.robot.write_data_to_sim()
            self._step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return x0 - self.pose()[0]

    def spin_to(self, target_yaw: float, timeout: float = 20.0) -> bool:
        dt = self.sim.get_physics_dt()
        deadline = timeout / dt
        waited = 0
        while waited < deadline:
            _, _, yaw = self.pose()
            err = (yaw - target_yaw + 3.14159265) % (2 * 3.14159265) \
                - 3.14159265
            if abs(err) < 0.15:
                self.wheels(0.0, 0.0)
                return True
            rate = max(0.6, min(2.0, 1.5 * abs(err)))
            # +w in wheels() yaws the base DOWN; err>0 means yaw must
            # decrease toward the target (probe-verified sign)
            w = -rate if err > 0 else rate
            self.wheels(
                -w * HALF_TRACK / WHEEL_RADIUS,
                w * HALF_TRACK / WHEEL_RADIUS,
            )
            self.robot.write_data_to_sim()
            self._step()
            self._refresh()
            waited += 1
        self.wheels(0.0, 0.0)
        return False

def reset_tote(tote, pos) -> None:
    """Teleport a tote back to its slot (instant in PhysX)."""
    try:
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        tote.write_root_pose_to_sim(
            torch.tensor([[pos[0], pos[1], pos[2],
                           1.0, 0.0, 0.0, 0.0]],
                         dtype=torch.float32, device=dev)
        )
    except Exception as exc:  # noqa: BLE001
        print(f"tote reset failed: {exc}", flush=True)
