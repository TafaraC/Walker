"""Synchronous SimSpark helpers used during BC demonstration collection."""

import numpy as np


class FallRecovery:
    """Detect falls and restore the robot to a stable pose without logging data."""

    def __init__(
        self,
        height_threshold=0.28,
        roll_threshold=60.0,
        pitch_threshold=60.0,
        confirmation_cycles=1,
        reset_position=(-3.0, 0.0),
        floating_cycles=10,
        stabilization_cycles=7,
    ):
        self.height_threshold = height_threshold
        self.roll_threshold = roll_threshold
        self.pitch_threshold = pitch_threshold
        self.confirmation_cycles = confirmation_cycles
        self.reset_position = np.asarray(reset_position, dtype=np.float32)
        self.floating_cycles = floating_cycles
        self.stabilization_cycles = stabilization_cycles
        self._low_height_cycles = 0
        self.fall_count = 0

    def check_and_recover(self, agent):
        """Return True after recovering from a confirmed fall."""
        head_height = float(agent.world.robot.loc_head_position[2])
        roll = abs(float(agent.world.robot.imu_torso_roll))
        pitch = abs(float(agent.world.robot.imu_torso_pitch))
        invalid_pose = (
            np.isfinite(head_height)
            and np.isfinite(roll)
            and np.isfinite(pitch)
            and (
                head_height < self.height_threshold
                or roll > self.roll_threshold
                or pitch > self.pitch_threshold
            )
        )
        if invalid_pose:
            self._low_height_cycles += 1
        else:
            self._low_height_cycles = 0
            return False

        if self._low_height_cycles < self.confirmation_cycles:
            return False

        self.fall_count += 1
        print(
            f"\n[Fall Recovery] Fall #{self.fall_count} detected "
            f"(head {head_height:.3f}m, roll {roll:.1f}deg, "
            f"pitch {pitch:.1f}deg). Resetting robot..."
        )
        self._recover(agent)
        self._low_height_cycles = 0
        return True

    @staticmethod
    def _sync_pose_cycle(agent):
        agent.behavior.execute("Zero_Bent_Knees")
        command = agent.world.robot.get_command()
        agent.scom.commit_and_send(command if command else b"")
        agent.scom.receive()

    def _recover(self, agent):
        robot = agent.world.robot
        x, y = self.reset_position

        # Repeated floating beams prevent the fallen body's momentum from
        # carrying into the reset while the joints approach a stable pose.
        for _ in range(self.floating_cycles):
            agent.scom.unofficial_beam((x, y, 0.50), 0)
            self._sync_pose_cycle(agent)

        agent.scom.unofficial_beam((x, y, robot.beam_height), 0)
        robot.joints_target_speed[0] = 0.01
        self._sync_pose_cycle(agent)

        for _ in range(self.stabilization_cycles):
            self._sync_pose_cycle(agent)

        agent.behavior.force_reset()
        print("[Fall Recovery] Robot reset and stabilized; resuming collection.")
