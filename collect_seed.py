import argparse
import socket
import struct
import time

import numpy as np

from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Walk import Walk
from bc_data import sample_own_half_target, save_dataset_atomic
from bc_sim import FallRecovery


def reset_server_state(host="127.0.0.1", trainer_port=3101):
    commands = (b"(init (trainer))", b"(playMode BeforeKickOff)", b"(playMode PlayOn)")
    try:
        with socket.create_connection((host, trainer_port), timeout=1.0) as connection:
            for index, command in enumerate(commands):
                connection.sendall(struct.pack("!I", len(command)) + command)
                if index == 0:
                    try:
                        connection.recv(1024)
                    except socket.timeout:
                        pass
                time.sleep(0.05)
        return True
    except OSError as error:
        print(f"[Trainer Warning] Could not set PlayOn mode: {error}")
        return False


def collect_seed(steps, dataset_path, seed):
    np.random.seed(seed)
    reset_server_state()
    agent = Base_Agent(
        host="localhost", agent_port=3100, monitor_port=3200, unum=1,
        robot_type=0, team_name="UnifiedCollector"
    )
    walk = Walk(agent)
    fall_recovery = FallRecovery()
    observations, oracle_actions = [], []
    target = sample_own_half_target()
    force_reset = True

    try:
        print(f"\n{'=' * 49}")
        print(f" COLLECTING DAGGER ORACLE SEED ({steps} STEPS)")
        print(f"{'=' * 49}")
        for step in range(1, steps + 1):
            recovered = fall_recovery.check_and_recover(agent)
            if recovered:
                force_reset = True
                target = sample_own_half_target()

            position = agent.world.robot.loc_head_position[:2]
            distance = np.linalg.norm(target - position)
            if distance < 0.5:
                target = sample_own_half_target()
                distance = np.linalg.norm(target - position)

            obs = walk.observe(step == 1 or force_reset, target, True, None, True, distance)
            action = walk.get_expert_action(obs)
            observations.append(np.copy(obs))
            oracle_actions.append(np.copy(action))
            walk.execute_action(action)
            force_reset = False

            command = agent.world.robot.get_command()
            agent.scom.commit_and_send(command if command else b"")
            agent.scom.receive()
            if step % 1000 == 0:
                print(
                    f"Seed Step: {step}/{steps} | Target Dist: {distance:.2f}m "
                    f"| Falls: {fall_recovery.fall_count}"
                )
    finally:
        try:
            agent.terminate()
        except Exception:
            pass

    save_dataset_atomic(
        dataset_path, np.asarray(observations, dtype=np.float32),
        np.asarray(oracle_actions, dtype=np.float32)
    )
    print(f"[Seed] Saved {len(observations)} oracle-labeled samples to '{dataset_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect the initial DAgger oracle dataset")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dataset", default="aggregated_dagger_data.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    collect_seed(args.steps, args.dataset, args.seed)
