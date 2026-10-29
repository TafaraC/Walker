import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Walk import Walk
from bc_data import (
    append_dataset, load_dataset, sample_own_half_target,
    save_dataset_atomic, save_json_atomic,
)
from bc_sim import FallRecovery
from collect_seed import reset_server_state
from train import BCPolicy


def default_beta(iteration):
    """Linearly remove oracle control over the first five DAgger iterations."""
    return max(0.0, 1.0 - 0.2 * iteration)


def run_rollout(
    iteration,
    steps=10000,
    beta=None,
    seed=42,
    model_path="dagger_policy.pth",
    dataset_path="aggregated_dagger_data.npz",
    metrics_path=None,
    label="dagger",
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DAgger policy not found: {model_path}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    beta = default_beta(iteration) if beta is None else beta
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")

    seed_obs, seed_actions = load_dataset(dataset_path)
    device = torch.device("cpu")
    policy = BCPolicy(seed_obs.shape[1], seed_actions.shape[1]).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    reset_server_state()
    agent = Base_Agent(
        host="localhost", agent_port=3100, monitor_port=3200, unum=1,
        robot_type=0, team_name="UnifiedCollector"
    )
    walk = Walk(agent)
    fall_recovery = FallRecovery()

    observations, oracle_actions = [], []
    squared_errors, absolute_errors = [], []
    roll_values, pitch_values, head_heights, action_changes = [], [], [], []
    previous_student_action = None
    expert_control_steps = 0
    targets_reached = 0
    target = sample_own_half_target()
    force_reset = True

    try:
        print(f"\n{'=' * 57}")
        print(f" DAGGER ROLLOUT {iteration}: {steps} STEPS | ORACLE BETA {beta:.2f}")
        print(" Student visits states; FC Portugal labels every state")
        print(f"{'=' * 57}")

        for step in range(1, steps + 1):
            recovered = fall_recovery.check_and_recover(agent)
            if recovered:
                force_reset = True
                previous_student_action = None
                target = sample_own_half_target()

            robot = agent.world.robot
            position = robot.loc_head_position[:2]
            distance = np.linalg.norm(target - position)
            if distance < 0.5:
                targets_reached += 1
                target = sample_own_half_target()
                distance = np.linalg.norm(target - position)

            obs = walk.observe(step == 1 or force_reset, target, True, None, True, distance)
            oracle_action = walk.get_expert_action(obs)
            with torch.no_grad():
                student_action = policy(
                    torch.from_numpy(obs).unsqueeze(0).to(device)
                ).squeeze(0).cpu().numpy().astype(np.float32)

            error = student_action - oracle_action
            observations.append(np.copy(obs))
            oracle_actions.append(np.copy(oracle_action))
            squared_errors.append(np.square(error))
            absolute_errors.append(np.abs(error))
            roll_values.append(float(robot.imu_torso_roll))
            pitch_values.append(float(robot.imu_torso_pitch))
            head_heights.append(float(robot.loc_head_position[2]))
            if previous_student_action is not None:
                action_changes.append(np.square(student_action - previous_student_action))
            previous_student_action = np.copy(student_action)

            use_oracle = rng.random() < beta
            executed_action = oracle_action if use_oracle else student_action
            expert_control_steps += int(use_oracle)
            walk.execute_action(executed_action)
            force_reset = False

            command = robot.get_command()
            agent.scom.commit_and_send(command if command else b"")
            agent.scom.receive()

            if step % 1000 == 0:
                print(
                    f"Rollout: {step}/{steps} | Dist: {distance:.2f}m "
                    f"| Falls: {fall_recovery.fall_count} | Goals: {targets_reached}"
                )
    finally:
        try:
            agent.terminate()
        except Exception:
            pass

    observations = np.asarray(observations, dtype=np.float32)
    oracle_actions = np.asarray(oracle_actions, dtype=np.float32)
    squared_errors = np.asarray(squared_errors, dtype=np.float32)
    absolute_errors = np.asarray(absolute_errors, dtype=np.float32)
    chunk_path = Path(dataset_path).with_name(
        f".{Path(dataset_path).stem}_iteration_{iteration}_chunk.npz"
    )
    save_dataset_atomic(chunk_path, observations, oracle_actions)
    aggregated_count = append_dataset(dataset_path, chunk_path)

    roll_values = np.asarray(roll_values, dtype=np.float32)
    pitch_values = np.asarray(pitch_values, dtype=np.float32)
    head_heights = np.asarray(head_heights, dtype=np.float32)
    action_changes = np.asarray(action_changes, dtype=np.float32)
    metrics = {
        "algorithm": "dagger",
        "policy_label": label,
        "iteration": iteration,
        "seed": seed,
        "requested_beta": beta,
        "realized_oracle_control_fraction": expert_control_steps / len(observations),
        "rollout_samples": len(observations),
        "aggregated_samples": aggregated_count,
        "on_policy_oracle_mse": float(squared_errors.mean()),
        "on_policy_oracle_rmse": float(np.sqrt(squared_errors.mean())),
        "on_policy_oracle_mae": float(absolute_errors.mean()),
        "per_action_rmse": np.sqrt(squared_errors.mean(axis=0)).tolist(),
        "per_action_mae": absolute_errors.mean(axis=0).tolist(),
        "falls": fall_recovery.fall_count,
        "falls_per_1000_steps": fall_recovery.fall_count * 1000.0 / len(observations),
        "targets_reached": targets_reached,
        "targets_per_1000_steps": targets_reached * 1000.0 / len(observations),
        "torso_roll_rms_deg": float(np.sqrt(np.mean(np.square(roll_values)))),
        "torso_roll_peak_deg": float(np.max(np.abs(roll_values))),
        "torso_pitch_rms_deg": float(np.sqrt(np.mean(np.square(pitch_values)))),
        "torso_pitch_peak_deg": float(np.max(np.abs(pitch_values))),
        "mean_head_height_m": float(np.mean(head_heights)),
        "minimum_head_height_m": float(np.min(head_heights)),
        "student_action_delta_rms": (
            float(np.sqrt(action_changes.mean())) if len(action_changes) else 0.0
        ),
    }
    metrics_path = metrics_path or f"dagger_rollout_metrics_iter_{iteration}.json"
    save_json_atomic(metrics_path, metrics)
    print(f"[DAgger] Aggregated dataset now contains {aggregated_count} samples")
    print(f"[DAgger] Rollout metrics saved to '{metrics_path}'")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a correctly labeled DAgger rollout")
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="dagger_policy.pth")
    parser.add_argument("--dataset", default="aggregated_dagger_data.npz")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--label", default="dagger")
    args = parser.parse_args()
    try:
        run_rollout(
            args.iter, args.steps, args.beta, args.seed, args.model,
            args.dataset, args.metrics_output, args.label
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"[DAgger Error] {error}", file=sys.stderr)
        raise SystemExit(1)
