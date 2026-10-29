import argparse
from pathlib import Path

import numpy as np
import torch

from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Walk import Walk
from bc_data import load_dataset, sample_own_half_target, save_json_atomic
from bc_sim import FallRecovery
from collect_seed import reset_server_state
from gail_models import GAILActorCritic
from train import BCPolicy


DT = 0.02


def load_policy(kind, model_path, obs_dim, action_dim):
    if kind in ("bc", "dagger"):
        model = BCPolicy(obs_dim, action_dim)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    elif kind == "gail":
        model = GAILActorCritic(obs_dim, action_dim)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        return None
    model.eval()
    return model


def select_action(kind, model, obs, oracle_action, rng, random_low, random_high):
    if kind == "oracle":
        return oracle_action
    if kind == "random":
        return rng.uniform(random_low, random_high).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(obs).unsqueeze(0)
        if kind == "gail":
            return model.actor_mean(tensor).squeeze(0).numpy().astype(np.float32)
        return model(tensor).squeeze(0).numpy().astype(np.float32)


def evaluate(kind, label, model_path, expert_data, steps, seed, output_dir,
             environment_id="nominal", field_slip=0.1):
    expert_obs, expert_actions = load_dataset(expert_data)
    obs_dim, action_dim = expert_obs.shape[1], expert_actions.shape[1]
    model = load_policy(kind, model_path, obs_dim, action_dim)
    random_low = expert_actions.min(axis=0)
    random_high = expert_actions.max(axis=0)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    reset_server_state()
    agent = Base_Agent(
        host="localhost", agent_port=3100, monitor_port=3200, unum=1,
        robot_type=0, team_name="UnifiedEvaluator"
    )
    walk = Walk(agent)
    recovery = FallRecovery()
    target = sample_own_half_target()
    force_reset = True
    targets_reached = 0
    episode_id = 0

    trace = {key: [] for key in (
        "obs", "actions", "oracle_actions", "joint_positions_deg",
        "positions", "targets", "target_progress", "planar_speed",
        "forward_velocity_mps", "target_direction_error", "torso_roll_deg", "torso_pitch_deg",
        "head_height_m", "episode_id",
    )}

    try:
        print(f"[Evaluation] {label} | seed={seed} | active steps={steps}")
        for step in range(steps):
            recovered = recovery.check_and_recover(agent)
            if recovered:
                episode_id += 1
                force_reset = True
                target = sample_own_half_target()

            robot = agent.world.robot
            position_before = np.asarray(robot.loc_head_position[:2], dtype=np.float32).copy()
            torso_angle_before = np.deg2rad(float(robot.imu_torso_orientation))
            distance_before = float(np.linalg.norm(target - position_before))
            if distance_before < 0.5:
                targets_reached += 1
                target = sample_own_half_target()
                distance_before = float(np.linalg.norm(target - position_before))

            obs = walk.observe(step == 0 or force_reset, target, True, None, True, distance_before)
            oracle_action = walk.get_expert_action(obs)
            action = select_action(
                kind, model, obs, oracle_action, rng, random_low, random_high
            )
            walk.execute_action(action)
            force_reset = False

            trace["obs"].append(np.copy(obs))
            trace["actions"].append(np.copy(action))
            trace["oracle_actions"].append(np.copy(oracle_action))
            trace["joint_positions_deg"].append(np.copy(robot.joints_position[:22]))
            trace["positions"].append(position_before)
            trace["targets"].append(np.copy(target))
            trace["torso_roll_deg"].append(float(robot.imu_torso_roll))
            trace["torso_pitch_deg"].append(float(robot.imu_torso_pitch))
            trace["head_height_m"].append(float(robot.loc_head_position[2]))
            trace["episode_id"].append(episode_id)

            command = robot.get_command()
            agent.scom.commit_and_send(command if command else b"")
            agent.scom.receive()

            position_after = np.asarray(robot.loc_head_position[:2], dtype=np.float32).copy()
            distance_after = float(np.linalg.norm(target - position_after))
            displacement = position_after - position_before
            speed = float(np.linalg.norm(displacement) / DT)
            forward_direction = np.array([np.cos(torso_angle_before), np.sin(torso_angle_before)])
            forward_velocity = float(np.dot(displacement / DT, forward_direction))
            target_vector = target - position_before
            target_norm = float(np.linalg.norm(target_vector))
            displacement_norm = float(np.linalg.norm(displacement))
            if displacement_norm > 1e-8 and target_norm > 1e-8:
                direction_error = float(np.linalg.norm(
                    displacement / displacement_norm - target_vector / target_norm
                ))
            else:
                direction_error = 1.0
            trace["target_progress"].append(distance_before - distance_after)
            trace["planar_speed"].append(speed)
            trace["forward_velocity_mps"].append(forward_velocity)
            trace["target_direction_error"].append(direction_error)

            if (step + 1) % 1000 == 0:
                print(
                    f"[Evaluation] {step + 1}/{steps} | Falls: {recovery.fall_count} "
                    f"| Targets: {targets_reached}"
                )
    finally:
        try:
            agent.terminate()
        except Exception:
            pass

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / f"{label}_seed_{seed}.npz"
    np.savez_compressed(trace_path, **{
        key: np.asarray(values, dtype=np.int64 if key == "episode_id" else np.float32)
        for key, values in trace.items()
    })
    progress = np.asarray(trace["target_progress"], dtype=np.float32)
    action_error = np.asarray(trace["actions"]) - np.asarray(trace["oracle_actions"])
    summary = {
        "policy_kind": kind,
        "policy_label": label,
        "model": model_path,
        "environment_id": environment_id,
        "field_contact_slip": field_slip,
        "seed": seed,
        "active_steps": steps,
        "active_seconds": steps * DT,
        "return_target_progress_m": float(progress.sum()),
        "mean_target_directed_velocity_mps": float(progress.mean() / DT),
        "mean_planar_speed_mps": float(np.mean(trace["planar_speed"])),
        "mean_forward_velocity_mps": float(np.mean(trace["forward_velocity_mps"])),
        "mean_target_direction_error": float(np.mean(trace["target_direction_error"])),
        "targets_reached": targets_reached,
        "falls": recovery.fall_count,
        "falls_per_active_minute": recovery.fall_count / (steps * DT / 60.0),
        "on_policy_oracle_action_mse": float(np.mean(np.square(action_error))),
        "trace": str(trace_path),
    }
    save_json_atomic(output / f"{label}_seed_{seed}.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate one walking policy")
    parser.add_argument("--policy", choices=("random", "oracle", "bc", "dagger", "gail"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--expert-data", default="expert_train_data.npz")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="evaluation/raw")
    parser.add_argument("--environment-id", default="nominal")
    parser.add_argument("--field-slip", type=float, default=0.1)
    args = parser.parse_args()
    if args.policy not in ("random", "oracle") and not args.model:
        parser.error("--model is required for learned policies")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    evaluate(
        args.policy, args.label, args.model, args.expert_data,
        args.steps, args.seed, args.output_dir,
        args.environment_id, args.field_slip,
    )
