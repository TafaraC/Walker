import argparse
import os

import numpy as np
import torch

from agent.Base_Agent import Base_Agent
from behaviors.custom.Walk.Walk import Walk
from bc_data import load_dataset, sample_own_half_target, save_json_atomic
from bc_sim import FallRecovery
from collect_seed import reset_server_state
from gail_adapter import GAILWalkAdapter
from gail_models import GAILActorCritic, initialize_actor_from_imitation


def run_gail_rollout(
    iteration,
    steps=4000,
    seed=42,
    model_path="gail_policy.pth",
    init_model_path="dagger_policy.pth",
    expert_file="expert_train_data.npz",
    rollout_path="gail_rollout_batch.npz",
    metrics_path=None,
    label="gail_dagger",
):
    expert_obs, expert_actions = load_dataset(expert_file)
    device = torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = GAILActorCritic(expert_obs.shape[1], expert_actions.shape[1]).to(device)
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location=device))
        initialization = model_path
    else:
        if init_model_path == "random":
            initialization = "random actor initialization"
        elif not os.path.exists(init_model_path):
            raise FileNotFoundError(
                f"Neither GAIL policy '{model_path}' nor initialization policy "
                f"'{init_model_path}' exists"
            )
        else:
            initialize_actor_from_imitation(policy, init_model_path, device)
            initialization = init_model_path
        # Persist the exact initialized critic and exploration parameters so
        # training evaluates the same policy that generated this rollout.
        torch.save(policy.state_dict(), model_path)
    policy.eval()

    reset_server_state()
    agent = Base_Agent(
        host="localhost", agent_port=3100, monitor_port=3200, unum=1,
        robot_type=0, team_name="UnifiedCollector"
    )
    walk = Walk(agent)
    adapter = GAILWalkAdapter(agent, walk)
    recovery = FallRecovery()

    observations, actions, logprobs, values, dones = [], [], [], [], []
    roll_values, pitch_values, head_heights, action_changes = [], [], [], []
    previous_action = None
    targets_reached = 0
    target = sample_own_half_target()
    force_reset = True
    bootstrap_value = 0.0

    try:
        print(f"\n{'=' * 53}")
        print(f" GAIL ROLLOUT {iteration}: {steps} STEPS")
        print(f" Actor initialization/current checkpoint: {initialization}")
        print(f"{'=' * 53}")

        for step in range(1, steps + 1):
            recovered = recovery.check_and_recover(agent)
            if recovered:
                # The preceding action produced the terminal fallen state.
                if dones:
                    dones[-1] = True
                force_reset = True
                previous_action = None
                target = sample_own_half_target()

            robot = agent.world.robot
            position = robot.loc_head_position[:2]
            distance = np.linalg.norm(target - position)
            if distance < 0.5:
                targets_reached += 1
                target = sample_own_half_target()
                distance = np.linalg.norm(target - position)

            obs = adapter.observe(step == 1 or force_reset, target, distance)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                distribution, value = policy(obs_tensor)
                action_tensor = distribution.sample()
                logprob = distribution.log_prob(action_tensor).sum(dim=-1)
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            adapter.step_action(action)
            force_reset = False

            observations.append(np.copy(obs))
            actions.append(np.copy(action))
            logprobs.append(float(logprob.item()))
            values.append(float(value.item()))
            dones.append(False)
            roll_values.append(float(robot.imu_torso_roll))
            pitch_values.append(float(robot.imu_torso_pitch))
            head_heights.append(float(robot.loc_head_position[2]))
            if previous_action is not None:
                action_changes.append(np.square(action - previous_action))
            previous_action = np.copy(action)

            command = robot.get_command()
            agent.scom.commit_and_send(command if command else b"")
            agent.scom.receive()

            if step % 1000 == 0:
                print(
                    f"Rollout: {step}/{steps} | Dist: {distance:.2f}m "
                    f"| Falls: {recovery.fall_count} | Goals: {targets_reached}"
                )

        if not dones[-1]:
            robot = agent.world.robot
            distance = np.linalg.norm(target - robot.loc_head_position[:2])
            next_obs = adapter.observe(False, target, distance)
            with torch.no_grad():
                _, next_value = policy(torch.from_numpy(next_obs).unsqueeze(0).to(device))
            bootstrap_value = float(next_value.item())
    finally:
        try:
            agent.terminate()
        except Exception:
            pass

    np.savez_compressed(
        rollout_path,
        obs=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        logprobs=np.asarray(logprobs, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        bootstrap_value=np.asarray(bootstrap_value, dtype=np.float32),
        iteration=np.asarray(iteration, dtype=np.int64),
    )

    roll_values = np.asarray(roll_values, dtype=np.float32)
    pitch_values = np.asarray(pitch_values, dtype=np.float32)
    head_heights = np.asarray(head_heights, dtype=np.float32)
    action_changes = np.asarray(action_changes, dtype=np.float32)
    metrics = {
        "algorithm": "gail",
        "policy_label": label,
        "actor_initialization": init_model_path,
        "iteration": iteration,
        "seed": seed,
        "rollout_samples": len(observations),
        "falls": recovery.fall_count,
        "falls_per_1000_steps": recovery.fall_count * 1000.0 / len(observations),
        "targets_reached": targets_reached,
        "targets_per_1000_steps": targets_reached * 1000.0 / len(observations),
        "torso_roll_rms_deg": float(np.sqrt(np.mean(np.square(roll_values)))),
        "torso_roll_peak_deg": float(np.max(np.abs(roll_values))),
        "torso_pitch_rms_deg": float(np.sqrt(np.mean(np.square(pitch_values)))),
        "torso_pitch_peak_deg": float(np.max(np.abs(pitch_values))),
        "mean_head_height_m": float(np.mean(head_heights)),
        "minimum_head_height_m": float(np.min(head_heights)),
        "action_delta_rms": (
            float(np.sqrt(action_changes.mean())) if len(action_changes) else 0.0
        ),
        "bootstrap_value": bootstrap_value,
    }
    metrics_path = metrics_path or f"gail_rollout_metrics_iter_{iteration}.json"
    save_json_atomic(metrics_path, metrics)
    print(f"[GAIL] Rollout saved to '{rollout_path}'")
    print(f"[GAIL] Mechanical metrics saved to '{metrics_path}'")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect a synchronous GAIL rollout")
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gail_policy.pth")
    parser.add_argument("--init-model", default="dagger_policy.pth")
    parser.add_argument("--expert-data", default="expert_train_data.npz")
    parser.add_argument("--rollout-output", default="gail_rollout_batch.npz")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--label", default="gail_dagger")
    args = parser.parse_args()
    run_gail_rollout(
        args.iter, args.steps, args.seed, args.model, args.init_model,
        args.expert_data, args.rollout_output, args.metrics_output, args.label
    )
