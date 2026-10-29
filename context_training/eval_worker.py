"""One context BC/oracle evaluation client; server is owned by the parent runner."""
import argparse
from pathlib import Path

import numpy as np
import torch

from variable_training.common import read, save
from .models import load_checkpoint, predict

DT = .02


def evaluate(job, directory):
    # Native modules are only needed in an explicitly launched live worker.
    from agent.Base_Agent import Base_Agent
    from behaviors.custom.Walk.Walk import Walk
    from bc_data import sample_own_half_target
    from bc_sim import FallRecovery
    from collect_seed import reset_server_state

    row, controller = job['row'], job['controller']
    model = None
    if controller['architecture'] != 'oracle':
        model, architecture = load_checkpoint(controller['model'])
        if architecture != controller['architecture']:
            raise ValueError('Wrong checkpoint architecture')
    np.random.seed(row['seed'])
    torch.manual_seed(row['seed'])
    reset_server_state()
    agent = Base_Agent(host='localhost', agent_port=3100, monitor_port=3200,
                       unum=1, robot_type=0, team_name='ContextEvaluator')
    walk, recovery = Walk(agent), FallRecovery()
    target, reset, episode, goals = sample_own_half_target(), True, 0, 0
    trace = {k: [] for k in ('obs', 'actions', 'oracle_actions', 'joint_positions_deg',
        'positions', 'targets', 'target_progress', 'planar_speed', 'forward_velocity_mps',
        'target_direction_error', 'torso_roll_deg', 'torso_pitch_deg', 'head_height_m', 'episode_id')}
    try:
        for step in range(job['steps']):
            if recovery.check_and_recover(agent):
                episode += 1
                target, reset = sample_own_half_target(), True
            robot = agent.world.robot
            before = np.asarray(robot.loc_head_position[:2], dtype=np.float32).copy()
            heading = np.deg2rad(float(robot.imu_torso_orientation))
            distance = float(np.linalg.norm(target-before))
            if distance < .5:
                goals += 1
                target = sample_own_half_target()
                distance = float(np.linalg.norm(target-before))
            obs = walk.observe(reset, target, True, None, True, distance)
            oracle = walk.get_expert_action(obs)
            action = oracle if model is None else predict(
                model, architecture, obs,
                row['slip'] if architecture=='hidden' else job['supplied_slip'])
            if np.asarray(action).shape != (16,) or not np.isfinite(action).all():
                raise ValueError('Invalid policy action')
            walk.execute_action(action)
            reset = False
            values = dict(obs=obs, actions=action, oracle_actions=oracle,
                joint_positions_deg=robot.joints_position[:22], positions=before, targets=target,
                torso_roll_deg=robot.imu_torso_roll, torso_pitch_deg=robot.imu_torso_pitch,
                head_height_m=robot.loc_head_position[2], episode_id=episode)
            for key, value in values.items():
                trace[key].append(np.copy(value))
            agent.scom.commit_and_send(robot.get_command() or b'')
            agent.scom.receive()
            after = np.asarray(robot.loc_head_position[:2], dtype=np.float32).copy()
            movement, direction = after-before, target-before
            mn, dn = np.linalg.norm(movement), np.linalg.norm(direction)
            trace['target_progress'].append(distance-float(np.linalg.norm(target-after)))
            trace['planar_speed'].append(float(mn/DT))
            trace['forward_velocity_mps'].append(float(np.dot(movement/DT, [np.cos(heading), np.sin(heading)])))
            trace['target_direction_error'].append(float(np.linalg.norm(movement/mn-direction/dn))
                                                    if mn > 1e-8 and dn > 1e-8 else 1.)
            if (step+1) % 1000 == 0:
                print(f'[Context evaluation] {controller["label"]} {step+1}/{job["steps"]} '
                      f'falls={recovery.fall_count} targets={goals}', flush=True)
        # Include a balance loss caused by the final action, without adding active steps.
        recovery.check_and_recover(agent)
        falls = recovery.fall_count
    finally:
        agent.terminate()
    arrays = {k: np.asarray(v, dtype=np.int64 if k=='episode_id' else np.float32) for k,v in trace.items()}
    np.savez_compressed(directory/'trace.npz', **arrays)
    progress = arrays['target_progress']
    summary = dict(row=row, controller=controller, supplied_slip=job['supplied_slip'],
        active_steps=job['steps'], active_seconds=job['steps']*DT,
        return_target_progress_m=float(progress.sum()),
        mean_forward_velocity_mps=float(arrays['forward_velocity_mps'].mean()),
        mean_planar_speed_mps=float(arrays['planar_speed'].mean()),
        mean_target_direction_error=float(arrays['target_direction_error'].mean()),
        targets_reached=goals, falls=falls, falls_per_active_minute=falls/(job['steps']*DT/60),
        on_policy_latent_action_mse=float(np.square(arrays['actions']-arrays['oracle_actions']).mean()))
    save(directory/'result.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    evaluate(read(args.job), args.job.parent)
