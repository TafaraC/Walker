"""One simulator client or one offline update per subprocess."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from variable_training.common import digest, read, save
from .models import load_policy, initialize
from context_training.models import policy_inputs


def collect(job, directory):
    from agent.Base_Agent import Base_Agent
    from behaviors.custom.Walk.Walk import Walk
    from bc_data import sample_own_half_target, sample_short_own_half_target
    from bc_sim import FallRecovery
    from collect_seed import reset_server_state
    from gail_models import GAILActorCritic
    from train import BCPolicy

    row, kind = job['row'], job['kind']
    actual_slip = float(f'{row["slip"]:.12g}') if row['slip'] is not None else .1
    np.random.seed(row['seed'])
    torch.manual_seed(row['seed'])
    mixing = np.random.default_rng(row['seed'] + 1)
    policy = None
    if kind != 'oracle':
        policy = load_policy(job['model'], kind, job.get('architecture', 'hidden'))
        policy.eval()
    reset_server_state()
    agent = Base_Agent(host='localhost', agent_port=3100, monitor_port=3200,
                       unum=1, robot_type=0, team_name='VariableSlip')
    walk, recovery = Walk(agent), FallRecovery()
    def target():
        if row['mode'] == 'short':
            return sample_short_own_half_target(agent.world.robot.loc_head_position[:2])
        return sample_own_half_target()
    destination, reset = target(), True
    data = {key: [] for key in ('obs', 'actions', 'executed_actions', 'episode_id',
                               'logprobs', 'values', 'dones')}
    episode, goals, controls = 0, 0, 0
    bootstrap = 0.
    start = time.monotonic()
    initial_step = agent.world.step
    try:
        for step in range(row['steps']):
            if recovery.check_and_recover(agent):
                if data['dones']:
                    data['dones'][-1] = True
                episode += 1
                destination, reset = target(), True
            robot = agent.world.robot
            distance = float(np.linalg.norm(destination - robot.loc_head_position[:2]))
            if distance < (.4 if row['mode'] == 'short' else .5):
                goals += 1
                destination = target()
                distance = float(np.linalg.norm(destination - robot.loc_head_position[:2]))
            obs = walk.observe(reset, destination, True, None, True, distance)
            net_obs = policy_inputs(obs, actual_slip, job.get('architecture', 'hidden'))
            value, logprob = 0., 0.
            if kind == 'gail':
                with torch.no_grad():
                    distribution, estimate = policy(torch.from_numpy(net_obs).unsqueeze(0))
                    sample = distribution.sample()
                    logprob = float(distribution.log_prob(sample).sum().item())
                    value = float(estimate.item())
                    executed = sample.squeeze(0).numpy()
                label = executed
            else:
                label = walk.get_expert_action(obs)
                use_oracle = kind == 'oracle' or mixing.random() < job.get('beta', 0.)
                if use_oracle:
                    executed = label
                    controls += 1
                else:
                    with torch.no_grad():
                        executed = policy(torch.from_numpy(net_obs).unsqueeze(0)).squeeze(0).numpy()
            for key, value_ in dict(obs=net_obs, actions=label, executed_actions=executed,
                                    episode_id=episode, logprobs=logprob, values=value, dones=False).items():
                data[key].append(np.copy(value_))
            walk.execute_action(np.asarray(executed, dtype=np.float32))
            agent.scom.commit_and_send(robot.get_command() or b'')
            agent.scom.receive()
            reset = False
            if (step + 1) % 1000 == 0:
                print(f'[Variable {kind}] {step+1}/{row["steps"]} falls={recovery.fall_count} targets={goals}', flush=True)
        # Detect a fall caused by the final action too; never bootstrap a fallen state.
        if recovery.check_and_recover(agent):
            data['dones'][-1] = True
        elif kind == 'gail':
            distance = float(np.linalg.norm(destination - agent.world.robot.loc_head_position[:2]))
            obs = walk.observe(False, destination, True, None, True, distance)
            net_obs = policy_inputs(obs, actual_slip, job.get('architecture', 'hidden'))
            with torch.no_grad():
                _, estimate = policy(torch.from_numpy(net_obs).unsqueeze(0))
                bootstrap = float(estimate.item())
        metrics = dict(active_steps=row['steps'], oracle_queries=0 if kind == 'gail' else row['steps'],
                       oracle_control_steps=controls, falls=recovery.fall_count, targets_reached=goals,
                       received_simulation_steps=agent.world.step,
                       startup_received_steps=initial_step,
                       rollout_and_recovery_steps=agent.world.step-initial_step,
                       wall_seconds=time.monotonic()-start, slip=row['slip'], mode=row['mode'],
                       model_sha256=digest(job['model']) if policy is not None else None)
    finally:
        agent.terminate()
    arrays = {key: np.asarray(value, dtype=(np.bool_ if key == 'dones' else
              np.int64 if key == 'episode_id' else np.float32)) for key, value in data.items()}
    np.savez_compressed(directory / 'data.npz', **arrays,
                        bootstrap_value=np.asarray(bootstrap, dtype=np.float32),
                        slip=np.asarray(actual_slip))
    save(directory / 'metrics.json', metrics)


def load_shards(paths, architecture='hidden'):
    from bc_data import load_dataset
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Dataset requires nonempty, unique shard paths')
    datasets = []
    for path in paths:
        x, y = load_dataset(path)
        if architecture == 'input' and x.shape[1] == 63:
            with np.load(path, allow_pickle=False) as shard:
                x = policy_inputs(x, float(shard['slip']), architecture)
        if x.shape[1] != (64 if architecture == 'input' else 63):
            raise ValueError('Observation/context dimension mismatch')
        datasets.append((x, y))
    return np.concatenate([x for x, _ in datasets]), np.concatenate([y for _, y in datasets])


def supervised(job, directory):
    from train import BCPolicy
    from torch.utils.data import DataLoader, TensorDataset
    if set(job['train']) & set(job['validation']):
        raise ValueError('Training and validation runs overlap')
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = load_shards(job['train'], job['architecture'])
    vx, vy = load_shards(job['validation'], job['architecture'])
    policy = BCPolicy(x.shape[1], 16).to(device)
    policy.load_state_dict(torch.load(job['model'], map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(policy.parameters(), lr=job['learning_rate'])
    if job.get('optimizer'):
        optimizer.load_state_dict(torch.load(job['optimizer'], map_location=device, weights_only=True))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=job['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(job['seed']))
    val = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)), batch_size=1024)
    best, updates, best_epoch = float('inf'), 0, 0
    for epoch in range(job['epochs']):
        policy.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = (policy(bx.to(device)) - by.to(device)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite supervised loss')
            loss.backward()
            optimizer.step()
            updates += 1
        policy.eval()
        with torch.no_grad():
            error = sum((policy(bx.to(device))-by.to(device)).square().sum().item() for bx, by in val)
        mse = error / vy.size
        if mse < best:
            best, best_epoch = mse, epoch + 1
            torch.save(policy.state_dict(), directory / 'policy.pth')
            torch.save(optimizer.state_dict(), directory / 'optimizer.pth')
        print(f'[Supervised] epoch={epoch+1} validation_mse={mse:.7f}', flush=True)
    save(directory / 'metrics.json', dict(validation_mse=best, best_epoch=best_epoch,
         train_samples=len(x), validation_samples=len(vx), gradient_updates=updates,
         sample_exposures=len(x)*job['epochs'], initialization=digest(job['model'])))


def gail(job, directory):
    from variable_training.gail_update import train_gail
    expert_x, expert_y = load_shards(job['expert'], job['architecture'])
    norm_x, norm_y = load_shards(job['normalizer'], job['architecture'])
    np.savez_compressed(directory / 'expert.npz', obs=expert_x, actions=expert_y)
    np.savez_compressed(directory / 'normalizer.npz', obs=norm_x, actions=norm_y)
    fields = {k: [] for k in ('obs', 'actions', 'logprobs', 'values', 'dones')}
    offsets, bootstraps = [0], []
    for path in job['rollouts']:
        with np.load(path, allow_pickle=False) as shard:
            for key in fields:
                fields[key].append(shard[key])
            offsets.append(offsets[-1] + len(shard['obs']))
            bootstraps.append(float(shard['bootstrap_value']))
    np.savez_compressed(directory / 'rollouts.npz', **{k: np.concatenate(v) for k,v in fields.items()},
                        segment_offsets=offsets, segment_bootstraps=bootstraps, bootstrap_value=0.)
    shutil.copy2(job['model'], directory / 'policy.pth')
    for key, name in [('discriminator', 'discriminator.pth'), ('optimizer', 'optimizer.pth')]:
        if job.get(key):
            shutil.copy2(job[key], directory / name)
    metrics = train_gail(job['iteration'], expert_file=str(directory / 'expert.npz'),
        rollout_file=str(directory / 'rollouts.npz'), model_path=str(directory / 'policy.pth'),
        discriminator_path=str(directory / 'discriminator.pth'), optimizer_path=str(directory / 'optimizer.pth'),
        init_model_path=job['model'], seed=job['seed'], label=job['label'],
        ppo_epochs=job.get('ppo_epochs', 3), discriminator_epochs=job.get('discriminator_epochs', 1),
        normalizer_file=str(directory / 'normalizer.npz'), metrics_path=str(directory / 'metrics.json'))
    metrics.update(segment_count=len(bootstraps), on_policy_checkpoint_sha256=digest(job['model']))
    save(directory / 'metrics.json', metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    path = Path(args.job).resolve()
    job = read(path)
    {'initialize': initialize, 'collect': collect, 'supervised': supervised, 'gail': gail}[job['task']](job, path.parent)


if __name__ == '__main__':
    main()
