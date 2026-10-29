#!/usr/bin/env python3
"""Interactive, wall-clock-paced demonstrations of the frozen walking policies."""

import argparse
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
STEPS = 1500
HZ = 50
POLICIES = (
    ('FC Portugal expert', 'oracle', None),
    ('Behavioural cloning (BC)', 'bc', 'bc_walk_policy.pth'),
    ('Original DAgger', 'dagger', 'dagger_policy.pth'),
    ('Mixed-seed DAgger', 'dagger', 'dagger_mixed_policy.pth'),
    ('GAIL initialized from DAgger', 'gail', 'gail_dagger_policy.pth'),
    ('GAIL initialized from BC', 'gail', 'gail_bc_policy.pth'),
    ('GAIL initialized randomly', 'gail', 'gail_random_policy.pth'),
    ('Uniform-random action baseline', 'random', None),
)


class StepPacer:
    """Enforce a 20-ms minimum cycle, with no fast catch-up after stalls."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self.last = clock()

    def wait(self):
        remaining = self.last + 1.0 / HZ - self.clock()
        if remaining > 0:
            self.sleep(remaining)
        self.last = self.clock()


def choose_policy():
    print('\nWalking agent demo — 1,500 active steps at 50 Hz (30 seconds)')
    for number, (label, _, model) in enumerate(POLICIES, 1):
        unavailable = model is not None and not (ROOT / model).is_file()
        print(f'  {number}. {label}' + (' [checkpoint missing]' if unavailable else ''))
    print('  0. Exit')
    while True:
        try:
            choice = input('Choose an agent number: ').strip()
        except EOFError:
            return None
        if choice == '0':
            return None
        if not choice.isdecimal() or not 1 <= int(choice) <= len(POLICIES):
            print(f'Enter a number from 0 to {len(POLICIES)}.')
            continue
        policy = POLICIES[int(choice) - 1]
        if policy[2] is not None and not (ROOT / policy[2]).is_file():
            print(f'Checkpoint not found: {policy[2]}. Choose another agent.')
            continue
        return policy


def run_demo(policy, seed):
    # Delay native/simulator imports until selection and preflight have completed.
    import numpy as np
    import torch
    from agent.Base_Agent import Base_Agent
    from behaviors.custom.Walk.Walk import Walk
    from bc_data import load_dataset, sample_own_half_target
    from bc_sim import FallRecovery
    from evaluate_policy import load_policy, select_action

    class PacedRecovery(FallRecovery):
        def _sync_pose_cycle(self, agent):
            super()._sync_pose_cycle(agent)
            pacer.wait()

    label, kind, model_path = policy
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = load_policy(kind, model_path, 63, 16)
    random_low = random_high = None
    if kind == 'random':
        _, actions = load_dataset(ROOT / 'expert_train_data.npz')
        random_low, random_high = actions.min(axis=0), actions.max(axis=0)

    agent = Base_Agent(
        host='localhost', agent_port=3100, monitor_port=3200, unum=1,
        robot_type=0, team_name='WalkerDemo', enable_log=False,
        wait_for_server=False,
    )
    try:
        pacer = StepPacer()
        recovery = PacedRecovery()
        walk = Walk(agent)
        # Use the established monitor connection rather than optional trainer port 3101.
        agent.scom.unofficial_set_play_mode('PlayOn')
        recovery._recover(agent)
        if agent.world.play_mode != agent.world.M_PLAY_ON:
            raise RuntimeError('Server did not enter PlayOn mode; demo stopped.')
        target = sample_own_half_target()
        force_reset = True
        targets_reached = 0
        start = time.monotonic()
        pacer = StepPacer()
        print(f'\n[Demo] {label} | seed={seed} | 30 active seconds', flush=True)
        print('[Demo] Startup and fall recovery are additional time. Ctrl+C stops the demo.')
        for step in range(STEPS):
            if recovery.check_and_recover(agent):
                force_reset = True
                target = sample_own_half_target()
            robot = agent.world.robot
            distance = float(np.linalg.norm(target - robot.loc_head_position[:2]))
            if distance < 0.5:
                targets_reached += 1
                target = sample_own_half_target()
                distance = float(np.linalg.norm(target - robot.loc_head_position[:2]))
            obs = walk.observe(force_reset, target, True, None, True, distance)
            oracle_action = walk.get_expert_action(obs) if kind == 'oracle' else None
            action = select_action(kind, model, obs, oracle_action, rng, random_low, random_high)
            walk.execute_action(action)
            force_reset = False
            command = robot.get_command()
            agent.scom.commit_and_send(command if command else b'')
            agent.scom.receive()
            pacer.wait()
            if (step + 1) % 250 == 0:
                print(f'[Demo] {step + 1}/{STEPS} steps | {(step + 1) / HZ:.0f}/30 active s'
                      f' | targets={targets_reached} | falls={recovery.fall_count}', flush=True)
        elapsed = time.monotonic() - start
        print(f'[Demo] Finished: {STEPS} active steps, {elapsed:.1f}s elapsed including recovery.')
    finally:
        agent.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=101, help='Target/action seed (default: 101)')
    args = parser.parse_args()
    os.chdir(ROOT)  # Native resources and policy paths are repository-relative.
    try:
        policy = choose_policy()
        if policy is None:
            return 0
        if policy[1] == 'random' and not (ROOT / 'expert_train_data.npz').is_file():
            raise FileNotFoundError('The random baseline requires expert_train_data.npz.')
        from check_simulator_config import main as check_config
        if check_config() != 0:
            return 1
        from scripts.commons.Script import Script
        Script.build_cpp_modules()
        run_demo(policy, args.seed)
        return 0
    except KeyboardInterrupt:
        print('\n[Demo] Stopped.')
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f'[Demo Error] {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
