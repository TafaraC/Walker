"""Adapted stabilized trainer: independent segment GAE and fixed normalization pool."""
import argparse
import copy
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from bc_data import load_dataset, save_json_atomic
from gail_models import Discriminator, GAILActorCritic, initialize_actor_from_imitation


def compute_gae(rewards, values, dones, bootstrap_value, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = float(bootstrap_value)
    for index in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * mask - values[index]
        last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        advantages[index] = last_advantage
        next_value = values[index]
    return advantages, advantages + values


def gradient_penalty(discriminator, expert_obs, expert_actions, policy_obs, policy_actions):
    batch_size = len(expert_obs)
    alpha = torch.rand(batch_size, 1, device=expert_obs.device)
    mixed_obs = (alpha * expert_obs + (1.0 - alpha) * policy_obs).requires_grad_(True)
    mixed_actions = (alpha * expert_actions + (1.0 - alpha) * policy_actions).requires_grad_(True)
    logits = discriminator(mixed_obs, mixed_actions)
    obs_gradient, action_gradient = torch.autograd.grad(
        outputs=logits,
        inputs=(mixed_obs, mixed_actions),
        grad_outputs=torch.ones_like(logits),
        create_graph=True,
    )
    gradient = torch.cat((obs_gradient, action_gradient), dim=-1)
    return ((gradient.norm(2, dim=-1) - 1.0) ** 2).mean()


def full_policy_kl(policy, observations, actions, old_logprobs, batch_size, device):
    parts = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            end = start + batch_size
            distribution, _ = policy(observations[start:end].to(device))
            new_logprob = distribution.log_prob(actions[start:end].to(device)).sum(dim=-1)
            log_ratio = new_logprob - old_logprobs[start:end].to(device)
            safe_log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
            parts.append(((torch.exp(safe_log_ratio) - 1.0) - safe_log_ratio).cpu())
    return float(torch.cat(parts).mean().item())


def train_gail(
    iteration,
    expert_file="expert_train_data.npz",
    rollout_file="gail_rollout_batch.npz",
    model_path="gail_policy.pth",
    discriminator_path="gail_disc.pth",
    init_model_path="dagger_policy.pth",
    optimizer_path="gail_optimizer_state.pth",
    ppo_epochs=3,
    discriminator_epochs=1,
    batch_size=256,
    actor_learning_rate=3e-6,
    critic_learning_rate=3e-4,
    discriminator_learning_rate=1e-4,
    entropy_coefficient=1e-3,
    bc_coefficient=0.05,
    gradient_penalty_coefficient=10.0,
    target_kl=0.02,
    rollback_kl=0.05,
    seed=42,
    metrics_path=None,
    label="gail_dagger",
    normalizer_file=None,
):
    expert_obs_np, expert_actions_np = load_dataset(expert_file)
    if not os.path.exists(rollout_file):
        raise FileNotFoundError(f"GAIL rollout not found: {rollout_file}")
    with np.load(rollout_file, allow_pickle=False) as rollout:
        required = ("obs", "actions", "logprobs", "values", "dones", "bootstrap_value")
        missing = [key for key in required if key not in rollout.files]
        if missing:
            raise ValueError(f"Rollout is missing arrays: {', '.join(missing)}")
        policy_obs_np = np.asarray(rollout["obs"], dtype=np.float32)
        policy_actions_np = np.asarray(rollout["actions"], dtype=np.float32)
        old_logprobs_np = np.asarray(rollout["logprobs"], dtype=np.float32)
        old_values_np = np.asarray(rollout["values"], dtype=np.float32)
        dones_np = np.asarray(rollout["dones"], dtype=np.bool_)
        bootstrap_value = float(np.asarray(rollout["bootstrap_value"]))
        offsets = np.asarray(rollout["segment_offsets"], dtype=np.int64)
        bootstraps = np.asarray(rollout["segment_bootstraps"], dtype=np.float32)

    count = len(policy_obs_np)
    if count == 0:
        raise ValueError("GAIL rollout is empty")
    if policy_obs_np.shape != (count, expert_obs_np.shape[1]):
        raise ValueError(f"Unexpected rollout observation shape: {policy_obs_np.shape}")
    if policy_actions_np.shape != (count, expert_actions_np.shape[1]):
        raise ValueError(f"Unexpected rollout action shape: {policy_actions_np.shape}")
    if old_logprobs_np.shape != (count,) or old_values_np.shape != (count,) or dones_np.shape != (count,):
        raise ValueError("Rollout scalar arrays must contain one value per step")
    if not all(np.isfinite(array).all() for array in (
        policy_obs_np, policy_actions_np, old_logprobs_np, old_values_np
    )) or not np.isfinite(bootstrap_value):
        raise ValueError("GAIL rollout contains NaN or infinite values")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim, action_dim = expert_obs_np.shape[1], expert_actions_np.shape[1]
    policy = GAILActorCritic(obs_dim, action_dim).to(device)
    discriminator = Discriminator(obs_dim, action_dim).to(device)
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location=device))
    elif init_model_path != "random":
        initialize_actor_from_imitation(policy, init_model_path, device)
    if os.path.exists(discriminator_path):
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))

    actor_parameters = list(policy.actor_mean.parameters()) + [policy.actor_log_std]
    actor_optimizer = optim.Adam(actor_parameters, lr=actor_learning_rate)
    critic_optimizer = optim.Adam(policy.critic.parameters(), lr=critic_learning_rate)
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=discriminator_learning_rate
    )
    if os.path.exists(optimizer_path):
        state = torch.load(optimizer_path, map_location=device)
        if all(key in state for key in ("actor", "critic", "discriminator")):
            actor_optimizer.load_state_dict(state["actor"])
            critic_optimizer.load_state_dict(state["critic"])
            discriminator_optimizer.load_state_dict(state["discriminator"])
        else:
            print("[GAIL Train] Ignoring optimizer state from the older optimizer layout")

    expert_obs = torch.from_numpy(expert_obs_np)
    expert_actions = torch.from_numpy(expert_actions_np)
    policy_obs = torch.from_numpy(policy_obs_np)
    policy_actions = torch.from_numpy(policy_actions_np)
    old_logprobs = torch.from_numpy(old_logprobs_np)

    # Normalize only the discriminator domain using fixed expert statistics.
    # Actor/critic observations retain the FC Portugal representation.
    norm_obs, norm_actions = load_dataset(normalizer_file or expert_file)
    obs_mean = torch.from_numpy(norm_obs).mean(dim=0)
    obs_std = torch.from_numpy(norm_obs).std(dim=0).clamp_min(1e-3)
    action_mean = torch.from_numpy(norm_actions).mean(dim=0)
    action_std = torch.from_numpy(norm_actions).std(dim=0).clamp_min(1e-3)
    normalized_expert_obs = (expert_obs - obs_mean) / obs_std
    normalized_expert_actions = (expert_actions - action_mean) / action_std
    normalized_policy_obs = (policy_obs - obs_mean) / obs_std
    normalized_policy_actions = (policy_actions - action_mean) / action_std
    bce = nn.BCEWithLogitsLoss()

    discriminator.train()
    discriminator_losses, classification_losses, penalties = [], [], []
    batches_per_epoch = max(1, math.ceil(count / batch_size))
    discriminator_early_stopped = False
    for _ in range(discriminator_epochs):
        for _ in range(batches_per_epoch):
            expert_indices = torch.randint(0, len(expert_obs), (batch_size,))
            policy_indices = torch.randint(0, count, (batch_size,))
            exp_s = normalized_expert_obs[expert_indices].to(device)
            exp_a = normalized_expert_actions[expert_indices].to(device)
            pol_s = normalized_policy_obs[policy_indices].to(device)
            pol_a = normalized_policy_actions[policy_indices].to(device)
            expert_logits = discriminator(exp_s, exp_a)
            policy_logits = discriminator(pol_s, pol_a)
            classification_loss = (
                bce(expert_logits, torch.full_like(expert_logits, 0.9))
                + bce(policy_logits, torch.full_like(policy_logits, 0.1))
            )
            penalty = gradient_penalty(discriminator, exp_s, exp_a, pol_s, pol_a)
            discriminator_loss = classification_loss + gradient_penalty_coefficient * penalty
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            discriminator_optimizer.step()
            discriminator_losses.append(float(discriminator_loss.item()))
            classification_losses.append(float(classification_loss.item()))
            penalties.append(float(penalty.item()))

        with torch.no_grad():
            sample_size = min(2048, count, len(expert_obs))
            exp_accuracy = (discriminator(
                normalized_expert_obs[:sample_size].to(device),
                normalized_expert_actions[:sample_size].to(device),
            ) > 0).float().mean().item()
            pol_accuracy = (discriminator(
                normalized_policy_obs[:sample_size].to(device),
                normalized_policy_actions[:sample_size].to(device),
            ) < 0).float().mean().item()
        if exp_accuracy > 0.95 and pol_accuracy > 0.95:
            discriminator_early_stopped = True
            break

    discriminator.eval()
    reward_parts = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = start + batch_size
            reward_parts.append(discriminator.predict_reward(
                normalized_policy_obs[start:end].to(device),
                normalized_policy_actions[start:end].to(device),
            ).squeeze(-1).cpu())
    rewards_np = torch.cat(reward_parts).numpy()
    advantages_np, returns_np = segmented_gae(
        rewards_np, old_values_np, dones_np, offsets, bootstraps
    )
    advantages = torch.from_numpy(advantages_np)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.from_numpy(returns_np)
    loader = DataLoader(
        TensorDataset(policy_obs, policy_actions, old_logprobs, advantages, returns),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    initial_actor_state = {
        key: value.detach().cpu().clone() for key, value in policy.actor_mean.state_dict().items()
    }
    initial_log_std = policy.actor_log_std.detach().cpu().clone()
    initial_actor_optimizer_state = copy.deepcopy(actor_optimizer.state_dict())
    effective_bc_coefficient = bc_coefficient * max(0.1, 1.0 - (iteration - 1) / 10.0)
    actor_losses, critic_losses, bc_losses = [], [], []
    entropies, approximate_kls, clip_fractions = [], [], []
    clip_epsilon = 0.2
    epochs_completed = 0
    target_kl_stopped = False
    policy.train()
    for _ in range(ppo_epochs):
        for batch_obs, batch_actions, batch_old_logprob, batch_advantage, batch_return in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_old_logprob = batch_old_logprob.to(device)
            batch_advantage = batch_advantage.to(device)
            batch_return = batch_return.to(device)
            distribution, _ = policy(batch_obs)
            new_logprobs = distribution.log_prob(batch_actions).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1).mean()
            log_ratio = new_logprobs - batch_old_logprob
            ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
            actor_loss = -torch.min(
                ratio * batch_advantage,
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantage,
            ).mean()

            expert_indices = torch.randint(0, len(expert_obs), (len(batch_obs),))
            anchor_obs = expert_obs[expert_indices].to(device)
            anchor_actions = expert_actions[expert_indices].to(device)
            bc_loss = nn.functional.mse_loss(policy.actor_mean(anchor_obs), anchor_actions)
            actor_objective = (
                actor_loss - entropy_coefficient * entropy
                + effective_bc_coefficient * bc_loss
            )
            actor_optimizer.zero_grad()
            actor_objective.backward()
            nn.utils.clip_grad_norm_(actor_parameters, 0.5)
            actor_optimizer.step()

            predicted_values = policy.critic(batch_obs).squeeze(-1)
            critic_loss = nn.functional.mse_loss(predicted_values, batch_return)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(policy.critic.parameters(), 0.5)
            critic_optimizer.step()

            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(critic_loss.item()))
            bc_losses.append(float(bc_loss.item()))
            entropies.append(float(entropy.item()))
            approximate_kls.append(float(((ratio - 1.0) - log_ratio).mean().item()))
            clip_fractions.append(float((torch.abs(ratio - 1.0) > clip_epsilon).float().mean().item()))

        epochs_completed += 1
        current_kl = full_policy_kl(
            policy, policy_obs, policy_actions, old_logprobs, batch_size, device
        )
        if current_kl > target_kl:
            target_kl_stopped = True
            print(
                f"[GAIL Train] PPO early stop after epoch {epochs_completed}: "
                f"KL {current_kl:.5f} > target {target_kl:.5f}"
            )
            break

    final_kl = full_policy_kl(
        policy, policy_obs, policy_actions, old_logprobs, batch_size, device
    )
    actor_rolled_back = final_kl > rollback_kl
    if actor_rolled_back:
        policy.actor_mean.load_state_dict(initial_actor_state)
        policy.actor_log_std.data.copy_(initial_log_std.to(device))
        actor_optimizer.load_state_dict(initial_actor_optimizer_state)
        print(
            f"[GAIL Train] Unsafe actor update rolled back: "
            f"KL {final_kl:.5f} > {rollback_kl:.5f}"
        )
        accepted_kl = 0.0
    else:
        accepted_kl = final_kl

    torch.save(policy.state_dict(), model_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    torch.save({
        "actor": actor_optimizer.state_dict(),
        "critic": critic_optimizer.state_dict(),
        "discriminator": discriminator_optimizer.state_dict(),
    }, optimizer_path)
    model = Path(model_path)
    disc = Path(discriminator_path)
    shutil.copy2(model, model.with_name(f"{model.stem}_iter_{iteration}{model.suffix}"))
    shutil.copy2(disc, disc.with_name(f"{disc.stem}_iter_{iteration}{disc.suffix}"))

    discriminator.eval()
    policy.eval()
    with torch.no_grad():
        evaluation_count = min(len(expert_obs), 20000)
        exp_logits = discriminator(
            normalized_expert_obs[:evaluation_count].to(device),
            normalized_expert_actions[:evaluation_count].to(device),
        )
        pol_logits = discriminator(
            normalized_policy_obs.to(device), normalized_policy_actions.to(device)
        )
        actor_means = policy.actor_mean(expert_obs[:evaluation_count].to(device))
        imitation_mse = nn.functional.mse_loss(
            actor_means, expert_actions[:evaluation_count].to(device)
        ).item()

    metrics = {
        "algorithm": "gail",
        "policy_label": label,
        "actor_initialization": init_model_path,
        "iteration": iteration,
        "seed": seed,
        "expert_samples": len(expert_obs),
        "rollout_samples": count,
        "reward_type": "log_D_non_saturating",
        "discriminator_updates": len(discriminator_losses),
        "discriminator_early_stopped": discriminator_early_stopped,
        "discriminator_loss": float(np.mean(discriminator_losses)),
        "discriminator_classification_loss": float(np.mean(classification_losses)),
        "gradient_penalty": float(np.mean(penalties)),
        "discriminator_expert_accuracy": float((exp_logits > 0).float().mean().item()),
        "discriminator_policy_accuracy": float((pol_logits < 0).float().mean().item()),
        "learned_reward_mean": float(np.mean(rewards_np)),
        "learned_reward_std": float(np.std(rewards_np)),
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "bc_anchor_loss": float(np.mean(bc_losses)),
        "bc_anchor_coefficient": effective_bc_coefficient,
        "entropy": float(np.mean(entropies)),
        "sampled_minibatch_approximate_kl": float(np.mean(approximate_kls)),
        "pre_rollback_full_kl": final_kl,
        "accepted_full_kl": accepted_kl,
        "clip_fraction": float(np.mean(clip_fractions)),
        "target_kl_early_stopped": target_kl_stopped,
        "actor_update_rolled_back": actor_rolled_back,
        "ppo_epochs_completed": epochs_completed,
        "actor_log_std_mean": float(policy.actor_log_std.mean().item()),
        "expert_dataset_actor_mean_mse": float(imitation_mse),
    }
    metrics_path = metrics_path or f"gail_training_metrics_iter_{iteration}.json"
    save_json_atomic(metrics_path, metrics)
    print(
        f"[GAIL Train] Disc Class Loss: {metrics['discriminator_classification_loss']:.4f} "
        f"| Reward: {metrics['learned_reward_mean']:.4f} "
        f"| Accepted KL: {accepted_kl:.5f}"
    )
    print(f"[GAIL Train] Metrics saved to '{metrics_path}'")
    return metrics


def segmented_gae(rewards, values, dones, offsets, bootstraps):
    if (len(offsets) != len(bootstraps) + 1 or offsets[0] != 0
            or offsets[-1] != len(rewards) or np.any(np.diff(offsets) <= 0)
            or not np.isfinite(bootstraps).all()):
        raise ValueError("Invalid segment boundaries or bootstraps")
    parts = [compute_gae(rewards[a:b], values[a:b], dones[a:b], bootstrap)
             for a, b, bootstrap in zip(offsets[:-1], offsets[1:], bootstraps)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

