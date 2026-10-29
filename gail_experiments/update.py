"""Noise-matched adversarial imitation with guarded, adaptive PPO updates."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence
from gail_models import GAILActorCritic, Discriminator
from consolidated.worker import load_shards
from variable_training.gail_update import segmented_gae
from variable_training.common import save, digest


def exact_kl(policy, observations, old_mean, old_std, batch=1024):
    device=next(policy.parameters()).device
    parts=[]
    with torch.no_grad():
        for i in range(0,len(observations),batch):
            new,_=policy(observations[i:i+batch].to(device))
            old=torch.distributions.Normal(old_mean[i:i+batch].to(device),old_std[i:i+batch].to(device))
            parts.append(kl_divergence(old,new).sum(-1).cpu())
    return float(torch.cat(parts).mean())


def actor_snapshot(policy, optimizer):
    return (copy.deepcopy(policy.actor_mean.state_dict()),policy.actor_log_std.detach().clone(),
            copy.deepcopy(optimizer.state_dict()))


def restore_actor(policy,optimizer,snapshot):
    policy.actor_mean.load_state_dict(snapshot[0])
    with torch.no_grad(): policy.actor_log_std.copy_(snapshot[1])
    optimizer.load_state_dict(snapshot[2])


def load_rollouts(paths):
    fields={k:[] for k in ('obs','actions','logprobs','values','dones')}
    offsets,boots=[0],[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            n=len(data['obs'])
            if not n or not np.array_equal(data['actions'],data['executed_actions']):
                raise ValueError('Empty rollout or action/log-probability contract changed')
            for k in fields:
                arr=data[k].copy()
                if len(arr)!=n or not np.isfinite(arr).all(): raise ValueError(f'Invalid rollout {k}')
                fields[k].append(arr)
            offsets.append(offsets[-1]+n);boots.append(float(data['bootstrap_value']))
    result={k:np.concatenate(v) for k,v in fields.items()}
    if result['obs'].shape!=(offsets[-1],63) or result['actions'].shape!=(offsets[-1],16):
        raise ValueError('This experiment uses hidden 63D observations and 16D gait actions')
    return result,np.asarray(offsets),np.asarray(boots)


def train(job,directory):
    cfg=job['settings']
    torch.manual_seed(job['seed']);np.random.seed(job['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy=GAILActorCritic(63,16).to(device)
    policy.load_state_dict(torch.load(job['model'],map_location=device,weights_only=True))
    disc=Discriminator(63,16).to(device)
    if job.get('discriminator'):
        disc.load_state_dict(torch.load(job['discriminator'],map_location=device,weights_only=True))
    actor_params=list(policy.actor_mean.parameters())+[policy.actor_log_std]
    actor=torch.optim.Adam(actor_params,lr=cfg['actor_lr'])
    critic=torch.optim.Adam(policy.critic.parameters(),lr=cfg['critic_lr'])
    discriminator=torch.optim.Adam(disc.parameters(),lr=cfg['disc_lr'])
    if job.get('optimizer'):
        state=torch.load(job['optimizer'],map_location=device,weights_only=True)
        for opt,key in [(actor,'actor'),(critic,'critic'),(discriminator,'discriminator')]:opt.load_state_dict(state[key])
    x,y=load_shards(job['expert'])
    nx,ny=load_shards(job['normalizer'])
    raw,offsets,boots=load_rollouts(job['rollouts'])
    count=len(raw['obs']);batch=cfg['batch_size']
    ex,ea=torch.from_numpy(x),torch.from_numpy(y)
    obs,actions,old_lp=[torch.from_numpy(raw[k]) for k in ('obs','actions','logprobs')]
    mean=torch.from_numpy(np.concatenate([nx,ny],axis=1).mean(0))
    scale=torch.from_numpy(np.concatenate([nx,ny],axis=1).std(0)).clamp_min(.05)
    old_means,old_stds,logprobs=[],[],[]
    with torch.no_grad():
        for i in range(0,count,batch):
            distribution,_=policy(obs[i:i+batch].to(device))
            old_means.append(distribution.mean.cpu());old_stds.append(distribution.stddev.cpu())
            logprobs.append(distribution.log_prob(actions[i:i+batch].to(device)).sum(-1).cpu())
    old_mean,old_std=torch.cat(old_means),torch.cat(old_stds)
    logprob_error=float((torch.cat(logprobs)-old_lp).abs().max())
    if logprob_error > .005: raise ValueError(f'Rollouts are not from the supplied actor: logprob error {logprob_error}')
    # Expert actions are deterministic. Match the learner's current exploration
    # covariance so the discriminator cannot win solely by detecting action noise.
    exploration_std=old_std[0]
    norm_policy=(torch.cat([obs,actions],-1)-mean)/scale
    def logits(z):return disc(z[:,:63],z[:,63:]).squeeze(-1)
    disc_losses=[];penalties=[];acc_exp=acc_pol=0.
    updates=max(1,math.ceil(count/batch))*cfg['disc_epochs']
    for i in range(updates):
        ei=torch.randint(len(ex),(batch,));pi=torch.randint(count,(batch,))
        noisy_actions=ea[ei]+torch.randn(batch,16)*exploration_std
        real=((torch.cat([ex[ei],noisy_actions],-1)-mean)/scale).to(device)
        fake=norm_policy[pi].to(device)
        real.requires_grad_(True)
        rl,fl=logits(real),logits(fake)
        classification=F.softplus(-rl).mean()+F.softplus(fl).mean()
        gradients=torch.autograd.grad(rl.sum(),real,create_graph=True)[0]
        penalty=gradients.square().sum(-1).mean()
        loss=classification+.5*cfg['r1_coefficient']*penalty
        if not torch.isfinite(loss):raise ValueError('Nonfinite discriminator loss')
        discriminator.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_(disc.parameters(),5.);discriminator.step()
        disc_losses.append(float(classification.detach()));penalties.append(float(penalty.detach()))
        # Stop based on a fresh random draw (it may overlap training batches), not
        # held-out experimental evaluation data (this remains a training diagnostic).
        if (i+1)%max(1,math.ceil(count/batch))==0:
            with torch.no_grad():
                ei=torch.randint(len(ex),(min(2048,len(ex)),));pi=torch.randint(count,(len(ei),))
                real=(torch.cat([ex[ei],ea[ei]+torch.randn(len(ei),16)*exploration_std],-1)-mean)/scale
                acc_exp=float((logits(real.to(device))>0).float().mean())
                acc_pol=float((logits(norm_policy[pi].to(device))<0).float().mean())
            if min(acc_exp,acc_pol)>.85:break
    rewards=[]
    with torch.no_grad():
        for i in range(0,count,batch):
            # Positive occupancy reward avoids negative per-step reward making
            # early fall termination intrinsically attractive. No task reward.
            rewards.append(F.softplus(logits(norm_policy[i:i+batch].to(device))).clamp(max=10).cpu())
    rewards=torch.cat(rewards).numpy()
    adv,ret=segmented_gae(rewards,raw['values'],raw['dones'],offsets,boots)
    advantages=torch.from_numpy((adv-adv.mean())/(adv.std()+1e-8))
    returns=torch.from_numpy(ret)
    start_lr=actor.param_groups[0]['lr'];attempts=0;accepted_epochs=0;rejected=0
    losses=[];value_losses=[]
    for epoch in range(cfg['ppo_epochs']):
        snapshot=actor_snapshot(policy,actor)
        accepted=False
        for retry in range(4):
            lr=actor.param_groups[0]['lr'];attempts+=1
            for idx in torch.randperm(count).split(batch):
                distribution,_=policy(obs[idx].to(device))
                logratio=distribution.log_prob(actions[idx].to(device)).sum(-1)-old_lp[idx].to(device)
                ratio=logratio.clamp(-20,20).exp()
                advantage=advantages[idx].to(device)
                loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                loss-=cfg['entropy_coefficient']*distribution.entropy().sum(-1).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite actor loss')
                actor.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor_params,.5);actor.step()
                with torch.no_grad():policy.actor_log_std.clamp_(-5.,-1.)
                losses.append(float(loss.detach()))
            kl=exact_kl(policy,obs,old_mean,old_std,batch)
            if math.isfinite(kl) and kl<=cfg['max_kl']:
                accepted=True;accepted_epochs+=1;break
            rejected+=1;restore_actor(policy,actor,snapshot)
            for g in actor.param_groups:g['lr']=max(cfg['min_actor_lr'],lr*.5)
        if not accepted or kl>=cfg['target_kl']:break
    # Critic training is independent of rejected actor epochs.
    for _ in range(cfg['critic_epochs']):
        for idx in torch.randperm(count).split(batch):
            value=policy.critic(obs[idx].to(device)).squeeze(-1)
            loss=F.mse_loss(value,returns[idx].to(device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite critic loss')
            critic.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.critic.parameters(),.5);critic.step()
            value_losses.append(float(loss.detach()))
    accepted_kl=exact_kl(policy,obs,old_mean,old_std,batch)
    if not math.isfinite(accepted_kl) or accepted_kl>cfg['max_kl']+1e-6:
        raise ValueError('Final actor violated exact KL bound; refusing checkpoint')
    # Adapt the next rollout's learning rate; no optimizer state is discarded.
    factor=1.25 if accepted_kl<cfg['target_kl']*.5 and not rejected else 1.
    for g in actor.param_groups:g['lr']=min(cfg['max_actor_lr'],max(cfg['min_actor_lr'],g['lr']*factor))
    with torch.no_grad():
        mse=[];shifts=[]
        for i in range(0,len(ex),batch):
            mse.append((policy.actor_mean(ex[i:i+batch].to(device))-ea[i:i+batch].to(device)).square().sum().item())
        for i in range(0,count,batch):
            shifts.append((policy.actor_mean(obs[i:i+batch].to(device))-old_mean[i:i+batch].to(device)).square().sum().item())
    torch.save(policy.state_dict(),directory/'policy.pth')
    torch.save(disc.state_dict(),directory/'discriminator.pth')
    torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),discriminator=discriminator.state_dict()),directory/'optimizer.pth')
    metrics=dict(recipe='adaptive',iteration=job['iteration'],rollout_samples=count,
        reward_type='clipped_softplus_logit',reward_mean=float(rewards.mean()),reward_std=float(rewards.std()),
        discriminator_classification_loss=float(np.mean(disc_losses)),r1_penalty=float(np.mean(penalties)),
        discriminator_updates=len(disc_losses),expert_accuracy=acc_exp,policy_accuracy=acc_pol,
        accepted_full_kl=accepted_kl,accepted_epochs=accepted_epochs,rejected_epoch_attempts=rejected,
        actor_lr_start=start_lr,actor_lr_next=actor.param_groups[0]['lr'],actor_attempts=attempts,
        critic_loss=float(np.mean(value_losses)),expert_train_mse=sum(mse)/y.size,
        initial_exploration_std=float(exploration_std.mean()),final_exploration_std=float(policy.actor_log_std.detach().exp().mean()),
        anchor_coefficient=0.,actor_mean_shift_mse=sum(shifts)/(count*16),logprob_reproduction_max_error=logprob_error,
        on_policy_checkpoint_sha256=digest(job['model']),segment_count=len(boots))
    save(directory/'metrics.json',metrics)
    print(f'[Adaptive GAIL] KL={accepted_kl:.5f} reward={rewards.mean():.4f} '
          f'accepted_epochs={accepted_epochs} rejected={rejected}',flush=True)
