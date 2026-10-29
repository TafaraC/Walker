import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Discriminator(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs raw logits (no Sigmoid)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

    def predict_reward(self, obs, action):
        """
        Non-saturating generator reward log(D(s,a)). With expert label 1,
        maximizing this reward moves policy occupancy toward expert occupancy.
        """
        with torch.no_grad():
            logits = self.forward(obs, action)
            return F.logsigmoid(logits)

class GAILActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(GAILActorCritic, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        # A small initial exploration scale is safer for locomotion than unit
        # variance, especially when actor means are initialized from DAgger.
        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -2.0))
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        mean = self.actor_mean(obs)
        std = torch.exp(torch.clamp(self.actor_log_std, -5.0, -1.0).expand_as(mean))
        dist = Normal(mean, std)
        value = self.critic(obs)
        return dist, value


def initialize_actor_from_imitation(policy, checkpoint_path, device="cpu"):
    """Initialize the GAIL actor mean from a BC/DAgger state dictionary."""
    state = torch.load(checkpoint_path, map_location=device)
    actor_state = policy.actor_mean.state_dict()
    mapping = {
        "0.weight": "net.0.weight", "0.bias": "net.0.bias",
        "2.weight": "net.2.weight", "2.bias": "net.2.bias",
        "4.weight": "net.4.weight", "4.bias": "net.4.bias",
    }
    converted = {actor_key: state[source_key] for actor_key, source_key in mapping.items()}
    for key, value in converted.items():
        if key not in actor_state or actor_state[key].shape != value.shape:
            raise ValueError(f"Incompatible imitation checkpoint tensor: {key}")
    policy.actor_mean.load_state_dict(converted)
