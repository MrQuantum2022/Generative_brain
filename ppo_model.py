import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Critic: Guesses the "Score" of the current state
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Actor: Decides the best action (Mean value)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Outputs between -1 and 1
        )
        
        # Action Randomness (Standard Deviation)
        # Learnable parameter to allow exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_action(self, state):
        """Selects an action based on state."""
        # --- Fix for NoneType error ---
        if state is None:
            raise ValueError("ActorCritic received 'None' state. The environment connection likely dropped.")
            
        # Ensure state is a float tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        
        # Create normal distribution and sample
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().numpy(), action_log_prob.detach()

    def evaluate(self, states, actions):
        """Used during training update."""
        action_mean = self.actor(states)
        action_std = self.log_std.exp()
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(actions).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(states)
        
        return action_logprobs, state_values.squeeze(), dist_entropy