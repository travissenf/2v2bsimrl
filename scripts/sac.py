import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import gym
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state, dim=0).to(dtype=torch.float32),
            torch.stack(action, dim=0).to(dtype=torch.float32),
            torch.tensor(list(reward), dtype=torch.float32),
            torch.stack(next_state, dim=0).to(dtype=torch.float32),
            torch.tensor(list(done), dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class ContinuousSACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=1.0, hidden_dim=256):
        super(ContinuousSACPolicy, self).__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.log_std_min = -20
        self.log_std_max = 2
        
        # Policy network
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log_std outputs
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability with change of variables
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds (tanh squashing correction)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Scale the action to environment action space
        scaled_action = action * self.action_scale
        
        return scaled_action, log_prob
    
    def get_action(self, state, evaluate=False):
        state = state.clone().detach().to(dtype=torch.float32).unsqueeze(0)
        
        if evaluate:
            # For evaluation, use the mean action
            with torch.no_grad():
                mean, _ = self.forward(state)
                action = torch.tanh(mean) * self.action_scale
            return action.detach().cpu()[0]
        else:
            # For training/exploration, sample from the distribution
            with torch.no_grad():
                action, _ = self.sample(state)
            return action.detach().cpu()[0]

class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 architecture (for twin Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        # Concatenate state and action
        sa = torch.cat([state, action], 1)
        
        return self.q1(sa), self.q2(sa)
    
    def q1_value(self, state, action):
        sa = torch.cat([state, action], 1)
        
        return self.q1(sa)

class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_scale=1.0,
        hidden_dim=256,
        buffer_size=1000000,
        batch_size=256,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        update_interval=1,
        auto_entropy_tuning=True,
        target_entropy_scale=-1.0
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Initialize actor and critic networks
        self.policy = ContinuousSACPolicy(state_dim, action_dim, action_scale, hidden_dim)
        self.critic = SACCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = SACCritic(state_dim, action_dim, hidden_dim)
        
        # Copy parameters from critic to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Automatic entropy tuning
        if auto_entropy_tuning:
            self.target_entropy = target_entropy_scale * action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        self.update_count = 0
    
    def select_action(self, state, evaluate=False):
        """Select an action from the policy."""
        return self.policy.get_action(state, evaluate)
    
    def update_parameters(self):
        """Update the networks' parameters using SAC."""
        # Skip update if buffer doesn't have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Increment update counter
        self.update_count += 1
        
        # Only update every update_interval steps
        if self.update_count % self.update_interval != 0:
            return
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Get current alpha value
        if self.auto_entropy_tuning:
            alpha = self.log_alpha.exp().item()
        else:
            alpha = self.alpha
        
        # Update critic networks
        with torch.no_grad():
            # Sample next actions from the policy
            next_state_action, next_state_log_prob = self.policy.sample(next_state_batch)
            
            # Target Q-values
            next_q1, next_q2 = self.critic_target(next_state_batch, next_state_action)
            next_q = torch.min(next_q1, next_q2)
            
            # Include entropy in the target (maximum entropy RL)
            target_q = reward_batch + (1 - done_batch) * self.gamma * (next_q + alpha * next_state_log_prob)
        
        # Current Q-values
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        
        # Compute critic loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update policy network
        # Get actions and log probabilities from the policy (reparameterized for gradient)
        pi, log_pi = self.policy.sample(state_batch)
        
        # Calculate Q-value for policy actions
        q1_pi = self.critic.q1_value(state_batch, pi)
        
        # Policy loss with maximum entropy
        policy_loss = (alpha * log_pi - q1_pi).mean()
        
        # Optimize the policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update automatic entropy adjustment (if enabled)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filename):
        """Save the model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        
        if self.auto_entropy_tuning:
            torch.save({
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
            }, filename + "_alpha")
    
    def load(self, filename):
        """Load the model."""
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_entropy_tuning:
            alpha_checkpoint = torch.load(filename + "_alpha")
            self.log_alpha = alpha_checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(alpha_checkpoint['alpha_optimizer_state_dict'])