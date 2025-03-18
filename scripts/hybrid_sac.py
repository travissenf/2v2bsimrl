import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Categorical
import gym
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, cont_action, disc_action, reward, next_state, done):
        self.buffer.append((state, cont_action, disc_action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, cont_action, disc_action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state, dim=0).to(dtype=torch.float32),
            torch.stack(cont_action, dim=0).to(dtype=torch.float32),
            torch.tensor(list(disc_action), dtype=torch.int64),
            torch.tensor(list(reward), dtype=torch.float32),
            torch.stack(next_state, dim=0).to(dtype=torch.float32),
            torch.tensor(list(done), dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class HybridSACPolicy(nn.Module):
    def __init__(self, state_dim, continuous_dim, discrete_dim):
        super(HybridSACPolicy, self).__init__()
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        
        # base model
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # continuous
        self.cont_mean = nn.Linear(256, continuous_dim)
        self.cont_log_std = nn.Linear(256, continuous_dim)
        self.cont_log_std_min = -20
        self.cont_log_std_max = 2
        
        # discrete
        self.discrete_logits = nn.Linear(256, discrete_dim)
        
        # head for each discrete action
        self.conditional_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            ) for _ in range(discrete_dim)
        ])
        
        self.conditional_means = nn.ModuleList([
            nn.Linear(128, continuous_dim) for _ in range(discrete_dim)
        ])
        
        self.conditional_log_stds = nn.ModuleList([
            nn.Linear(128, continuous_dim) for _ in range(discrete_dim)
        ])
    
    def forward(self, state):
        x = self.shared(state)
        
        # pi(a^d | s)
        discrete_logits = self.discrete_logits(x)
        discrete_dist = Categorical(logits=discrete_logits)
        
        # pi(a^c | s)
        cont_mean = self.cont_mean(x)
        cont_log_std = torch.clamp(self.cont_log_std(x), self.cont_log_std_min, self.cont_log_std_max)
        cont_std = cont_log_std.exp()
        cont_dist = Normal(cont_mean, cont_std)
        
        #  pi(a^c | a^d, s) 
        conditional_dists = []
        for i in range(self.discrete_dim):
            cond_features = self.conditional_nets[i](state)
            cond_mean = self.conditional_means[i](cond_features)
            cond_log_std = torch.clamp(self.conditional_log_stds[i](cond_features), 
                                      self.cont_log_std_min, self.cont_log_std_max)
            cond_std = cond_log_std.exp()
            conditional_dists.append(Normal(cond_mean, cond_std))
        
        return discrete_dist, cont_dist, conditional_dists
    
    def sample_actions(self, state):
        discrete_dist, cont_dist, conditional_dists = self.forward(state)
        
        # sample from π(a^d | s)
        discrete_action = discrete_dist.sample()
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        
        batch_size = state.size(0)
        continuous_actions = []
        continuous_log_probs = []
        
        for i in range(batch_size):
            d_action = discrete_action[i].item()
            # sample from π(a^c | a^d, s)
            cond_action = conditional_dists[d_action].rsample() #breaks here, I have lost my mind debugging this
            cond_log_prob = conditional_dists[d_action].log_prob(cond_action.unsqueeze(0))
            
            continuous_actions.append(cond_action)
            continuous_log_probs.append(cond_log_prob)
            
        continuous_action = torch.stack(continuous_actions, dim=0)  
        continuous_log_prob = torch.cat(continuous_log_probs, dim=0)
        
        continuous_action_tanh = torch.tanh(continuous_action)
        
        log_prob_adjustment = torch.log(1 - continuous_action_tanh.pow(2) + 1e-6)
        adjusted_continuous_log_prob = continuous_log_prob - log_prob_adjustment
        adjusted_continuous_log_prob = adjusted_continuous_log_prob.sum(dim=-1)
        
        return continuous_action_tanh, discrete_action, adjusted_continuous_log_prob, discrete_log_prob
    
    def calculate_entropy(self, state, alpha_d, alpha_c):
        discrete_dist, _, conditional_dists = self.forward(state)

        # H(pi(a^d, a^c | s)) = alpha^d * H(pi(a^d | s)) + alpha^c * sum_{a^d} pi(a^d | s) * H(pi(a^c | a^d, s))
        # frist term
        discrete_entropy = discrete_dist.entropy()
        
        # get pi(a^d | s)
        discrete_probs = F.softmax(discrete_dist.logits, dim=-1)
        conditional_entropies = torch.zeros_like(discrete_entropy)
        
        for a_d in range(self.discrete_dim):
            # get pi(a^d | s) for this action
            p_a_d = discrete_probs[:, a_d]
            
            # get continuous entropy for action
            cond_dist_entropy = torch.sum(conditional_dists[a_d].entropy(), dim=-1)
            
            # add to sum
            conditional_entropies += p_a_d * cond_dist_entropy
        
        total_entropy = alpha_d * discrete_entropy + alpha_c * conditional_entropies
        
        return total_entropy, discrete_entropy, conditional_entropies

class HybridSACCritic(nn.Module):
    def __init__(self, state_dim, continuous_dim, discrete_dim):
        super(HybridSACCritic, self).__init__()
        self.discrete_dim = discrete_dim
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + continuous_dim + discrete_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + continuous_dim + discrete_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, continuous_action, discrete_action):
        if continuous_action.dim() == 3:
            continuous_action = continuous_action.squeeze(1)
        
        discrete_one_hot = F.one_hot(discrete_action, self.discrete_dim).float()
        
        sa = torch.cat([state, continuous_action, discrete_one_hot], 1)
        
        return self.q1(sa), self.q2(sa)

    def q1_value(self, state, continuous_action, discrete_action):
        if continuous_action.dim() == 3:
            continuous_action = continuous_action.squeeze(1)
            
        discrete_one_hot = F.one_hot(discrete_action, self.discrete_dim).float()
        sa = torch.cat([state, continuous_action, discrete_one_hot], 1)
        return self.q1(sa)

class HybridSAC:
    def __init__(self, state_dim,continuous_dim, discrete_dim,):
        self.lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.01
        self.alpha_d = 0.1
        self.alpha_c = 0.1
        self.batch_size = 256
        self.update_interval = 5
        self.auto_entropy_tuning = True
        self.buffer_size = 100000
        
        self.policy = HybridSACPolicy(state_dim, continuous_dim, discrete_dim)
        self.critic = HybridSACCritic(state_dim, continuous_dim, discrete_dim)
        self.critic_target = HybridSACCritic(state_dim, continuous_dim, discrete_dim)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        if self.auto_entropy_tuning:
            self.target_entropy_d = -1.0 * np.log(discrete_dim)
            self.target_entropy_c = -1.0 * continuous_dim
            
            self.log_alpha_d = torch.zeros(1, requires_grad=True)
            self.log_alpha_c = torch.zeros(1, requires_grad=True)
            self.alpha_d_optimizer = optim.Adam([self.log_alpha_d], lr=self.lr)
            self.alpha_c_optimizer = optim.Adam([self.log_alpha_c], lr=self.lr)
        
        self.update_count = 0
    
    def select_action(self, state, evaluate=False):
        """Select an action from the policy."""
        state = state.clone().detach().to(dtype=torch.float32).unsqueeze(0)
        
        if evaluate:
            discrete_dist, _, conditional_dists = self.policy.forward(state)
            
            discrete_action = torch.argmax(discrete_dist.probs, dim=1)
            
            d_action = discrete_action.item()
            continuous_action = conditional_dists[d_action].mean
            
            continuous_action = torch.tanh(continuous_action)
            
            return continuous_action.detach().cpu()[0], discrete_action.item()
        else:
            continuous_action, discrete_action, _, _ = self.policy.sample_actions(state)
            return continuous_action.detach().cpu()[0], discrete_action.item()
    
    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.update_count += 1
        
        if self.update_count % self.update_interval != 0:
            return
        
        state_batch, cont_action_batch, disc_action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        if self.auto_entropy_tuning:
            alpha_d = self.log_alpha_d.exp().item()
            alpha_c = self.log_alpha_c.exp().item()
        else:
            alpha_d = self.alpha_d
            alpha_c = self.alpha_c
        
        with torch.no_grad():
            next_cont_action, next_disc_action, next_cont_log_prob, next_disc_log_prob = self.policy.sample_actions(next_state_batch)
            
            next_entropy, _, _ = self.policy.calculate_entropy(next_state_batch, alpha_d, alpha_c)
            
            next_q1, next_q2 = self.critic_target(next_state_batch, next_cont_action, next_disc_action)
            next_q = torch.min(next_q1, next_q2)
            
            target_q = reward_batch + (1 - done_batch) * self.gamma * (next_q + next_entropy.unsqueeze(1))
        
        current_q1, current_q2 = self.critic(state_batch, cont_action_batch, disc_action_batch)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        sampled_cont_action, sampled_disc_action, cont_log_prob, disc_log_prob = self.policy.sample_actions(state_batch)
        
        entropy, discrete_entropy, conditional_entropy = self.policy.calculate_entropy(state_batch, alpha_d, alpha_c)
        
        q1 = self.critic.q1_value(state_batch, sampled_cont_action, sampled_disc_action)
        
        policy_loss = -(q1 + entropy.unsqueeze(1)).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        if self.auto_entropy_tuning:
            alpha_d_loss = -(self.log_alpha_d * (discrete_entropy.mean() + self.target_entropy_d).detach()).mean()
            alpha_c_loss = -(self.log_alpha_c * (conditional_entropy.mean() + self.target_entropy_c).detach()).mean()
            
            self.alpha_d_optimizer.zero_grad()
            alpha_d_loss.backward()
            self.alpha_d_optimizer.step()
            
            self.alpha_c_optimizer.zero_grad()
            alpha_c_loss.backward()
            self.alpha_c_optimizer.step()
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filename):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        
        if self.auto_entropy_tuning:
            torch.save({
                'log_alpha_d': self.log_alpha_d,
                'log_alpha_c': self.log_alpha_c,
                'alpha_d_optimizer_state_dict': self.alpha_d_optimizer.state_dict(),
                'alpha_c_optimizer_state_dict': self.alpha_c_optimizer.state_dict(),
            }, filename + "_alpha")
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_entropy_tuning:
            alpha_checkpoint = torch.load(filename + "_alpha")
            self.log_alpha_d = alpha_checkpoint['log_alpha_d']
            self.log_alpha_c = alpha_checkpoint['log_alpha_c']
            self.alpha_d_optimizer.load_state_dict(alpha_checkpoint['alpha_d_optimizer_state_dict'])
            self.alpha_c_optimizer.load_state_dict(alpha_checkpoint['alpha_c_optimizer_state_dict'])
