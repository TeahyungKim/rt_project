import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
import numpy as np
import random
import matplotlib.pyplot as plt

#===============================================================================
# global variables
#===============================================================================
seed = 1
gamma = 0.99
tau = 0.005
alpha = 0.2
lr = 0.0003
hidden_size = 256
epsilon = 1e-6
replay_size = 1000000
start_steps = 10000
updates_per_step = 1
batch_size = 256
num_steps = 1000000

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

class SAC(object):
    def __init__(self, num_inputs, action_space):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_range = [action_space.low, action_space.high]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.value = ValueNetwork(num_inputs, hidden_size).to(device=self.device)
        self.value_target = ValueNetwork(num_inputs, hidden_size).to(self.device)
        self.value_optim = Adam(self.value.parameters(), lr=lr)
        hard_update(self.value_target, self.value)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _, _ = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        return self.rescale_action(action)

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
                (self.action_range[1] + self.action_range[0]) / 2.0

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            vf_next_target = self.value_target(next_state_batch)
            next_q_value = reward_batch + mask_batch * self.gamma * (vf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Regularization Loss (optional)
        reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        vf = self.value(state_batch)

        with torch.no_grad():
            vf_target = min_qf_pi - (self.alpha * log_pi)

        vf_loss = F.mse_loss(vf, vf_target)

        self.value_optim.zero_grad()
        vf_loss.backward()
        self.value_optim.step()

        soft_update(self.value_target, self.value, self.tau)

        return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item()

def main():
    env = gym.make('Pendulum-v1')

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SAC(env.observation_space.shape[0], env.action_space)
    memory = ReplayMemory(replay_size, seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    ep_r_store = []

    for i_episode in range(1000):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset()

        while not done:
            if start_steps > total_numsteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:
                for i in range(updates_per_step):  # Number of updates per step in environment
                    # Update parameters of all the networks
                    value_loss, critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, batch_size, updates)
                    updates += 1

            next_state, reward, terminated, truncated, _ = env.step(action) # Step
            done = (terminated or truncated)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

            if done:
                ep_r_store.append(episode_reward)

        if total_numsteps > num_steps:
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    env.close()

    plt.plot(ep_r_store)
    plt.title('SAC')
    plt.xlabel('episode number')
    plt.ylabel('return')
    plt.grid(True)
    plt.savefig("sac.png")


if __name__ == '__main__':
    main()
