import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# global variables
#===============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

capacity = 50000
batch_size = 128
gamma = 0.99
tau = 0.005
exploration_noise = 0.1
policy_noise = 0.2
noise_clip = 0.5
actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 1000

class Replay_buffer():

    def __init__(self, max_size=capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=None))
            y.append(np.array(Y, copy=None))
            u.append(np.array(U, copy=None))
            r.append(np.array(R, copy=None))
            d.append(np.array(D, copy=None))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

class TD3():
    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer()

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    #===========================================================================
    # TD3 update rule
    #===========================================================================
    def update(self, num_iteration):

        for i in range(num_iteration):

            # sample mini-batch from the replay buffer
            x, y, u, r, d = self.memory.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # select next action according to the target policy
            noise = torch.ones_like(action).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # compute target Q-value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * gamma * target_Q).detach()

            # update critic 1
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # update critic 2
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # delayed policy updates
            if i % 2 == 0:

                # compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()

                # update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # soft replacement
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)


def main():

    agent = TD3(state_dim, action_dim, max_action)
    ep_r = 0
    ep_r_store = []

    for i in range(num_episodes):
        state, _ = env.reset()

        for t in range(200):
            action = agent.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=action_dim)
            action = action.clip(env.action_space.low, env.action_space.high)

            # perform action and obtain transition info
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = (terminated or truncated)
            ep_r += reward

            # add transition to replay buffer
            agent.memory.push((state, next_state, action, reward, float(done)))

            # start updating networks when the replay buffer has enough samples
            if len(agent.memory.storage) >= batch_size:
                agent.update(1)

            state = next_state
            if done or t == 199:
                print("Episode %5d: return is %.2f"%(i, ep_r))
                ep_r_store.append(ep_r)
                ep_r = 0
                break

    plt.plot(ep_r_store)
    plt.title('TD3')
    plt.xlabel('episode number')
    plt.ylabel('return')
    plt.grid(True)
    plt.savefig("td3.png")

if __name__ == '__main__':
    main()
