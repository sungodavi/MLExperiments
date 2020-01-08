import gym
import numpy as np
import copy
import torch
from torch import nn
from rl.lib.nets import MLP
from rl.lib.memory import ReplayBuffer
from rl.lib.torch_utils import to_tensor
from rl.lib.off_policy import train, run


class Actor(nn.Module):
    def __init__(self, state_size, action_range, hidden_layers):
        super().__init__()
        self.action_range = to_tensor(action_range)
        self.net = MLP((state_size[0], *hidden_layers, action_size[0]))

    def forward(self, X):
        out = self.net(X)
        out = torch.tanh(out)
        return out * self.action_range


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super().__init__()
        self.net = MLP((state_size[0] + action_size[0], *hidden_layers, 1))

    def forward(self, obs, action):
        X = torch.cat((obs, action), dim=1)
        return self.net(X).squeeze()


class Agent:
    def __init__(self, env, q_net, policy, q_lr, policy_lr, action_noise, gamma, polyak):
        self.env = env
        self.q_net = q_net
        self.policy = policy
        self.q_target = copy.deepcopy(q_net)
        self.policy_target = copy.deepcopy(policy)

        self.gamma = gamma
        self.polyak = polyak
        self.action_noise = action_noise

        self.q_optimizer = torch.optim.Adam(q_net.parameters(), lr=q_lr)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)

    def select_action(self, obs, train=True):
        obs_tensor = to_tensor(obs)
        action = self.policy(obs_tensor).detach().numpy()
        if train:
            action = np.clip(action + np.random.randn(action.size) * self.action_noise, env.action_space.low, env.action_space.high)

        return action

    def decay_eps(self):
        pass

    def train(self, transitions):
        states, actions, rewards, dones, next_states = transitions

        dones_tensor = to_tensor(dones)
        actions_tensor = to_tensor(actions)
        rewards_tensor = to_tensor(rewards)
        states_tensor = to_tensor(states)
        next_states_tensor = to_tensor(next_states)

        q_vals = self.q_net(states_tensor, actions_tensor)

        next_actions = self.policy_target(next_states_tensor)
        next_q_target_vals = self.q_target(next_states_tensor, next_actions)
        target_q_vals = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_target_vals

        q_loss = nn.MSELoss()(q_vals, target_q_vals)
        self.q_net.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        policy_q_vals = self.q_net(states_tensor, self.policy(states_tensor))
        policy_loss = -torch.mean(policy_q_vals)
        self.policy.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def target_update(self):
        self.soft_update(self.q_net, self.q_target)
        self.soft_update(self.policy, self.policy_target)

    def soft_update(self, curr, target):
        for target_param, curr_param in zip(target.parameters(), curr.parameters()):
            target_param.data.copy_(self.polyak * target_param + (1 - self.polyak) * curr_param)


MEMORY_SIZE = 1000000
Q_LEARNING_RATE = 1e-4
POLICY_LEARNING_RATE = 3e-4
ACTION_NOISE = 0.01
POLYAK = 0.995
GAMMA = 0.99
TRAIN_WAIT = 7500
EPOCHS = 3000
MAX_EP_LEN = 1000
BATCH_SIZE = 128
RENDER_EVERY = 100
UPDATE_EVERY = 1
TRAIN_EVERY = 1

TRAIN = True
PATH = 'ddpg_weights.p'

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    obs_size = env.observation_space.shape
    action_size = env.action_space.shape

    policy = Actor(obs_size, env.action_space.low, (256,))
    q_net = Critic(obs_size, action_size, (256, 256, 128))

    memory = ReplayBuffer(MEMORY_SIZE, state_size=obs_size, action_size=action_size)
    agent = Agent(env, q_net, policy,
                  q_lr=Q_LEARNING_RATE,
                  policy_lr=POLICY_LEARNING_RATE,
                  action_noise=ACTION_NOISE,
                  gamma=GAMMA,
                  polyak=POLYAK)

    if TRAIN:
        train(env, memory, agent,
              n_epochs=EPOCHS,
              max_ep_len=MAX_EP_LEN,
              batch_size=BATCH_SIZE,
              train_wait=TRAIN_WAIT,
              render_every=RENDER_EVERY,
              update_every=UPDATE_EVERY,
              train_every=TRAIN_EVERY)
        torch.save(q_net, PATH)
        input('Press Enter to Continue...')

    run(env, agent, n_epochs=10)
    env.close()
