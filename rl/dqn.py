import gym
import copy
import numpy as np
import torch
from torch import nn
import random
from rl.lib.memory import PriorityBuffer
from rl.lib.nets import MLP
from rl.lib.torch_utils import to_tensor
from rl.lib.off_policy import train, run

PATH = 'dqn_weights.p'


class DQN(nn.Module):
    """
    Dueling DQN network. Has a separate Value and Advantage function, and uses those together
    to get Q values
    """
    def __init__(self, in_dim, out_dim, q_arch, v_arch, a_arch):
        super().__init__()
        self.l1 = MLP((*in_dim, *q_arch))
        self.v = MLP((q_arch[-1], *v_arch, 1))
        self.a = MLP((q_arch[-1], *a_arch, out_dim))

    def forward(self, obs):
        out = self.l1(obs)
        v = self.v(out)
        a = self.a(out)

        # Subtract mean to have the advantage be about 0 for the highest action
        q = v + (a - a.mean())
        return q


class Agent:
    def __init__(self, q_net, lr, n_actions, eps_decay, gamma):
        self.q_net = q_net
        self.q_target = copy.deepcopy(q_net)
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps = 1
        self.n_actions = n_actions
        self.optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    def select_action(self, obs, train=True):
        if train and random.random() < self.eps:
            return random.randrange(self.n_actions)
        obs_tensor = to_tensor(obs)
        return torch.argmax(self.q_net(obs_tensor)).numpy()

    def decay_eps(self):
        self.eps *= self.eps_decay

    def train(self, transitions, is_weights=None):
        states, actions, rewards, dones, next_states = transitions

        dones_tensor = to_tensor(dones)
        rewards_tensor = to_tensor(rewards)
        states_tensor = to_tensor(states)
        next_states_tensor = to_tensor(next_states)
        batch_indices = np.arange(states.shape[0])

        q_vals = self.q_net(states_tensor)
        selected_q_vals = q_vals[batch_indices, actions]

        next_q_vals = self.q_net(next_states_tensor)
        next_q_target_vals = self.q_target(next_states_tensor)
        selected_actions = next_q_vals.argmax(dim=1)
        target_q_vals = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_target_vals[batch_indices, selected_actions]

        td_error = (selected_q_vals - target_q_vals)
        if is_weights is not None:
            is_weights_tensor = to_tensor(is_weights)
            loss = torch.mean(is_weights_tensor * td_error ** 2)
        else:
            loss = torch.mean(td_error ** 2)

        self.q_net.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error

    def target_update(self):
        self.q_target.load_state_dict(self.q_net.state_dict())


MEMORY_SIZE = 250000
LEARNING_RATE = 1e-4
EPS_DECAY = 0.99941
GAMMA = 0.99
TRAIN_WAIT = 7500
EPOCHS = 5000
MAX_EP_LEN = 1000  # Max cap for lunar lander is 1000
BATCH_SIZE = 64
RENDER_EVERY = 100
UPDATE_EVERY = 200
TRAIN_EVERY = 1

TRAIN = True


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    obs_size = env.observation_space.shape
    n_actions = env.action_space.n

    # q_net = DQN(obs_size, n_actions, (64,), (32,), (32,)).float()
    q_net = MLP((obs_size[0], 32, 32, n_actions))
    if not TRAIN:
        q_net.load_state_dict(torch.load(PATH), strict=False)

    memory = PriorityBuffer(MEMORY_SIZE, obs_size)
    agent = Agent(q_net, LEARNING_RATE, n_actions, EPS_DECAY, GAMMA)

    if TRAIN:
        train(env, memory, agent,
              n_epochs=EPOCHS,
              max_ep_len=MAX_EP_LEN,
              batch_size=BATCH_SIZE,
              train_wait=TRAIN_WAIT,
              render_every=RENDER_EVERY,
              update_every=UPDATE_EVERY,
              train_every=TRAIN_EVERY,
              mem_type='priority')
        torch.save(q_net, PATH)
        input('Press Enter to Continue...')

    run(env, agent, n_epochs=10)
    env.close()
