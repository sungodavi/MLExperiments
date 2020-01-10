import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from rl.lib.nets import MLP
from rl.lib.memory import ReplayBuffer
from rl.lib.torch_utils import to_tensor
from rl.simple_policy import train as train_policy, run as run_policy


class Model(nn.Module):
    def __init__(self, state_size, hidden_dims, init_states):
        super().__init__()
        self.state_size = state_size
        self.init_states = init_states
        self.curr_state = None

        input_dim = state_size + 1
        self.l1 = MLP((input_dim, *hidden_dims), output_activation=F.relu)

        self.n_s = nn.Linear(hidden_dims[-1], state_size)
        self.r = nn.Linear(hidden_dims[-1], 1)
        self.d = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        net_input = torch.cat([state, action.unsqueeze(dim=1)], dim=1)
        out = self.l1(net_input)

        next_state = self.n_s(out)
        r = self.r(out)
        d_logits = self.d(out)
        d_sig = torch.sigmoid(d_logits)

        return next_state, r, d_sig, d_logits

    def step(self, action, threshold=0.5):
        curr_state_tensor = to_tensor(self.curr_state.reshape(1, -1))
        action_tensor = torch.Tensor([action]).float()
        next_state, r, d_sig, _ = self.forward(curr_state_tensor, action_tensor)
        return (
            next_state[0].detach().numpy(),
            r[0, 0].detach().item(),
            d_sig[0, 0].item() > threshold,
            {}
        )

    def reset(self):
        self.curr_state = self.init_states.sample()
        return self.curr_state


class Policy(MLP):
    def __init__(self, hidden_dims):
        super().__init__(hidden_dims)

    def get_action(self, obs):
        logits = self.forward(to_tensor(obs))
        return Categorical(logits=logits).sample().numpy()


class StateBuffer:
    def __init__(self, capacity, state_size):
        self.states = np.empty((capacity, state_size))
        self.index = 0
        self.capacity = capacity
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.index

    def store(self, state):
        self.states[self.index] = state

        transition_index = self.index
        self.index = (self.index + 1) % self.capacity
        if not self.full:
            self.full = self.index == 0

        return transition_index

    def sample(self):
        size = len(self)
        if size == 0:
            raise Exception('Buffer is empty')

        index = np.random.randint(0, size)
        return self.states[index]


def train_model(env, model, policy, memory, eps, train_eps, batch_size, lr):
    obs = env.reset()
    model.init_states.store(obs)
    for i in range(eps):
        action = policy.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        memory.store((obs, action, reward, done, next_obs))
        if done:
            obs = env.reset()
            model.init_states.store(obs)
        else:
            obs = next_obs

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_losses = []
    for i in range(train_eps):
        indices = np.arange(len(memory))
        np.random.shuffle(indices)
        for batch_start in range(0, len(memory), batch_size):
            batch = memory.get_data(indices[batch_start:batch_start + batch_size])
            states, actions, rewards, dones, next_states = batch
            states_tensor = to_tensor(states)
            actions_tensor = to_tensor(actions)
            rewards_tensor = to_tensor(rewards)
            dones_tensor = to_tensor(dones)
            next_states_tensor = to_tensor(next_states)
            pred_ns, pred_r, _, pred_d_logits = model(states_tensor, actions_tensor)

            ns_loss = nn.MSELoss()(pred_ns, next_states_tensor)
            r_loss = nn.MSELoss()(pred_r.squeeze(dim=1), rewards_tensor)
            d_loss = nn.BCEWithLogitsLoss()(pred_d_logits.squeeze(dim=1), dones_tensor)
            total_loss = ns_loss + r_loss + 100 * d_loss

            model.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_losses.append(total_loss)

    return total_losses


N_EPS = 10
MODEL_EPS = 12000
TRAIN_EPS = 100
BATCH_SIZE = 512
LR = 4e-4
EPOCHS = 50
POLICY_LR = 1e-3
V_LR=1e-3
TRAIN_V_ITERS = 80
POLICY_BATCH_SIZE = 1000


def train(env, model, policy, v_fn, memory):
    for i in range(N_EPS):
        print(f'Episode {i}. Model is training...')
        train_model(env, model, policy, memory,
                    eps=MODEL_EPS,
                    train_eps=TRAIN_EPS,
                    batch_size=BATCH_SIZE,
                    lr=LR)
        print('Model Training Complete. Begin Policy Training')
        train_policy(model, policy, v_fn,
                     epochs=EPOCHS,
                     policy_lr=POLICY_LR,
                     v_lr=V_LR,
                     train_v_iters=TRAIN_V_ITERS,
                     batch_size=POLICY_BATCH_SIZE)
        print('Testing')
        _, _, _, rewards = run_policy(env, 500, policy)
        mean_reward = sum([sum(ep_rewards) for ep_rewards in rewards]) / len(rewards)
        print(f'Test Episode {i + 1}: Mean Reward = {mean_reward:.2f}')


def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    init_states = StateBuffer(1000, state_size[0])
    memory = ReplayBuffer(10000, state_size)
    model = Model(state_size[0], (32, 32), init_states).float()
    policy = Policy((state_size[0], 24, 24, action_size))
    v_fn = MLP((state_size[0], 32, 32, 1))

    train(env, model, policy, v_fn, memory)


if __name__ == '__main__':
    main()
