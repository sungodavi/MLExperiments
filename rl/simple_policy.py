import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from rl.lib.nets import MLP
from rl.lib.on_policy import reward_to_go, run


def policy_loss(logits, actions, rewards):
    """
    :param logits: (batch_size, act_dim) 2d array output of the policy
    :param actions (batch_size)
    :param rewards (batch_size)
    :return: A scalar value representing the loss
    """
    batch_size = logits.shape[0]
    log_probs = logits.log_softmax(dim=1)[np.arange(batch_size), actions]

    # Negate so you can minimize
    return -torch.mean(log_probs * rewards)


def train(env, epochs=100, policy_lr=1e-2, v_lr=1e-3, train_v_iters=80, batch_size=5000):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = MLP(dims=(obs_dim, 24, 24, act_dim)).float()
    V = MLP(dims=(obs_dim, 32, 32, 1)).float()

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    v_optimizer = torch.optim.Adam(V.parameters(), lr=v_lr)
    mean_rewards = []
    for i in range(epochs):
        states, actions, logits, rewards = run(env, batch_size, policy)
        mean_reward = sum([sum(ep_rewards) for ep_rewards in rewards]) / len(rewards)
        print(f'Episode {i + 1}: Mean Reward = {mean_reward:.2f}')
        mean_rewards.append(mean_reward)

        states_tensor = torch.Tensor(states)
        actions_tensor = torch.tensor(actions)
        logits_tensor = torch.stack(logits).squeeze(dim=1)

        values = V(states_tensor).squeeze(dim=1)
        weighted_rewards = reward_to_go(rewards, gamma=0.99)
        weighted_rewards_tensor = torch.Tensor(weighted_rewards)
        advantages = weighted_rewards_tensor - values
        epoch_loss = policy_loss(logits_tensor, actions_tensor, advantages)

        policy.zero_grad()
        epoch_loss.backward(retain_graph=True)
        policy_optimizer.step()

        for _ in range(train_v_iters):
            values = V(states_tensor).squeeze(dim=1)
            value_loss = torch.nn.MSELoss()(values, weighted_rewards_tensor)
            V.zero_grad()
            value_loss.backward()
            v_optimizer.step()

    return policy, mean_rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    policy, mean_rewards = train(env)

    plt.plot(mean_rewards)
    plt.show()

    for i in range(10):
        _, _, _, rewards = run(env, 1, policy, render=True)
        total_reward = sum(rewards[0])
        print(f'Episode {i}: Reward = {total_reward}')

    env.close()


