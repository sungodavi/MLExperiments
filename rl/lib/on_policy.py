import numpy as np
import torch
from torch.distributions.categorical import Categorical


def reward_to_go(batch_rewards, gamma=1):
    """
    :param batch_rewards: (n_eps, ep_len)
    :param gamma: (Int)
    :return: (batch_size) A torch tensor with the rewards
    """
    arrs = []
    for rewards in batch_rewards:
        n = len(rewards)
        to_go_rewards = np.empty(n)
        for i in reversed(range(n)):
            to_go_rewards[i] = gamma ** i * rewards[i] + (to_go_rewards[i + 1] if i + 1 < n else 0)
        arrs.append(to_go_rewards)
    return np.concatenate(arrs)


def run(env, batch_size, policy, render=False):
    first_render = True
    logits = []
    actions = []
    rewards = []
    states = []
    ep_rewards = []

    obs = env.reset()
    while True:
        if render and first_render:
            env.render()

        obs_tensor = torch.from_numpy(obs.reshape(1, -1)).float()
        obs_logits = policy(obs_tensor)
        logits.append(obs_logits)
        [action] = Categorical(logits=obs_logits).sample().numpy()
        states.append(obs)
        actions.append(action)

        obs, reward, done, _ = env.step(action)
        ep_rewards.append(reward)

        if done:
            first_render = False
            rewards.append(ep_rewards)
            ep_rewards = []
            obs = env.reset()

            if len(actions) > batch_size:
                break

    return states, actions, logits, rewards
