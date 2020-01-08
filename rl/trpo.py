import copy
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from rl.lib.nets import MLP
from rl.lib.on_policy import reward_to_go, run


def flat_grad(y, x, create_graph=False):
    grads = torch.autograd.grad(y, x, create_graph=create_graph)
    return torch.cat([g.view(-1) for g in grads])


def kl_divergence(model, old_model, states):
    """
    :param model: Torch Module -> (batch_size, act_dim)
    :param states: (batch_size, obs_dim)
    :return: a scalar
    """
    new_log_probs = model(states).log_softmax(dim=1)
    old_log_probs = old_model(states).log_softmax(dim=1).detach()
    kl = torch.exp(old_log_probs) * (old_log_probs - new_log_probs)
    # sum across all actions, mean over batch
    return kl.sum(dim=1).mean()


def kl_hessian_vector_product(kl, model, g):
    damping = 1e-2  # TODO: What is this

    grads = flat_grad(kl, model.parameters(), create_graph=True)
    grads_v = torch.sum(grads * g)  # Should be the same shape since the gradient has same # of input params
    grads_grads_v = torch.autograd.grad(grads_v, model.parameters(), retain_graph=True)
    flat_grad_grad_v = torch.cat([grad.contiguous().view(-1) for grad in grads_grads_v])
    return flat_grad_grad_v + g * damping


def policy_loss(logits, actions, advantages):
    """
    TODO: (PPO) Implement loss function with clipping
    :return:
    """
    batch_size = logits.shape[0]
    log_probs = logits.log_softmax(dim=1)[np.arange(batch_size), actions]
    old_log_probs = log_probs.detach()  # Loss is evaluated at theta_k = theta

    loss = torch.exp(log_probs - old_log_probs) * advantages
    return loss.mean()


def conjugate_gradient(model, kl, b, iters=10, residual_limit=1e-10):
    """
    Returns F^(-1)b where F is the Hessian of the KL divergence
    """
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = r.double().dot(r.double())
    for _ in range(iters):
        z = kl_hessian_vector_product(kl, model, p).squeeze(0)
        v = rdotr / p.double().dot(z.double())
        x += v * p
        r -= v * z
        newrdotr = r.double().dot(r.double())
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr

        if rdotr < residual_limit:
            break
    return x


def line_search(model, theta_step, delta, curr_loss, states, actions, advantages, n_steps=10, backtrack_coeff=0.8):
    theta = parameters_to_vector(model.parameters())
    new_model = copy.deepcopy(model)
    for j in range(n_steps):
        alpha = backtrack_coeff ** j
        new_theta = theta + alpha * theta_step
        vector_to_parameters(new_theta, new_model.parameters())

        loss, kl = policy_loss(new_model(states), actions, advantages), kl_divergence(new_model, model, states)
        if loss <= curr_loss and kl <= delta:
            return new_model

    print('Warning, using old model')
    return model


def train(env, epochs=50, kl_delta=1e-2, v_lr=1e-3, train_v_iters=80, batch_size=5000):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = MLP(dims=(obs_dim, 24, 24, act_dim)).float()
    V = MLP(dims=(obs_dim, 32, 32, 1)).float()

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
        advantages = weighted_rewards_tensor - values  # TODO: Do I have to normalize these? Also, what is GAE
        epoch_loss = policy_loss(logits_tensor, actions_tensor, advantages)
        kl = kl_divergence(policy, policy, states_tensor)

        policy.zero_grad()
        grads = flat_grad(epoch_loss, policy.parameters(), create_graph=True)
        hessian_grad_prod = conjugate_gradient(policy, kl, grads)
        denom = torch.matmul(hessian_grad_prod.T, kl_hessian_vector_product(kl, policy, hessian_grad_prod))
        theta_step = torch.sqrt(2 * kl_delta / denom) * hessian_grad_prod
        policy = line_search(policy, theta_step, kl_delta, epoch_loss, states_tensor, actions_tensor, advantages)

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
