def train(env, memory, agent, n_epochs, max_ep_len, batch_size, train_wait, render_every, update_every, train_every, mem_type='replay'):
    n_steps = 0
    total_rewards = []
    for e in range(n_epochs):
        obs = env.reset()
        total_reward = 0
        t = 0
        for t in range(max_ep_len):
            if e % render_every == 0:
                env.render()

            if n_steps < train_wait:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            memory.store((obs, action, reward, done, next_obs))
            total_reward += reward

            if n_steps >= train_wait and n_steps % train_every == 0:
                if memory == 'priority':
                    _train_priority(agent, memory, batch_size)
                else:
                    _train_replay(agent, memory, batch_size)

            if n_steps % update_every == 0:
                agent.target_update()

            n_steps += 1
            if done:
                break
            obs = next_obs

        total_rewards.append(total_reward)
        avg_reward = sum(total_rewards[-100:]) / len(total_rewards[-100:])
        if hasattr(agent, 'eps'):
            print(f'Episode {e + 1:3d}: Eps = {agent.eps:.2f}. Episode Length = {t:3d}. '
                  f'Reward = {total_reward:7.2f}. Avg Reward = {avg_reward:7.2f}')
        else:
            print(f'Episode {e + 1:3d}: Episode Length = {t:3d}. '
                  f'Reward = {total_reward:7.2f}. Avg Reward = {avg_reward:7.2f}')
        if n_steps > train_wait:
            agent.decay_eps()


def _train_replay(agent, memory, batch_size):
    agent.train(memory.sample(batch_size))


def _train_priority(agent, memory, batch_size):
    transitions, is_weights, update_buffer_priorities = memory.sample(batch_size)
    td_err = agent.train(transitions, is_weights)
    update_buffer_priorities(td_err.detach().numpy())


def run(env, agent, n_epochs):
    for e in range(n_epochs):
        done = False
        n_steps = 0
        total_reward = 0
        obs = env.reset()
        while not done:
            env.render()
            action = agent.select_action(obs, train=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            n_steps += 1

        print(f'Episode {e + 1}: Episode Length = {n_steps}. Reward = {total_reward:.2f}')

