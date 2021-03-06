import gym


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, '04_cartpole_random_monitor-recording', force = True)

    total_reward = 0
    total_steps = 0
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1

    print('Episode done in %d steps. Total reward: %d' % (total_steps, total_reward))

    env.close()
