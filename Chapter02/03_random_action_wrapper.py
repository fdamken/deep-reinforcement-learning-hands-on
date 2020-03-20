import random

import gym



class RandomActionWrapper(gym.ActionWrapper):
    _epsilon: float


    def __init__(self, env, epsilon = 0.1):
        super(RandomActionWrapper, self).__init__(env)

        self._epsilon = epsilon


    def action(self, action):
        if random.random() < self._epsilon:
            print('Choosing random action.')
            return self.env.action_space.sample()
        return action



if __name__ == "__main__":
    env = RandomActionWrapper(gym.make('CartPole-v0'), epsilon = 0.1)

    total_reward = 0
    total_steps = 0
    env.reset()
    done = False
    while not done:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        total_steps += 1

    print('Episode done in %d steps. Total reward: %d' % (total_steps, total_reward))

    env.close()
