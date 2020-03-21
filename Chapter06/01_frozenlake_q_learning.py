import collections

import gym
from tensorboardX import SummaryWriter


log = gym.logger
log.set_level(gym.logger.INFO)

ENV_NAME = 'FrozenLake-v0'
TEST_EPISODES = 20
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.2



class Agent:
    def __init__(self, env_name):
        #  { (state, action) --> action value }
        self.action_values = collections.defaultdict(float)

        self.env = gym.make(env_name)
        self.state = self.env.reset()


    def sample_env(self):
        action = self.env.action_space.sample()
        obs, reward, done, _ = self.env.step(action)

        old_state = self.state
        self.state = self.env.reset() if done else obs

        return old_state, action, reward, obs


    def bellman_update(self):
        old_state, action, reward, new_state = self.sample_env()
        new_q = reward + DISCOUNT_FACTOR * self.select_action(new_state)[0]
        old_q = self.action_values[(old_state, action)]
        self.action_values[(old_state, action)] = (1 - LEARNING_RATE) * old_q + LEARNING_RATE * new_q


    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.action_values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action


    def play_episode(self, env):
        total_reward = 0.0
        obs = env.reset()
        while True:
            _, action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

            obs = next_obs

        return total_reward



if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent(ENV_NAME)
    writer = SummaryWriter(comment = '-frozenlake_q_learning')

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1

        # Executing multiple bellman updates in one iteration indeed leads to less iterations.
        agent.bellman_update()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        writer.add_scalar('reward', reward, iter_no)

        if reward > best_reward:
            log.info('Iter. %d: Best reward updated: %.3f --> %.3f', iter_no, best_reward, reward)
            best_reward = reward

        if reward > 0.8:
            log.info('Solved in %d iterations!', iter_no)
            break

    writer.close()
