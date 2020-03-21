import collections

import gym
from tensorboardX import SummaryWriter


log = gym.logger
log.set_level(gym.logger.INFO)

ENV_NAME = 'FrozenLake-v0'
RANDOM_STEPS = 100
TEST_EPISODES = 20
DISCOUNT_FACTOR = 0.9



class Agent:
    def __init__(self, env_name):
        #  { (source state, action) --> reward }
        self.rewards = collections.defaultdict(float)
        #  { (source state, action) --> { target state --> count } }
        self.transitions = collections.defaultdict(collections.Counter)
        #  { (state, action) --> value }
        self.action_values = collections.defaultdict(float)

        self.env = gym.make(env_name)
        self.state = self.env.reset()


    def play_n_random_steps(self, count):
        # Waiting to the end of an episode is not necessary!
        for _ in range(count):
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)

            self.rewards[(self.state, action, next_obs)] = reward
            self.transitions[(self.state, action)][next_obs] += 1

            self.state = self.env.reset() if done else next_obs


    def q_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                dest_counts = self.transitions[(state, action)]
                total = sum(dest_counts.values())
                q = 0.0
                for dest_state, count in dest_counts.items():
                    q += (count / total) * (self.rewards[(state, action, dest_state)] + DISCOUNT_FACTOR * self.action_values[(dest_state, self.select_action(dest_state))])
                self.action_values[(state, action)] = q


    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.action_values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action


    def play_episode(self, env):
        total_reward = 0.0
        obs = env.reset()
        while True:
            action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            self.rewards[(obs, action, next_obs)] = reward
            self.transitions[(obs, action)][next_obs] += 1

            if done:
                break

            obs = next_obs

        return total_reward



if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent(ENV_NAME)
    writer = SummaryWriter(comment = '-frozenlake_v_iteration')

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1

        agent.play_n_random_steps(RANDOM_STEPS)
        agent.q_iteration()

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
