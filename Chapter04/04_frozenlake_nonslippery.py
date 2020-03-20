from collections import namedtuple
from typing import List

import gym
import gym.envs.toy_text.frozen_lake
import gym.wrappers
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim


log = gym.logger
log.set_level(gym.logger.INFO)

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.001

Episode = namedtuple('Episode', field_names = ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names = ['observation', 'action'])



class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(DiscreteOneHotWrapper, self).__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Discrete)

        shape = (env.observation_space.n,)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape, dtype = np.float32)


    def observation(self, observation):
        obs = np.copy(self.observation_space.low)
        obs[observation] = 1.0
        return obs



class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
        )


    def forward(self, x):
        return self.net(x)



def iterate_batches(env: gym.Env, net: Net, batch_size: int):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim = 1)

    while True:
        # noinspection PyArgumentList
        obs_v = torch.FloatTensor([obs])

        # Get the probability distribution, sample and execute an action.
        act_probs = sm(net(obs_v)).data.numpy()[0]
        action = np.random.choice(len(act_probs), p = act_probs)
        next_obs, reward, done, _ = env.step(action)

        # Collect metrics for applying the cross-entropy method.
        episode_reward += reward
        step = EpisodeStep(observation = obs, action = action)
        episode_steps.append(step)

        if done:
            episode = Episode(reward = episode_reward, steps = episode_steps)
            batch.append(episode)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs



def filter_batch(batch: List[Episode], percentile: float):
    # As the reward is only given at the end of an episode, this is equivalent of multiplying each sub-reward with the discount factor.
    disc_rewards = list(map(lambda s: s.reward * (DISCOUNT_FACTOR ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)
    reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))

    train_obs = []
    train_act = []
    elite_batch = []
    for example, disc_reward in zip(batch, disc_rewards):
        if disc_reward > reward_bound:
            train_obs.extend(map(lambda s: s.observation, example.steps))
            train_act.extend(map(lambda s: s.action, example.steps))
            elite_batch.append(example)

    # noinspection PyArgumentList
    return elite_batch, torch.FloatTensor(train_obs), torch.LongTensor(train_act), reward_bound, reward_mean



if __name__ == "__main__":
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery = False)
    env.spec = gym.spec('FrozenLake-v0')
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 100)
    env = DiscreteOneHotWrapper(env)
    # env = gym.wrappers.Monitor(env, '02_frozenlake_nonslippery-recording', force = True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = net.parameters(), lr = LEARNING_RATE)
    writer = SummaryWriter(comment = '-frozenlake_tweaked')

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # Get the elite policies.
        full_batch, obs_v, acts_v, reward_bound, reward_mean = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        # Limit to 500 policies.
        full_batch = full_batch[-500:]

        # Optimize the net.
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        # Monitoring.
        log.info('Iter. %d: loss=%.3f, reward_bound=%.3f, reward_mean=%.3f, batch=%d', iter_no, loss_v.item(), reward_bound, reward_mean, len(full_batch))
        writer.add_scalar('loss', loss_v.item(), iter_no)
        writer.add_scalar('reward_bound', reward_bound, iter_no)
        writer.add_scalar('reward_mean', reward_mean, iter_no)

        if reward_mean > 0.8:
            log.info('Solved!')
            break

    writer.close()
