from collections import namedtuple
from typing import List

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim


log = gym.logger
log.set_level(gym.logger.INFO)

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
LEARNING_RATE = 0.01

Episode = namedtuple('Episode', field_names = ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names = ['observation', 'action'])



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
    rewards = list(map(lambda e: e.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue

        train_obs.extend(map(lambda s: s.observation, steps))
        train_act.extend(map(lambda s: s.action, steps))

    # noinspection PyArgumentList
    return torch.FloatTensor(train_obs), torch.LongTensor(train_act), reward_bound, reward_mean



if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, '01_cartpole-recording', force = True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = net.parameters(), lr = LEARNING_RATE)
    writer = SummaryWriter(comment = '-cartpole')

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # Get the elite policies.
        obs_v, acts_v, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)

        # Optimize the net.
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        # Monitoring.
        log.info('Iter. %d: loss=%.3f, reward_bound=%.3f, reward_mean=%.3f', iter_no, loss_v.item(), reward_bound, reward_mean)
        writer.add_scalar('loss', loss_v.item(), iter_no)
        writer.add_scalar('reward_bound', reward_bound, iter_no)
        writer.add_scalar('reward_mean', reward_mean, iter_no)

        # The maximum number of time steps in Gym is 200, so abort before Gym aborts us (remember: the CartPole environment yields a reward of 1 for each survived time step).
        if reward_mean > 199:
            log.info('Solved!')
            break

    writer.close()
