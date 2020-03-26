import argparse
import collections
import time

import gym.wrappers
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim

from Chapter06.lib import wrappers
from Chapter06.lib.dqn_model import DQN


log = gym.logger
log.set_level(gym.logger.INFO)

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
MEAN_REWARD_BOUND = 19.0

DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPISLON_FINAL = 0.01

Experience = collections.namedtuple('Experience', field_names = ('state', 'action', 'reward', 'done', 'next_state'))



class ExperienceBuffer():
    def __init__(self, capacity: int):
        self._buffer = collections.deque(maxlen = capacity)


    def __len__(self):
        return len(self._buffer)


    def append(self, experience):
        self._buffer.append(experience)


    def sample(self, batch_size):
        indices = np.random.choice(len(self._buffer), batch_size, replace = False)
        states, actions, rewards, dones, next_states = zip(*[self._buffer[idx] for idx in indices])
        return np.array(states), \
               np.array(actions), \
               np.array(rewards, dtype = np.float32), \
               np.array(dones, dtype = np.uint8), \
               np.array(next_states)



class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self.total_reward = None
        self._reset()


    @torch.no_grad()
    def play_step(self, net, epsilon, device: torch):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy = False)
            state_v = torch.tensor(state_a).to(device)
            # The network computes Q(s, a) for all actions a once.
            qs_v = net(state_v)
            action = torch.argmax(qs_v, dim = 1).item()

        next_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.exp_buffer.append(Experience(self.state, action, reward, done, next_state))
        self.state = next_state

        done_reward = None
        if done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0



def calc_loss(batch, net, target_net, device):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(states, copy = False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    # noinspection PyArgumentList
    dones_v = torch.BoolTensor(dones).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy = False)).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = target_net(next_states_v).max(1)[0]
        next_state_values[dones_v] = 0.0
        next_state_values = next_state_values.detach()

    return nn.MSELoss()(state_action_values, rewards_v + DISCOUNT_FACTOR * next_state_values)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default = False, action = 'store_true', help = 'Enable CUDA computation.')
    parser.add_argument('--env', default = DEFAULT_ENV_NAME, help = 'Name of the environment, defaults to ' + DEFAULT_ENV_NAME + '.')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    env_name = args.env

    env = wrappers.make_env(env_name)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment = '-dqn-' + env_name)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME, EPISLON_FINAL)

        reward = agent.play_step(net, epsilon, device)
        if reward is not None:
            # Collect metrics only iff the game has finished. Learning is performed anyway!

            total_rewards.append(reward)

            now = time.time()
            speed = (frame_idx - ts_frame) / (now - ts)
            ts_frame = frame_idx
            ts = now

            # Average the mean reward over the last 100 rewards.
            mean_reward = np.mean(total_rewards[-100:])

            log.info('Frame %d: done %d games, reward mean %.3f, reward %.3f, epsilon %.3f, speed %.2f f/s', frame_idx, len(total_rewards), mean_reward, reward, epsilon, speed)
            writer.add_scalar('epsilon', epsilon, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('reward_100', mean_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)
            writer.flush()

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), env_name + '-best_%010.3f.dat' % mean_reward)
                if best_mean_reward is not None:
                    log.info('Best reward updated: %.3f --> %.3f', best_mean_reward, mean_reward)
                best_mean_reward = mean_reward

            if mean_reward > MEAN_REWARD_BOUND:
                log.info('Solved in %d frames with a reward of %.3f!', frame_idx, mean_reward)
                break

        if len(buffer) < REPLAY_START_SIZE:
            # Do not start learning until the replay buffer is filled enough.
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())

        # Learn!
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_v = calc_loss(batch, net, target_net, device)
        loss_v.backward()
        optimizer.step()

    writer.close()
