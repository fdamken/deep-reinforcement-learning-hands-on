import argparse
import collections
import time

import gym.wrappers
import numpy as np
import torch

from Chapter06.lib import wrappers
from Chapter06.lib.dqn_model import DQN


DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
FPS = 25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required = True, help = 'Model file to load.')
    parser.add_argument('-e', '--env', default = DEFAULT_ENV_NAME, help = 'Name of the environment, defaults to ' + DEFAULT_ENV_NAME + '.')
    parser.add_argument('-r', '--record', help = 'Directory for video.')
    parser.add_argument('--no-vis', default = True, dest = 'vis', help = 'Disable visualization.', action = 'store_false')
    args = parser.parse_args()
    model_file = args.model
    env_name = args.env
    record_dir = args.record
    visualize = args.vis

    env = wrappers.make_env(env_name)
    if record_dir:
        env = gym.wrappers.Monitor(env, record_dir)

    net = DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(model_file, map_location = lambda stg, _: stg))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    while True:
        start_ts = time.time()
        if visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy = False))
        qs_v = net(state_v)
        action = torch.argmax(qs_v, dim = 1).item()
        c[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

        if visualize:
            # Perform a little sleep to allow the human eye to track the progress (gym runs extremely fast...).
            now = time.time()
            delta = 1 / FPS - (now - start_ts)
            if delta > 0:
                time.sleep(delta)

    print('Total reward: %.3f' % total_reward)
    print('Action counts:', list([(env.unwrapped.get_action_meanings()[index], value) for (index, value) in c.items()]))
    if record_dir:
        env.close()
