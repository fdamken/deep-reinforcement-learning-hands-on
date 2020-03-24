import argparse

import gym
import ptan
import torch
from ignite.engine import Engine
from torch import optim

from Chapter08.lib.common import batch_generator, calc_loss_dqn, EpsilonTracker, HYPERPARAMS, setup_ignite
from Chapter08.lib.dqn_model import DQN


NAME = "01_baseline"
DEFAULT_HYPERPARAMS_SET_NAME = 'pong'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default = False, action = 'store_true', help = 'Enable CUDA computation.')
    parser.add_argument('--env', default = DEFAULT_HYPERPARAMS_SET_NAME, help = 'Name of the hyperparameter set, defaults to ' + DEFAULT_HYPERPARAMS_SET_NAME + '.')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    params = HYPERPARAMS[args.env]

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon = params.epsilon_start)
    epsilon_tracker = EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device = device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma = params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size = params.replay_size)

    optimizer = optim.Adam(net.parameters(), lr = params.learning_rate)



    def process_batch(engine: Engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss_dqn(batch, net, target_net.target_model, params.gamma, device = device)
        loss_v.backward()
        optimizer.step()

        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            target_net.sync()

        return { 'loss': loss_v.item(), 'epsilon': selector.epsilon }



    engine = Engine(process_batch)
    setup_ignite(engine, params, exp_source, NAME)
    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))
