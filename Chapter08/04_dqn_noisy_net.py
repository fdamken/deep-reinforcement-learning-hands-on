import argparse

import gym
import ptan
import torch
from ignite.engine import Engine
from torch import optim

from Chapter08.lib.common import batch_generator, calc_loss_dqn, HYPERPARAMS, setup_ignite
from Chapter08.lib.dqn_extra import NoisyDQN


NAME = "04_noisy"
DEFAULT_HYPERPARAMS_SET_NAME = 'pong'
MONITOR_NOISY_SNR_ITERS = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default = False, action = 'store_true', help = 'Enable CUDA computation.')
    parser.add_argument('--env', default = DEFAULT_HYPERPARAMS_SET_NAME, help = 'Name of the hyperparameter set, defaults to ' + DEFAULT_HYPERPARAMS_SET_NAME + '.')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    params = HYPERPARAMS[args.env]

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device = device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma = params.gamma)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size = params.replay_size)

    optimizer = optim.Adam(net.parameters(), lr = params.learning_rate)



    def process_batch(engine: Engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss_dqn(batch, net, target_net.target_model, params.gamma, device = device)
        loss_v.backward()
        optimizer.step()

        if engine.state.iteration % params.target_net_sync == 0:
            target_net.sync()
        if engine.state.iteration % MONITOR_NOISY_SNR_ITERS == 0:
            for layer_idx, snr in enumerate(net.noisy_layers_snr()):
                engine.state.metrics[f'snr_{layer_idx}'] = snr

        return { 'loss': loss_v.item() }



    engine = Engine(process_batch)
    setup_ignite(engine, params, exp_source, NAME, extra_metrics = ('snr_0', 'snr_1'))
    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))
