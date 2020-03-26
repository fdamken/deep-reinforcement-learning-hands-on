import argparse

import gym
import ptan
import torch
from ignite.engine import Engine
from torch import nn, optim

from Chapter08.lib.common import batch_generator, EpsilonTracker, HYPERPARAMS, setup_ignite, unpack_batch
from Chapter08.lib.dqn_model import DQN


NAME = "03_double"
DEFAULT_HYPERPARAMS_SET_NAME = 'pong'



def calc_loss_dqn_double(batch, net, target_net, gamma, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    # noinspection PyArgumentList
    dones_v = torch.BoolTensor(dones).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    qs_v = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_actions = net(next_states_v).max(1)[1]
        next_qs_v = target_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        next_qs_v[dones_v] = 0.0
    bellman_vals = rewards_v + gamma * next_qs_v.detach()
    return nn.MSELoss()(qs_v, bellman_vals)



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
        loss_v = calc_loss_dqn_double(batch, net, target_net.target_model, params.gamma, device = device)
        loss_v.backward()
        optimizer.step()

        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            target_net.sync()

        return { 'loss': loss_v.item(), 'epsilon': selector.epsilon }



    engine = Engine(process_batch)
    setup_ignite(engine, params, exp_source, NAME)
    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))
