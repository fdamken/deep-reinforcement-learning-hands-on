import warnings
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Iterable, List

import numpy as np
import ptan
import ptan.ignite
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from torch import nn


HYPERPARAMS = {
        'pong': SimpleNamespace(**{
                'env_name':        'PongNoFrameskip-v4',
                'stop_reward':     18.0,
                'run_name':        'pong',
                'replay_size':     100000,
                'replay_initial':  10000,
                'target_net_sync': 1000,
                'epsilon_frames':  10 ** 5,
                'epsilon_start':   1.0,
                'epsilon_final':   0.02,
                'learning_rate':   0.0001,
                'gamma':           0.99,
                'batch_size':      32
        })
}



def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for (state, action, reward, last_state) in batch:
        state = np.array(state)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(last_state is None)
        # Append state initial state if the episode was done. This simplifies the implementation of
        # the Bellman equation and does not break it as the "done values" are getting masked and zeroed.
        last_states.append(state if last_state is None else np.array(last_state))
    return np.array(states, copy = False), \
           np.array(actions), \
           np.array(rewards, dtype = np.float32), \
           np.array(dones, dtype = np.uint8), \
           np.array(last_states, copy = False)



@torch.no_grad()
def calc_loss_dqn(batch, net, target_net, gamma, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    # noinspection PyArgumentList
    dones_v = torch.BoolTensor(dones).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    qs_v = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_qs_v = target_net(next_states_v).max(1)[0]
    next_qs_v[dones_v] = 0.0
    bellman_vals = rewards_v + gamma * next_qs_v.detach()
    return nn.MSELoss()(qs_v, bellman_vals)



def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer, initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)



def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source: ptan.experience.ExperienceSource, run_name: str, extra_metrics: Iterable[str] = ()):
    warnings.simplefilter('ignore', category = UserWarning)
    ptan.ignite.EndOfEpisodeHandler(exp_source, bound_avg_reward = params.stop_reward).attach(engine)
    ptan.ignite.EpisodeFPSHandler().attach(engine)


    @engine.on(ptan.ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print('Episode %d: reward=%.3f, steps=%s, speed=%.1f f/s, elapsed=%s' % (trainer.state.episode,
                                                                                 trainer.state.episode_reward,
                                                                                 trainer.state.episode_steps,
                                                                                 trainer.state.metrics.get('avg_fps', 0),
                                                                                 timedelta(seconds = int(passed))))


    @engine.on(ptan.ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print('Game solved in %s, after %d episodes and %d iterations!' % (timedelta(seconds = int(passed)),
                                                                           trainer.state.episode,
                                                                           trainer.state.iteration))
        trainer.should_terminate = True


    now = datetime.now().isoformat(timespec = 'minutes')
    tb = TensorboardLogger(log_dir = f'runs/{now}-{params.run_name}-{run_name}')
    # Episodes metrics.
    handler = OutputHandler(tag = 'episodes', metric_names = ['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler = handler, event_name = ptan.ignite.EpisodeEvents.EPISODE_COMPLETED)
    # Training metrics.
    ptan.ignite.PeriodicEvents().attach(engine)
    RunningAverage(output_transform = lambda v: v['loss']).attach(engine, 'avg_loss')
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = OutputHandler(tag = 'train', metric_names = metrics, output_transform = lambda a: a)
    tb.attach(engine, log_handler = handler, event_name = ptan.ignite.PeriodEvents.ITERS_100_COMPLETED)



class EpsilonTracker:
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector, params: SimpleNamespace):
        self._selector = selector
        self._params = params
        self.frame(0)


    def frame(self, frame_idx: int):
        epsilon = self._params.epsilon_start - frame_idx / self._params.epsilon_frames
        self._selector.epsilon = max(epsilon, self._params.epsilon_final)
