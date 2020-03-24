import gym.wrappers
import ptan
import torch
from torch import nn, optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
TARGET_NET_SYNC = 10
DISCOUNT_FACTOR = 0.9
REPLAY_SIZE = 1000
LEARNING_RATE = 1e-3
EPSILON_DECAY = 0.99



class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()

        self.pipe = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions)
        )


    def forward(self, x):
        return self.pipe(x.float())



@torch.no_grad()
def unpack_batch(batch, net, discount_factor):
    states = []
    actions = []
    rewards = []
    dones = []
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.tensor(states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    last_states_v = torch.tensor(last_states)
    best_last_q_v = torch.max(net(last_states_v), dim = 1)[0]
    best_last_q_v[dones] = 0.0
    return states_v, actions_v, rewards_v + discount_factor * best_last_q_v



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    net = Net(env.observation_space.shape[0], HIDDEN_SIZE, env.action_space.n)
    target_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon = 1.0, selector = selector)
    agent = ptan.agent.DQNAgent(net, selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma = DISCOUNT_FACTOR)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size = REPLAY_SIZE)

    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)

    step = 0
    episode = 0
    solved = False
    while True:
        step += 1
        buffer.populate(1)

        for reward, steps in exp_source.pop_rewards_steps():
            episode += 1
            print('Step %d: episode %d done, reward %.3f, epsilon %.3f' % (step, episode, reward, selector.epsilon))
            solved = reward > 150

        if solved:
            print('Solved!')
            break
        if len(buffer) < 2 * BATCH_SIZE:
            # Fill buffer.
            continue

        batch = buffer.sample(BATCH_SIZE)
        states_v, actions_v, target_q_v = unpack_batch(batch, target_net.target_model, DISCOUNT_FACTOR)
        optimizer.zero_grad()
        q_v = net(states_v)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = torch.nn.functional.mse_loss(q_v, target_q_v)
        loss_v.backward()
        optimizer.step()

        selector.epsilon *= EPSILON_DECAY

        if step % TARGET_NET_SYNC == 0:
            target_net.sync()
