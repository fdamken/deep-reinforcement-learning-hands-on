import ptan
import torch
from torch import nn



class DQNNet(nn.Module):
    def __init__(self, n_actions: int):
        super(DQNNet, self).__init__()

        self.n_actions = n_actions


    def forward(self, x):
        return torch.eye(x.size()[0], self.n_actions)



class PolicyNet(nn.Module):
    def __init__(self, n_actions: int):
        super(PolicyNet, self).__init__()

        self.n_actions = n_actions


    def forward(self, x):
        # Produce logits where the actions 0, 1 have the same values and the other actions are all
        # zero (as softmax is applied, this does not correspond to a zero probability!).
        shape = (x.size()[0], self.n_actions)
        logits = torch.zeros(shape, dtype = torch.float32)
        logits[:, 0] = 1
        logits[:, 1] = 1
        return logits



if __name__ == "__main__":
    net = DQNNet(3)
    print('net out:\n', net(torch.zeros(2, 10)))

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(dqn_model = net, action_selector = selector)
    print('argmax-agent (actions, states):', agent(torch.zeros(2, 5)))

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon = 1.0)
    agent = ptan.agent.DQNAgent(dqn_model = net, action_selector = selector)
    print('epsilon-%.2f-greedy-agent actions:' % selector.epsilon, agent(torch.zeros(10, 5))[0])
    selector.epsilon = 0.5
    print('epsilon-%.2f-greedy-agent actions:' % selector.epsilon, agent(torch.zeros(10, 5))[0])
    selector.epsilon = 0.1
    print('epsilon-%.2f-greedy-agent actions:' % selector.epsilon, agent(torch.zeros(10, 5))[0])

    net = PolicyNet(5)
    print('net out:\n', net(torch.zeros(6, 10)))

    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(model = net, action_selector = selector, apply_softmax = True)
    print('policy-agent actions:', agent(torch.zeros(6, 5))[0])
