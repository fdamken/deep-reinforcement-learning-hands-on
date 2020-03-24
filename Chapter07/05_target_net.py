import ptan
from torch import nn



class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()

        self.pipe = nn.Linear(5, 3)


    def forward(self, x):
        return self.pipe(x)



if __name__ == '__main__':
    net = DQNNet()
    print(net)
    target_net = ptan.agent.TargetNet(net)
    print('Main net:', net.pipe.weight)
    print('Target net:', target_net.target_model.pipe.weight)
    net.pipe.weight.data += 1.0
    print('After update')
    print('Main net:', net.pipe.weight)
    print('Target net:', target_net.target_model.pipe.weight)
    target_net.sync()
    print('After sync')
    print('Main net:', net.pipe.weight)
    print('Target net:', target_net.target_model.pipe.weight)
