import numpy as np
import torch
from torch import nn



class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init = 0.017, bias = True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)

        # Sigma is the factor of how much effect the noise has. These sigmas will be updated using backprop.

        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer('epsilon_weight', z)

        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer('epsilon_bias', z)

        self.reset_parameters()


    def reset_parameters(self):
        std = np.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)


    def forward(self, x):
        # Sample random noise for weights and biases and add it.
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias += self.sigma_bias * self.epsilon_bias.data
        v = self.weight + self.sigma_weight * self.epsilon_weight.data
        return nn.functional.linear(x, v, bias)



class NoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero = 0.4, bias = True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias)

        sigma_init = sigma_zero / np.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z_in = torch.zeros(1, in_features)
        self.register_buffer('epsilon_input', z_in)
        z_out = torch.zeros(out_features, 1)
        self.register_buffer('epsilon_output', z_out)

        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)


    def forward(self, x):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)
        bias = self.bias
        if bias is not None:
            bias += self.sigma_bias * eps_out.t()
        v = self.weight + self.sigma_weight * torch.mul(eps_in, eps_out)
        return nn.functional.linear(x, v, bias)



class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size = 8, stride = 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
                nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = (
                NoisyLinear(conv_out_size, 512),
                NoisyLinear(512, n_actions)
        )
        self.fc = nn.Sequential(
                self.noisy_layers[0],
                nn.ReLU(),
                self.noisy_layers[1]
        )


    def forward(self, x):
        x_float = x.float() / 256
        conv_out = self.conv(x_float).view(x_float.size()[0], -1)
        return self.fc(conv_out)


    def noisy_layers_snr(self):
        return [(layer.weight ** 2).mean().sqrt().item() / (layer.sigma_weight ** 2).mean().sqrt().item() for layer in self.noisy_layers]


    def _get_conv_out(self, shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))
