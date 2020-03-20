import argparse
import random

import cv2
import gym
import numpy as np
import torch
import torchvision.utils as vutils
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from torch import nn, optim


log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCRIMINATOR_FILTERS = 64
GENERATOR_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000



class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)

        assert isinstance(self.observation_space, gym.spaces.Box)

        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(old_space.low),
                                                self.observation(old_space.high),
                                                dtype = np.float32)


    def observation(self, observation):
        # Resize from Atari screen size to 64x64.
        obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        obs = np.moveaxis(obs, 2, 0)
        return obs.astype(np.float32)



class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.pipe = nn.Sequential(
                nn.Conv2d(in_channels = input_shape[0], out_channels = DISCRIMINATOR_FILTERS, kernel_size = 4, stride = 2, padding = 1),
                nn.ReLU(),
                nn.Conv2d(in_channels = DISCRIMINATOR_FILTERS, out_channels = DISCRIMINATOR_FILTERS * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(DISCRIMINATOR_FILTERS * 2),
                nn.ReLU(),
                nn.Conv2d(in_channels = DISCRIMINATOR_FILTERS * 2, out_channels = DISCRIMINATOR_FILTERS * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(DISCRIMINATOR_FILTERS * 4),
                nn.ReLU(),
                nn.Conv2d(in_channels = DISCRIMINATOR_FILTERS * 4, out_channels = DISCRIMINATOR_FILTERS * 8, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(DISCRIMINATOR_FILTERS * 8),
                nn.ReLU(),
                nn.Conv2d(in_channels = DISCRIMINATOR_FILTERS * 8, out_channels = 1, kernel_size = 4, stride = 1, padding = 0),
                nn.Sigmoid()
        )


    def forward(self, x):
        return self.pipe(x).view(-1, 1).squeeze(dim = 1)



class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()

        self.pipe = nn.Sequential(
                nn.ConvTranspose2d(in_channels = LATENT_VECTOR_SIZE, out_channels = GENERATOR_FILTERS * 8, kernel_size = 4, stride = 1, padding = 0),
                nn.BatchNorm2d(GENERATOR_FILTERS * 8),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels = GENERATOR_FILTERS * 8, out_channels = GENERATOR_FILTERS * 4, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(GENERATOR_FILTERS * 4),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels = GENERATOR_FILTERS * 4, out_channels = GENERATOR_FILTERS * 2, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(GENERATOR_FILTERS * 2),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels = GENERATOR_FILTERS * 2, out_channels = GENERATOR_FILTERS, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(GENERATOR_FILTERS),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels = GENERATOR_FILTERS, out_channels = output_shape[0], kernel_size = 4, stride = 2, padding = 1)
        )


    def forward(self, x):
        return self.pipe(x)



def iterate_batches(envs, batch_size = BATCH_SIZE):
    batch = [e.reset() for e in envs]
    for env in iter(lambda: random.choice(envs), None):
        obs, reward, done, _ = env.step(env.action_space.sample())
        if np.mean(obs) > 0.01:
            # Bugfix for some games. Usually this gets appends directly.
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalize the batch between -1 and 1.
            batch_np = np.array(batch, dtype = np.float32)
            batch_np *= 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if done:
            # Start the environment over to generate more/infinite samples.
            env.reset()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default = False, action = 'store_true', help = 'Enable CUDA computation.')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')]
    input_shape = envs[0].observation_space.shape

    net_dis = Discriminator(input_shape).to(device)
    net_gen = Generator(input_shape).to(device)
    objective = nn.BCELoss()

    dis_optimizer = optim.Adam(params = net_dis.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
    gen_optimizer = optim.Adam(params = net_gen.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))

    true_labels_v = torch.ones(BATCH_SIZE, device = device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device = device)



    def process_batch(trainer, batch):
        # noinspection PyArgumentList
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1).to(device)
        batch_v = batch.to(device)
        gen_output_v = net_gen(gen_input_v)

        # Train discriminator.
        dis_optimizer.zero_grad()
        dis_output_true_v = net_dis(batch_v)
        dis_output_fake_v = net_dis(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()

        # Train generator.
        gen_optimizer.zero_grad()
        dis_output_v = net_dis(gen_output_v)
        gen_loss = objective(dis_output_v, true_labels_v)
        gen_loss.backward()
        gen_optimizer.step()

        if trainer.state.iteration % SAVE_IMAGE_EVERY_ITER == 0:
            trainer.tb.writer.add_image('fake', vutils.make_grid(gen_output_v.data[:64], normalize = True), trainer.state.iteration)
            trainer.tb.writer.add_image('real', vutils.make_grid(batch_v.data[:64], normalize = True), trainer.state.iteration)

        return dis_loss.item(), gen_loss.item()



    engine = Engine(process_batch)
    tb = tb_logger.TensorboardLogger(log_dir = None)
    engine.tb = tb
    RunningAverage(output_transform = lambda out: out[0]).attach(engine, 'avg_loss_dis')
    RunningAverage(output_transform = lambda out: out[1]).attach(engine, 'avg_loss_gen')
    handler = tb_logger.OutputHandler(tag = 'train', metric_names = ['avg_loss_dis', 'avg_loss_gen'])
    tb.attach(engine, log_handler = handler, event_name = Events.ITERATION_COMPLETED)



    @engine.on(Events.ITERATION_COMPLETED)
    def log_losses(trainer):
        if trainer.state.iteration % REPORT_EVERY_ITER == 0:
            log.info('Iter %d: gen_loss=%.3f, dis_loss=%.3f',
                     trainer.state.iteration,
                     trainer.state.metrics['avg_loss_gen'],
                     trainer.state.metrics['avg_loss_dis'])



    engine.run(data = iterate_batches(envs))
