# https://distill.pub/2020/growing-ca/

import torch
from torch import nn
from torch.nn import functional as f
import torch.optim as opt
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


class UpdateGrid(torch.nn.Module):  # Class for updating the grid

    def __init__(self):
        super(UpdateGrid, self).__init__()
        # Initialize first layer
        self.dense1 = nn.Conv2d(N_CHANNEL * len(filters), HIDDEN_LAYER_NODES, (1, 1))

        # Initialize second layer
        self.dense2 = nn.Conv2d(HIDDEN_LAYER_NODES, N_CHANNEL, (1, 1))
        # Weights of th second layer equal to 0
        torch.nn.init.zeros_(self.dense2.weight)

    def forward(self, x):
        # We define the perception matrix
        perception = torch.empty((BATCH_SIZE, len(filters) * N_CHANNEL, width, height))
        # We apply the filters
        for f_i, fil in enumerate(filters):
            perception[:, (f_i * N_CHANNEL):((f_i + 1) * N_CHANNEL), :, :] = f.conv2d(x, fil, padding=[1, 1], groups=N_CHANNEL)
        # Input to first hidden layer
        dx = self.dense1(perception)
        # Activation function
        dx = f.relu(dx)
        # Second hidden layer
        dx = self.dense2(dx)
        # Output with a stochastic update + input (random masking)
        random_mask = torch.from_numpy(np.random.randint(0, 2, (BATCH_SIZE, 1, width, height))).repeat(1, 16, 1, 1)
        x = x + dx * random_mask
        # We check that no cells are mature or growing and not surrounded by mature cells
        alive = f.conv2d((x[:, 3:4, :, :] > 0.1).type(torch.int), torch.from_numpy(np.ones((1, 1, 3, 3))).int(), padding=1)
        alive = (alive > 0).type(torch.int)
        alive = alive.repeat(1, 16, 1, 1)
        return x * alive


# Define variables
POOL_SIZE = 1024
N_CHANNEL = 16
HIDDEN_LAYER_NODES = 128
width = height = int(math.sqrt(POOL_SIZE))
BATCH_SIZE = 4
SAVE_IMG = 5
n_img = 0

# Define filters, Ojito con el shape para el conv2D
sobelX = torch.from_numpy(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float)).repeat((N_CHANNEL, 1, 1, 1))
sobelY = torch.from_numpy(np.array([[1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(float)).repeat((N_CHANNEL, 1, 1, 1))
cellId = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(float)).repeat((N_CHANNEL, 1, 1, 1))
filters = [sobelX, sobelY, cellId]

# Define ion functionality from math plot lib
plt.ion()

im = Image.open('diamond.png')
target = torch.tensor(np.array(im) / 255.)
updateGrid = UpdateGrid()

# Initialize loss function
loss_function = nn.MSELoss()

# Create stochastic gradient descent
optimizer = opt.Adam(updateGrid.parameters(), lr=1e-4)
scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

# We normalize the gradient
for p in updateGrid.parameters():
    p.register_hook(lambda grad: grad / (torch.norm(grad, 2) + 1e-8))

# Training loop
for fit in range(10001):
    STEPS = np.random.randint(64, 96)
    # Define grid
    grid = np.zeros((width, height, N_CHANNEL))
    grid[height // 2, width // 2, 3:] = 1.0
    batch_grid = np.repeat(grid[np.newaxis, ...], BATCH_SIZE, axis=0)
    result = torch.from_numpy(batch_grid).permute(0, 3, 1, 2)
    # Generations
    for step in range(STEPS):
        result = torch.clamp(updateGrid.forward(result), 0.0, 1.0)

        if ((fit + 1) % SAVE_IMG == 0) and (step % 2 == 0):
            batch_images = result[0, :4, :, :].detach().numpy()
            fig, ax = plt.subplots(1, 1)

            imRes = np.clip(batch_images.transpose(1, 2, 0)[:, :, :4], 0.0, 1.0)
            plt.imshow(imRes)
            # fig, ax = plt.subplots()
            # plt.set_facecolor('white')
            plt.title('Generation: ' + str(step) + 'Training: ' + str(fit))
            plt.draw()
            plt.pause(0.01)
            # n_img += 1
            # plt.imsave('./evolution/gen_' + str(n_img) + '.png', imRes)
            # plt.show()
            plt.clf()

    # We clean the gradient buffer
    optimizer.zero_grad()

    # Output and loss function
    output = result[:, :4, :, :].permute(0, 2, 3, 1)
    loss = loss_function(output, target.repeat((BATCH_SIZE, 1, 1, 1)))

    # Back propagation in one step
    loss.backward()
    optimizer.step()

    print('Training loss: ', loss)
