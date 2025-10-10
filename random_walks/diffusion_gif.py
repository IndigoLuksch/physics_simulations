#output: GIF

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

#parameters
particle_count = 300
steps_per_frame = int(particle_count * 1)
num_frames = 15000

x_dim = 100
y_dim = 100

#initialise grid
grid = np.zeros((x_dim, y_dim)) #blank

'''
#initialise RANDOM DISTRIBUTION
x = [random.randint(0, x_dim - 1) for _ in range(particle_count)]
y = [random.randint(0, y_dim - 1) for _ in range(particle_count)]
for p in range(particle_count):
    grid[x[p], y[p]] = 1.0
'''

#initialise CLUSTERED PARTICLES
y_seg = list(np.arange(0, y_dim, 1))
x_seg = list(np.zeros(x_dim))
x = []
y = []
for i in range(int(particle_count/y_dim)):
    x +=  x_seg
    x_seg = [i + 1 for i in x_seg]
    y += y_seg
print(f"len x = {len(x)}, len y = {len(y)}")

#random walk
def update(frame):
    for _ in range(steps_per_frame):
        not_overlapping = False
        while not not_overlapping: #if particle moves on top of another particle, retry
            i = random.randint(0, particle_count - 1) #random particle
            dx = random.randint(-1,1)  #new random position
            dy = random.randint(-1,1)
            x_new = int(x[i] + dx)
            y_new = int(y[i] + dy)

            if -1 < x_new < x_dim and y_new > -1 and y_new < y_dim: #within spacial domain
                if grid[x_new, y_new] == 0.0:
                    not_overlapping = True

        #update position in spatial domain
        if -1 < x_new < x_dim:
            x[i] = x_new
        if -1 < y_new < y_dim:
            y[i] = y_new

    grid.fill(0) #clear all
    for q in range(particle_count): #update particle positions
        grid[int(x[q]), int(y[q])] = 1.0

    print(f"frame: {frame}/{num_frames}")

    im.set_data(grid)
    return [im]

#prepare
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap='gray', vmin=0, vmax=1)

#animate and save
ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=20)
ani.save('diffusion_animation.gif', writer='pillow')
