#output: mean x_pos graph

import numpy as np
import matplotlib.pyplot as plt
import random

#parameters
particle_count = 300
steps_per_frame = int(particle_count * 1)
num_frames = 25000
repeats = 15

x_dim = 100
y_dim = 100

#initialise grid
grid = np.zeros((x_dim, y_dim)) #blank

#define random walk function
def update(frame):
    for _ in range(steps_per_frame):
        not_overlapping = False
        while not not_overlapping: #if particle moves on top of another particle, retry
            i = random.randint(0, particle_count - 1) #random particle
            dx = random.randint(-1,1)  #new random position
            dy = random.randint(-1,1)
            x_new = int(x[i] + dx)
            y_new = int(y[i] + dy)

            if -1 < x_new < x_dim and -1 < y_new < y_dim: #within spacial domain
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

    #track average y pos
    x_av = np.mean(x)
    x_avs.append(x_av)


    print(f"repeat: {repeat+1}/{repeats}; frame: {frame+1}/{num_frames}")

    im.set_data(grid)
    return [im]


#run simulation
plt.figure(figsize=(14,10))
x_avs_list = np.zeros((repeats, num_frames))
for repeat in range(repeats):
    #starting position
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

    x_avs = []

    #simulation
    for frame in range(num_frames):
      update(frame)
    x_avs_list[repeat, :] = x_avs
    plt.plot(np.arange(0, len(x_avs), 1), x_avs, 'midnightblue', linewidth=1)

x_avs_av = np.mean(x_avs_list, axis=0)

#plot evolution of average x pos
plt.plot(np.arange(0, len(x_avs_av), 1), x_avs_av, 'yellow')

plt.xlim(0, len(x_avs_av))
plt.ylim(0, x_dim)
plt.xlabel('time')
plt.ylabel('mean x position')
plt.grid()
plt.show()
