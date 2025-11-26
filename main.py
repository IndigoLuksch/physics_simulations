import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm #progres bar
import sys #to exit if error occurs
import time #to calculate processing time
import datetime #for saving file
import os

import config
from config import SimulationConfig
from physics import Simulation

#---setup configuration---
start_time = time.time()

# You can change 'Al' to other materials if you implemented them
cfg = SimulationConfig.from_material('Al')
sim = Simulation(cfg)

print(f"Simulation initialised: {cfg.x_dim}x{cfg.y_dim} grid")
print(f"Physics steps per frame: {cfg.steps_per_frame}")

date_time = datetime.datetime.now()
folder = date_time.strftime("%Y-%m-%d_%H%M")
folder_path = config.OUTPUTS_DIR + f"/{folder}"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#---setup visualisation---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#grains
cmap = plt.get_cmap('tab20c')
cmap.set_under('black')  #liqiud (0) --> black
vmax = sim.cfg.x_dim * sim.cfg.y_dim
im_grains = ax1.imshow(sim.grid, cmap=cmap, vmin=1, vmax=vmax, origin='upper', interpolation='nearest') #vmin=1 --> 0 maps to 'under' colour (black)
ax1.set_title("Grain Growth Simulation")
ax1.axis('off')

#temperature
im_temp = ax2.imshow(sim.T_grid, cmap='plasma', vmin=0)
plt.title('Temperature')
plt.colorbar(im_temp, ax=ax2, label='Temperature / K')
text = ax2.text(0,33,[])
ax2.axis('off')
'''
#temperature difference 
im_temp_dif = ax3.imshow(sim.T_grid_dif, cmap='plasma', vmin=0, vmax = 0.01)
plt.title('Temperature difference from mean')
plt.colorbar(im_temp_dif, ax=ax3, label='Temperature / K')
ax3.axis('off')
'''

#progress bar
total_frames = int(cfg.ani_duration * cfg.ani_fps)
pbar = tqdm(total=total_frames, desc="Simulating", unit="frame")


#---animation loop---
def update(frame):
    for _ in range(cfg.steps_per_frame): #multiple physics steps per frame
        sim.step()

    grid_copy = sim.grid.copy() #copy grid to avoid potential threading issues
    T_grid_copy = sim.T_grid.copy()
    im_grains.set_data(grid_copy)
    im_temp.set_data(T_grid_copy)
    text.set_text(f"Average temperature = {np.mean(T_grid_copy):.2f} K")

    if sim.detect_oscillation():
        print("Oscillation detected!")
        sys.exit()

    pbar.update(1)
    return [im_grains, im_temp]

#---run and save---
ani = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=1000 / cfg.ani_fps,  #ms
    blit=True
)

#save mp4
print(f"Saving animation...")
mp4_filename = 'grain_growth.mp4'
mp4_full_path = f"{folder_path}/{mp4_filename}"
ani.save(mp4_full_path,
         writer='ffmpeg',
         fps=120,
         #progress_callback=update_progress,
         dpi=150) #resolution
pbar.close()

end_time = time.time()

#save parameters
processing_time = round((end_time - start_time)/60)
parameters_filename = 'parameters.txt'
parameters_full_path = f"{folder_path}/{parameters_filename}"
parameters_string = f"Material: {cfg.material_name}\nProcessing time: {processing_time}\nSimulation time: {cfg.simulation_time}s\ndt: {cfg.dt}s\nx, y dimensions: {cfg.x_dim}x{cfg.y_dim}\nPixel size: {cfg.pix_dim}\nPhysics steps per frame: {cfg.steps_per_frame}"
with open(parameters_full_path, 'w') as f:
    f.write(parameters_string)

print(f"last temperature value at [0,10]: {sim.T_history.copy()[-1][0,10]}\ntemperature drop here = {cfg.T_initial - sim.T_history.copy()[-1][0,10]}")