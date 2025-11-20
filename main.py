import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
import sys

import config
from config import SimulationConfig
from physics import Simulation

#---setup configuration---
# You can change 'Al' to other materials if you implemented them
cfg = SimulationConfig.from_material('Al')
sim = Simulation(cfg)

print(f"Simulation Initialized: {cfg.x_dim}x{cfg.y_dim} grid")
print(f"Physics Steps per Frame: {cfg.steps_per_frame}")

#---setup visualisation---
fig, ax = plt.subplots(figsize=(6, 6))

cmap = plt.get_cmap('tab20c')
cmap.set_under('black')  #liqiud (0) --> black

vmax = sim.cfg.x_dim * sim.cfg.y_dim

#create initial image object
#vmin=1 --> 0 maps to 'under' colour (black)
im = ax.imshow(sim.grid, cmap=cmap, vmin=1, vmax=vmax, origin='upper', interpolation='nearest')
ax.set_title("Grain Growth Simulation")
ax.axis('off')  # Hide axes ticks

#progress bar
total_frames = int(cfg.ani_duration * cfg.ani_fps)
pbar = tqdm(total=total_frames, desc="Simulating", unit="frame")


#---animation loop---
def update(frame):
    for _ in range(cfg.steps_per_frame): #multiple physics steps per frame
        sim.step()

    data = sim.grid.copy() #copy grid to avoid potential threading issues
    im.set_data(data)

    if sim.detect_oscillation():
        print("Oscillation detected!")
        sys.exit()

    pbar.update(1)
    return [im]

#---run and save---
ani = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=1000 / cfg.ani_fps,  #ms
    blit=True
)

# Save to GIF (requires Pillow) or MP4 (requires FFmpeg)
print(f"Saving animation...")
filename = 'grain_growth.mp4'
full_path = config.OUTPUTS_DIR + f"/{filename}"
ani.save(full_path,
         writer='ffmpeg',
         fps=120,
         #progress_callback=update_progress,
         dpi=150) #resolution
pbar.close()