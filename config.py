import os
import sys

#paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

#create directories if they do not exist already
for directory in [OUTPUTS_DIR, PROJECT_ROOT]:
    os.makedirs(directory, exist_ok=True)

#PARAMETERS

#---simulation parameters---
x_dim = 10 #arb
y_dim = 10 #arb
pix_dim = 2e-5 #side length of each pixel
dt = 7e-7
simulation_time = 1
q_reduction = 1
latent_heat_reduction = 0
same_state_pref = 10

#---animation parameters---
ani_fps = 60
ani_duration = 20

#---material parameters---
material = 'Al'

if material == 'Al':
    #PURE ALUMINIUM
    #phase diagram
    T_melt = 660 + 273
    #k_S = 220           #W/(m*K) (thermal conductivity) (assume constant within each phase)
    #k_L = 100
    #heat flow -- note thermal conductivity is for temp gradient, heat transfer coefficient is for interface
    k_LS = 230          #thermal conductivity: S/L<-->S/L [7]
    h_Sm = 0.5          #heat transfer coefficient: S<-->mould [6]
    h_Lm = 1.3          #heat transfer coefficient: L<-->mould [6]

    shc = 921           #J/(K kg)
    #energy
    dH_LS = 1067000000  #J/(m^3)
    E_srf_SS = 0.3         #surface energy between two grains (assume independent of grain angle) (assume E_srf_LS = 0) (source: [1])
    E_srf_SS = 2 #delete this
    E_srf_LS = 0.2 * E_srf_SS
    #other
    density = 2700      #kg/(m^3)
    n = 5.979e28 * pix_dim**3  #atoms per pixel (mm^3)
    #casting parameters
    T_initial = 900 +273 #change to 1500°C
    T_mold_preheat = 1000 + 273 # <--- not used currently
    T_mould = 30 + 273        #(assume constant)

    #q = -k ∆T (W/m^2)
else:
    print("Material choice not recognised. Please retry.\n\n[EXECUTION STOPPED]")
    sys.exit()