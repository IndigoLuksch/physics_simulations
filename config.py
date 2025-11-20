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

#processed parameters
steps_per_frame = max(1, int(simulation_time / (dt * ani_fps * ani_duration) ) )

#---constants---
k_B = 1.38e-23

#---material parameters---
material = 'Al'

if material == 'Al':
    #PURE ALUMINIUM
    n = 5.979e28 * pix_dim**3  #atoms per pixel (mm^3)
    #casting parameters
    T_initial = 900 +273 #change to 1500°C
    T_mold_preheat = 1000 + 273 # <--- not used currently
    T_mould = 30 + 273        #(assume constant)

    #q = -k ∆T (W/m^2)
else:
    print("Material choice not recognised. Please retry.\n\n[EXECUTION STOPPED]")
    sys.exit()


from dataclasses import dataclass

@dataclass(frozen=True)
class PureMaterialProperties:
    T_melt: float
    k_LS: float  #thermal conductivity: S/L<-->S/L
    h_Sm: float  #heat transfer coefficient: S<-->mould
    h_Lm:float #heat transfer coefficient: L<-->mould
    shc: float #J/(K kg)
    dH_LS: float
    E_srf_SS: float #surface energy between two grains (assume independent of grain angle) (assume E_srf_LS = 0)
    E_srf_LS: float
    density: float  #kg/(m^3)

ALUMINIUM = PureMaterialProperties(
    660 + 273,
    230, #[7]
    0.5, #[6]
    1.3, #[6]
    921,
    1067000000,
    0.3, #[1]
    0.2,
    2700
)




#processed parameters
dS_LS = dH_LS / T_melt