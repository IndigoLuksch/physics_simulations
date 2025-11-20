import os
import sys
from dataclasses import dataclass, field

#paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

#create directories if they do not exist already
for directory in [OUTPUTS_DIR, PROJECT_ROOT]:
    os.makedirs(directory, exist_ok=True)

@dataclass
class SimulationConfig:
    #---simulation parameters---
    x_dim: int = 30
    y_dim: int = 30
    pix_dim: float = 2e-5
    dt: float = 7e-7
    simulation_time: float = 1.0

    #---animation and tuning---
    ani_fps: float = 60
    ani_duration: float = 20

    q_reduction: float = 1.0
    latent_heat_reduction: float = 0.0
    same_state_pref: float = 10.0

    #---material properties (default: Al)---
    T_melt: float = 933.0  # K
    T_initial: float = 1173.0  # K
    T_mould: float = 303.0  # K

    k_LS: float = 230.0  # W/(m*K)
    h_Sm: float = 0.5
    h_Lm: float = 1.3
    dH_LS: float = 1.067e9  # J/m^3

    E_srf_SS: float = 0.3
    density: float = 2700.0
    shc: float = 921.0

    # --- Constants ---
    k_B: float = 1.38e-23

    #---calculated values---
    # init=False means "Do not ask for this in the constructor"
    n: float = field(init=False)
    steps_per_frame: int = field(init=False)
    dS_LS: float = field(init=False)
    E_srf_LS: float = field(init=False)

    def __post_init__(self):
        """"calculates derived values"""

        self.dS_LS = self.dH_LS / self.T_melt
        self.n = 5.979e28 * (self.pix_dim ** 3)
        self.steps_per_frame = max(1, int(self.simulation_time / (self.dt * self.ani_fps * self.ani_duration)))
        self.E_srf_LS = 0.2 * self.E_srf_SS

    @classmethod
    def from_material(cls, material_name: str):
        if material_name == 'Al':
            return cls(
                T_melt=933,
                k_LS=230,
                h_Sm=0.5,
                h_Lm=1.3,
                dH_LS=1.067e9,
                E_srf_SS=0.3,
                density=2700,
                shc=921,
                T_initial=1173,
                T_mould=303
            )
        raise ValueError(f"Unknown material: {material_name}")
