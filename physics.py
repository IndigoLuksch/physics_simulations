#only import numerical libraries -- importing matplotlib and defining animation function here could cause problems e.g. if performing parameter sweep
import numba
import numpy as np
import random
import sys

from config import SimulationConfig

#determine if state is L (0) or S (1)
@numba.jit
def phase(state):
    if state == 0:
        return 0
    else:
        return int(state/state)

#calculate E_srf at an atomic site
@numba.jit
def calc_E_srf(state, x, y, grid, E_srf_SS, E_srf_LS):
    x_dim, y_dim = grid.shape
    new_E_srf = 0
    if state != 0: #if not L
        dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dx_dy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim: #if neighbour pixel in grid
                if grid[nx, ny] > 0 and grid[nx, ny] != state: #if different from new state and not L
                    new_E_srf += E_srf_SS
    else: #if L
        dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dx_dy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim: #if neighbour pixel in grid
                if grid[nx, ny] > 0: #if S
                    new_E_srf += E_srf_LS
    return new_E_srf

@numba.jit
def calc_dG(state_1, state_2, x, y, grid, T_grid, dH_LS, dS_LS, pix_dim, n):
    """calculate ∆G for state change at one site"""
    #state_1 = old state, state_2 = proposed new state
    #find E_srf contribution (penalty)
    initial_E_srf = calc_E_srf(state_1, x, y, grid)
    new_E_srf = calc_E_srf(state_2, x, y, grid)
    dE_srf = (new_E_srf - initial_E_srf) * 40


    state_1 = phase(state_1) #must be after E_srf loop
    state_2 = phase(state_2)

    #dG = -(dH_LS - T_grid[x,y] * dS_LS) * (state_2 - state_1) * (pix_dim**3) / n + dE_srf * (pix_dim**2) / (6 * n ** (2/3)) #apparently last denominator should just be n
    dG = -(dH_LS - T_grid[x,y] * dS_LS) * (state_2 - state_1) * (pix_dim**3) / n + dE_srf * (pix_dim**2) / n #apparently last denominator should just be n
    return dG
    #returns dG per atom, on average

@numba.jit
def test_T_oscillation(T_current, T_prev_1, T_prev_2, x_dim, y_dim, window=0.1, pix_to_test=3):
    """detects if temperature field is oscillating (not physically correct)"""
    #pixels to test
    for _ in range(pix_to_test):
        rx = random.randint(int(window * x_dim), int((1 - window) * x_dim))
        ry = random.randint(int(window * y_dim), int((1 - window) * y_dim))

        p1 = T_prev_2[rx, ry]
        p2 = T_prev_1[rx, ry]
        p3 = T_current[rx, ry]

        #if large jump and sign change
        if abs(p1 - p2) > 100:
            if np.sign(p1 - p2) != np.sign(p2 - p3):
                return True
    return False

def update_temperature(T_grid, grid, phase_changes, T_mould, k_LS, h_Sm, h_Lm, pix_dim, density, dH_LS, dt, shc, q_reduction):
    T_new = T_grid.copy()

    #---internal conduction---
    k_grid = np.zeros_like(grid)
    k_grid[grid == 0] = k_LS  #liquid conductivity
    k_grid[grid > 0] = k_LS   #Solid conductivity

    #thermal diffusivity grid
    alpha_grid = k_grid / (density * shc)

    laplacian = (
        np.roll(T_grid, 1, axis=0) +
        np.roll(T_grid, -1, axis=0) +
        np.roll(T_grid, 1, axis=1) +
        np.roll(T_grid, -1, axis=1) - 4 * T_grid
    ) / (pix_dim**2)

    T_new += alpha_grid * laplacian * dt * q_reduction

    #---boundaries---
    # Top and bottom
    T_new[0, :] += (T_mould - T_grid[0, :]) * h_Sm * dt / (shc * density * pix_dim)
    T_new[-1, :] += (T_mould - T_grid[-1, :]) * h_Sm * dt / (shc * density * pix_dim)
    # Left and right
    T_new[:, 0] += (T_mould - T_grid[:, 0]) * h_Sm * dt / (shc * density * pix_dim)
    T_new[:, -1] += (T_mould - T_grid[:, -1]) * h_Sm * dt / (shc * density * pix_dim)

    #---latent heat---
    latent_heat_T_change = (phase_changes * dH_LS) / (shc * density)
    T_new += latent_heat_T_change * latent_heat_reduction

    '''
    sigma = [0.5, 0.5]
    T_new = sp.ndimage.filters.gaussian_filter(T_new, sigma, mode='constant')
    '''
    return T_new

@numba.jit(nopython=True)
def get_neighbors(x, y, grid, x_dim, y_dim):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim:
                if grid[nx, ny] > 0:
                    neighbors.append(grid[nx, ny])
    return neighbors

@numba.jit
def monte_carlo_step(grid, T_grid, x_dim, y_dim, pix_dim, n, dH_LS, dS_LS, E_srf_SS, E_srf_LS, k_B, same_state_pref):
    phase_changes = np.zeros((x_dim, y_dim), dtype=np.float64)

    candidates = np.zeros(7, dtype=np.int32)
    weights = np.zeros(7, dtype=np.float64)

    batch_size = int(x_dim * y_dim * 0.05)

    for _ in range(batch_size):
        #random pixel to simualate
        x = random.randint(0, x_dim - 1)
        y = random.randint(0, y_dim - 1)

        current_state = grid[x, y]
        T = T_grid[x, y]

        #---create list of candidates---
        count = 0

        #add current state, liquid, random stte
        candidates[count] = current_state
        count += 1
        candidates[count] = 0
        count += 1
        candidates[count] = random.randint(1, x_dim * y_dim)
        count += 1

        #add neighbours
        dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dx_dy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim:
                neighbor = grid[nx, ny]
                if neighbor > 0:
                    candidates[count] = neighbor
                    count += 1

        #---calculate weights---
        #no need to remove duplicates -- it does not affect probabilities
        sum_weights = 0.0

        for i in range(count):
            state = candidates[i]

            #physics
            dG = calc_dG(current_state, state, x, y, grid, T,
                         pix_dim, n, dH_LS, dS_LS, E_srf_SS, E_srf_LS)

            if state == current_state:
                dG *= same_state_pref

            #exponential with guardrails
            exponent = -dG / (k_B * T)
            if exponent > 10:
                prob = 22026.0  #~exp(10)
            elif exponent < -10:
                prob = 0.000045  #~exp(-10)
            else:
                prob = np.exp(exponent)


            weights[i] = prob
            sum_weights += prob

        #---random selection---
        if sum_weights > 0:
            r = random.random() * sum_weights
            cumulative = 0.0
            new_state = current_state

            for i in range(count):
                if weights[i] > 0:  # Skip duplicates/zeros
                    cumulative += weights[i]
                    if r <= cumulative:
                        new_state = candidates[i]
                        break

            #if change in state, update things
            if new_state != current_state:
                phase_new = phase(new_state)
                phase_old = phase(current_state)
                phase_changes[x, y] = phase_new - phase_old
                grid[x, y] = new_state

    return phase_changes

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        #initialise grids
        self.grid = np.zeros((config.x_dim, config.y_dim), dtype=np.int32)
        self.T_grid = np.ones_like(self.grid, dtype=np.float64) * config.T_initial
        self.phase_changes = np.zeros_like(self.grid, dtype=np.float64)

        # History
        self.T_history = []

    def Courant_Friedrichs_Lewy_condition(self, k_LS, dt, pix_dim, shc, density):
        "Courant-Friedrichs-Lewy condition"

        if k_LS * dt / (pix_dim ** 2 * shc * density) > 0.25 * 1.05:  # 0.25 with a slight buffer
            print(
                f"np.max(k_LS) * dt / (pix_dim**2 * shc * density) = {np.max(k_LS) * dt / (pix_dim ** 2 * shc * density)} > 0.25\n\nCourant-Friedrichs-Lewy condition not met --> unstable temperature field expected\ndecrease dt or increase pix_dim")
            sys.exit()
        else:
            print(
                f"np.max(k_LS) * dt / (pix_dim**2 * shc * density) = {np.max(k_LS) * dt / (pix_dim ** 2 * shc * density)} < 0.25\n\nCourant-Friedrichs-Lewy condition met")

    def step(self):
        """1 simulation step: monte carlo + temperature"""

        # 1. Monte Carlo (Grain Growth)
        self.phase_changes = monte_carlo_step(
            self.grid, self.T_grid,
            self.cfg.x_dim, self.cfg.y_dim, self.cfg.pix_dim, self.cfg.n,
            self.cfg.dH_LS, self.cfg.dS_LS, self.cfg.E_srf_SS, self.cfg.E_srf_LS,
            self.cfg.k_B, self.cfg.same_state_pref
        )

        # 2. Thermal Update
        self.T_grid = update_temperature(
            self.T_grid, self.grid, self.phase_changes, self.T_mould,
            self.cfg.dt, self.cfg.pix_dim, self.cfg.density, self.cfg.shc,
            self.cfg.k_LS, self.cfg.dH_LS, self.cfg.h_Lm, self.cfg.h_Sm, self.cfg.q_reduction
        )

        # 3. History Logging (Optimized: Only keep last 3)
        self.T_history.append(self.T_grid.copy())  # Use copy!
        if len(self.T_history) > 3:
            self.T_history.pop(0)

    def detect_oscillation(self):
        """
        wrapper for numba-accelerated oscillation test
        rewritten to give access to class variables
        """
        if len(self.T_history) < 3:
            return False
        else:
            return test_T_oscillation(
                self.T_history[-1],
                self.T_history[-2],
                self.T_history[-3],
                self.cfg.x_dim,
                self.cfg.y_dim
            )