import numba

import config

def initialise():
    grid = np.zeros((x_dim, y_dim))  # states

    T_grid = np.ones((x_dim, y_dim)) * T_initial  # temp
    T_grid_history = []

    phase_changes = np.zeros((x_dim, y_dim))  # ~d/dt(states)

#---Courant-Friedrichs-Lewy condition---
def Courant_Friedrichs_Lewy_condition():
    if np.max(k_LS) * dt / (pix_dim**2 * shc * density) > 0.25 * 1.05: #0.25 with a slight buffer
        print(f"np.max(k_LS) * dt / (pix_dim**2 * shc * density) = {np.max(k_LS) * dt / (pix_dim**2 * shc * density)} > 0.25\n\nCourant-Friedrichs-Lewy condition not met --> unstable temperature field expected\ndecrease dt or increase pix_dim")
        sys.exit()
    else: print(f"np.max(k_LS) * dt / (pix_dim**2 * shc * density) = {np.max(k_LS) * dt / (pix_dim**2 * shc * density)} < 0.25\n\nCourant-Friedrichs-Lewy condition met")

#determine if state is L (0) or S (1)
@numba.jit
def phase(state):
    if state == 0:
        return 0
    else:
        return int(state/state)

#calculate E_srf at an atomic site
@numba.jit
def calc_E_srf(state, x, y, f_grid, E_srf_SS=E_srf_SS, E_srf_LS=E_srf_LS):
    new_E_srf = 0
    if state != 0: #if not L
        dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dx_dy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim: #if neighbour pixel in grid
                if f_grid[nx, ny] > 0 and f_grid[nx, ny] != state: #if different from new state and not L
                    new_E_srf += E_srf_SS
    else: #if L
        dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dx_dy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim: #if neighbour pixel in grid
                if f_grid[nx, ny] > 0: #if S
                    new_E_srf += E_srf_LS
    return new_E_srf


#calculate ∆G for state change at one site
@numba.jit
def calc_dG(state_1, state_2, x, y, grid, T_grid, dH_LS=dH_LS, dS_LS = dS_LS):
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

#test for temperature oscillations (instability)
@numba.jit
def test_T_oscillation(T_grid_history, window, pix_to_test=3):
    if len(T_grid_history) >= pix_to_test:
        T_3 = T_grid_history[-1]
        T_2 = T_grid_history[-2]
        T_1 = T_grid_history[-3]

        #random pixels to test
        pixels = []
        for _ in range(pix_to_test):
            pixels.append([random.randint(int(window*x_dim),int((1-window)*x_dim)),random.randint(int(window*y_dim),int((1-window)*y_dim))])

        #test
        for pixel in pixels:
            p1 = T_1[pixel[0],pixel[1]]
            p2 = T_2[pixel[0],pixel[1]]
            p3 = T_3[pixel[0],pixel[1]]
            if abs(p1-p2) > 100: #test if large ∆T
                if np.sign(p1-p2) != np.sign(p2-p3): #test if ∆T changes sign
                    return True
                else: return False
            else: return False
    return None

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

#temperature loop
def update_temperature(T_grid, grid, phase_changes, k_LS, h_Sm, h_Lm, pix_dim, density, dH_LS, dt, shc, q_reduction):
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

def update(frame):
    for _ in range(steps_per_frame):
        global grid, im, num_frames, phase_changes, T_grid, batch_size

        phase_changes = np.zeros_like(phase_changes)
        selected_sites = [[random.randint(0,x_dim-1), random.randint(0,y_dim-1)] for _ in range(batch_size)]

        #loop across selected sites

        for x, y in selected_sites:
            #create list of possible states
            possible_states = get_neighbors(x, y, grid, x_dim, y_dim)
            possible_states.append(grid[x,y]) #add current state
            possible_states.append(0) #add L

            possible_states.append(random.randint(1,x_dim*y_dim)) #add new grain

            possible_states = list(set(possible_states)) #remove duplicates
            possible_states = [int(i) for i in possible_states]

            #calc probability of each state
            state_probability = []
            for state in possible_states:
                dG = calc_dG(grid[x,y], state, x, y, grid, T_grid) #apply function

                if state == grid[x,y]: #if no change
                    dG *= same_state_pref #increase probability

                exponent = -dG / (k_B * T_grid[x,y])
                if exponent > 10: probability = np.exp(10)
                else: probability = np.exp(exponent) #Boltzmann distribution

                state_probability.append(probability)
            #print(f"dG = {dG}, possible_states = {possible_states}, state_probability = {state_probability}, T = {T_grid[x,y]}")

            #select state
            if np.sum(state_probability) > 0:
                index = random.choices(np.arange(0,len(possible_states),1), state_probability)[0]
                new_state = (possible_states[index]) #int? plt.imshow doesnt like
                #print(f"new_state: {new_state}")

                #if state changes
                if new_state != grid[x,y]:
                    phase_change = phase(new_state) - phase(grid[x,y])
                    phase_changes[x,y] = phase_change #update phase_changes matrix
                    grid[x,y] = new_state #updates state
                #print(f"new_grid_xy: {grid[x,y]}")

    if frame % 1 == 0:
        T_grid = update_temperature(T_grid, grid, phase_changes, k_LS, h_Sm, h_Lm, pix_dim, density, dH_LS, dt, shc, q_reduction)
        T_grid_history.append(T_grid)

        #print(T_grid[0,5])



        oscillation = test_T_oscillation(T_grid_history, 0.1)
        if oscillation:
            print("OSCILLATION DETECTED")


    #stats
    '''
    unique, counts = np.unique(grid, return_counts=True)
    num_crystals_add = len(unique)
    count_dict = dict(zip(unique, counts))
    percent_L_add = 100*count_dict[0]/len(grid)
    num_crystals.append(num_crystals_add)
    percent_L.append(percent_L_add)
    '''
    grid = grid.astype(np.float32)
    #im_grain = ax.imshow(grid, cmap='hsv', vmin=0)

    im_grain.set_data(grid)
    im_temp.set_data(T_grid)
    return [im_grain, im_temp]