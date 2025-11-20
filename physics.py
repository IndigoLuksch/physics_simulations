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