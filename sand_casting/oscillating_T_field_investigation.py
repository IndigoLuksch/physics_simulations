'''PURE elements -- TESTING FOR OSILLATION STABILITY LIMIT'''

#---import libraries---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
from tqdm import tqdm
import scipy as sp
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import random
import ffmpeg
import numba

#---notes---
'''
• THERMODYNAMIC CALCULATIONS AND GRAIN GROWTH SIMULATIONS ARE NOT ACCURATE. THIS CODE IS PURELY USED TO INVESTIGATE TEMPERATURE FIELD DYNAMICS. GRAIN GROWTH SIMULATIONS CAN BE DISABLED
• simulates pure elemental metals (i.e. T_solidus, T_liquidus are constant)
• all units are SI
• each pixel represents a voxel
• we include surface energy contributions in the Gibbs free energy
'''

#---simulation parameters---
x_dim = 100 #arb
y_dim = 100 #arb
pix_dim = 1e-6
dt = 1e-6
q_reduction = 1
simulation_time = 0.01
same_state_pref = 10

#---material parameters---
#material = str(input("Choose a material.\n\nOptions: 'Al'"))
material = 'Al'

if material == 'Al':
    #PURE ALUMINIUM
    #phase diagram
    T_melt = 660 + 273
    #k_S = 220           #W/(m*K) (thermal conductivity) (assume constant within each phase)
    #k_L = 100
    #heat flow
    h_LS = 230          #heat transfer coefficient: S/L<-->S/L [7]
    h_Sm = 0.5          #heat transfer coefficient: S<-->mould [6]
    h_Lm = 1.3          #heat transfer coefficient: L<-->mould [6]

    #CHANGE BELOW
    h_Sm = h_LS
    h_Lm = h_LS
    #CHANGE ABOVE

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
    T_initial = 700 +273 #change to 1500°C
    T_mold_preheat = 1000 + 273 # <--- not used currently
    T_mould = 30 + 273        #(assume constant)

    #q = -k ∆T (W/m^2)
else:
    print("Material choice not recognised. Please retry.\n\n[EXECUTION STOPPED]")
    sys.exit()


#---initial processing of parameters---
k_B = 1.38e-23
dS_LS = dH_LS / T_melt

num_frames = int(round(simulation_time / dt))

grid = np.zeros((x_dim,y_dim)) #states

#---random nucleation site initialisation---
nucleation_sites = []
for _ in range(30):
    nucleation_site = [random.randint(0,x_dim-1),random.randint(0,y_dim-1)]
    if nucleation_site not in nucleation_sites:
        nucleation_sites.append(nucleation_site)

#for each nucleation site, initialise new phase #note: need to reorder
for i in range(len(nucleation_sites)):
    j = nucleation_sites[i]
    grid[j[0],j[1]] = i+1

phase_changes = np.zeros((x_dim,y_dim)) #~d/dt(states)

T_grid = np.ones((x_dim, y_dim)) * T_initial #temp
T_grid_history = []

#---define functions---
#determine if state is L (0) or S (1)
def phase(state):
    if state == 0:
        return 0
    else:
        return int(state/state)


#calculate E_srf at an atomic site
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
def calc_dG(state_1, state_2, x, y, grid, T_grid, dH_LS=dH_LS, dS_LS = dS_LS):
    #state_1 = old state, state_2 = proposed new state
    #find E_srf contribution (penalty)
    initial_E_srf = calc_E_srf(state_1, x, y, grid)
    new_E_srf = calc_E_srf(state_2, x, y, grid)
    dE_srf = new_E_srf - initial_E_srf

    state_1 = phase(state_1) #must be after E_srf loop
    state_2 = phase(state_2)

    #dG = -(dH_LS - T_grid[x,y] * dS_LS) * (state_2 - state_1) * (pix_dim**3) / n + dE_srf * (pix_dim**2) / (6 * n ** (2/3)) #apparently last denominator should just be n
    dG = (dH_LS - T_grid[x,y] * dS_LS) * (state_2 - state_1) * (pix_dim**3) / n + dE_srf * (pix_dim**2) / n #apparently last denominator should just be n
    return dG
    #returns dG per atom, on average

#---define loop functions
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

def update_temperature(T_grid, grid, phase_changes):
    global h_LS, h_Sm, h_Lm, pix_dim, density, dH_LS, dt
    k_grid = np.zeros_like(grid)
    k_grid[grid == 0] = h_LS  #liquid conductivity
    k_grid[grid > 0] = h_LS   #Solid conductivity

    T_new = T_grid.copy()

    #thermal diffusivity grid 
    alpha_grid = k_grid / (density * shc)

    laplacian = (
        np.roll(T_grid, 1, axis=0) +
        np.roll(T_grid, -1, axis=0) +
        np.roll(T_grid, 1, axis=1) +
        np.roll(T_grid, -1, axis=1) - 4 * T_grid
    ) / (pix_dim**2)

    # Update temperature from conduction
    T_new += alpha_grid * laplacian * dt * q_reduction
    h_mould_top = np.where(grid[0, :] == 0, h_Lm, h_Sm) # h based on phase
    q_top = (T_mould - T_grid[0, :]) * h_mould_top # Heat flux (W/m^2)
    dT_top = (q_top * dt) / (shc * density * pix_dim) # dT = (Flux * Area * dt) / (c_p * m)
    T_new[0, :] += dT_top
    latent_heat_T_change = (phase_changes * dH_LS) / (shc * density)
    T_new += latent_heat_T_change

    sigma = [2, 2]
    T_new = sp.ndimage.filters.gaussian_filter(T_new, sigma, mode='constant')

    return T_new

#state loop
def update():
    global grid, num_frames, phase_changes, T_grid

    phase_changes = np.zeros_like(phase_changes)
    batch_size = int(x_dim * y_dim * 0.05) #<--- PURPOSE OF THIS?
    selected_sites = [[random.randint(0,x_dim-1), random.randint(0,y_dim-1)] for _ in range(batch_size)]

    #loop across selected sites

    for x, y in selected_sites:
        #create list of possible states
        neighbours = []
        dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dx_dy:
            nx, ny = x + dx, y + dy
            if 0 <= nx < x_dim and 0 <= ny < y_dim and grid[nx, ny] > 0:
                neighbours.append(grid[nx, ny])
        possible_states = neighbours
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

    T_grid = update_temperature(T_grid, grid, phase_changes)
    T_grid_history.append(T_grid)
    osc = test_T_oscillation(T_grid_history, 0.1)
    if osc:
        print("OSCILLATION DETECTED")
    return osc

'''TESTING FOR STABILITY LIMIT'''
dts = [5e-4]
pix_dims = []
for dt in dts:
    pix_dim = 1e-6
    grid = np.zeros((x_dim,y_dim))
    T_grid = np.ones((x_dim, y_dim)) * T_initial #temp
    T_grid_history = []
    f = 2.5
    pix_dim = 1e-5
    oscillations = []

    for repeat in range(15):
        oscillation = False
        for _ in range (10000):
            oscillation = update()
            if oscillation:
                break
        oscillations.append(oscillation)
        if len(oscillations) > 1:
            if oscillations[-1] != oscillations[-2]:
                f = 0.5*(f-1)+1
        if oscillation:
            pix_dim *= f
        else:
            pix_dim /= f
        print(f"dt: {dt} ({repeat}/10), oscillation: {oscillation}, pix_dim: {pix_dim}")
    pix_dims.append(pix_dim)

plt.plot(dts, pix_dims)
plt.show()
