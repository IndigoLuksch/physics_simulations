

#---Courant-Friedrichs-Lewy condition---
def Courant_Friedrichs_Lewy_condition():
    if np.max(k_LS) * dt / (pix_dim**2 * shc * density) > 0.25 * 1.05: #0.25 with a slight buffer
        print(f"np.max(k_LS) * dt / (pix_dim**2 * shc * density) = {np.max(k_LS) * dt / (pix_dim**2 * shc * density)} > 0.25\n\nCourant-Friedrichs-Lewy condition not met --> unstable temperature field expected\ndecrease dt or increase pix_dim")
        sys.exit()
    else: print(f"np.max(k_LS) * dt / (pix_dim**2 * shc * density) = {np.max(k_LS) * dt / (pix_dim**2 * shc * density)} < 0.25\n\nCourant-Friedrichs-Lewy condition met")