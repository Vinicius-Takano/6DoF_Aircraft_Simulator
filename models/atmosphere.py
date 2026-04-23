import numpy as np

def compute_atmospheric_properties(state):
    # Simple exponential atmosphere model
    _, _, p_d = state[9:12]
    
    altitude = -p_d  # Convert down to altitude

    if altitude < 11000:
        p0 = 101325  # Sea level standard atmospheric pressure (Pa)
        T0 = 288.15  # Sea level standard temperature (K)
        T=288.15-6.5*10**(-3)*altitude
        V_sound = np.sqrt(1.4*287*T)
        rho = 1.225*(T/T0)**(-1-9.81/(287*(-6.5*10**(-3)))) 
    else:
        T = 216.66  
        V_sound = np.sqrt(1.4*287*T) 
        rho = 0.36391 * (np.exp(-9.8*(altitude-11000)/(287*T)))

    return rho, V_sound