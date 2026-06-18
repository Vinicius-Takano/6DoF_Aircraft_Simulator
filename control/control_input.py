# control/control_input.py

import numpy as np

def apply_control_pulse(t, base_control, start_time, duration, channel_idx, pulse_value):
    """
    Applies a single step/pulse input.
    """
    current_control = np.copy(base_control)
    
    if start_time <= t <= (start_time + duration):
        # Aerodynamic surfaces (deg -> rad)
        if channel_idx in [0, 1, 2]:
            modifier = np.radians(pulse_value)
        # Throttles and Wind (apply raw value)
        else:
            modifier = pulse_value 
            
        current_control[channel_idx] += modifier
        
    return current_control


def apply_control_doublet(t, base_control, start_time, duration, channel_idx, doublet_magnitude):
    """
    Applies a doublet input (positive pulse followed immediately by a negative pulse).
    `duration` is the length of ONE half of the doublet.
    """
    current_control = np.copy(base_control)
    
    # First half: Positive magnitude
    if start_time <= t <= (start_time + duration):
        if channel_idx in [0, 1, 2]:
            modifier = np.radians(doublet_magnitude)
        else:
            modifier = doublet_magnitude
            
        current_control[channel_idx] += modifier
        
    # Second half: Negative magnitude
    elif (start_time + duration) < t <= (start_time + 2 * duration):
        if channel_idx in [0, 1, 2]:
            modifier = np.radians(-doublet_magnitude)
        else:
            modifier = -doublet_magnitude
            
        current_control[channel_idx] += modifier
        
    return current_control


def apply_alpha_beta_gust(t, state, base_control, start_time, duration, delta_alpha_deg=0.0, delta_beta_deg=0.0):
    """
    Calculates and injects an Earth-fixed wind gust to induce a specific delta alpha or beta.
    Requires the current `state` vector to resolve body-to-earth transformations.
    """
    current_control = np.copy(base_control)
    
    if start_time <= t <= (start_time + duration):
        # Extract kinematics
        u, v, w = state[0:3]
        phi, theta, psi = state[6:9]
        
        # Calculate current airspeed (ignoring existing wind for the gust delta calculation)
        V_a = np.sqrt(u**2 + v**2 + w**2)
        V_a = np.maximum(V_a, 1e-6) # Prevent division by zero on the ground
        
        # 1. Calculate required body-frame wind velocities using small angle trig:
        # tan(delta_alpha) = W_body / V_a  -->  W_body = V_a * tan(delta_alpha)
        # W_body is downward (positive Z). To increase alpha, wind must hit from below (negative Z)
        w_wind_body = -V_a * np.tan(np.radians(delta_alpha_deg))
        
        # tan(delta_beta) = V_body / V_a   --> V_body = V_a * tan(delta_beta)
        # To increase beta (nose left relative to wind), wind must hit from the right (negative Y)
        v_wind_body = -V_a * np.tan(np.radians(delta_beta_deg))
        
        u_wind_body = 0.0 # A headwind/tailwind gust changes Va, but doesn't change angles
        
        V_wind_body = np.array([u_wind_body, v_wind_body, w_wind_body])
        
        # 2. Transform the body-frame wind gust into the Earth frame
        # DCM matrix (Body to Earth is the transpose of Earth to Body)
        R_b2e = np.array([
            [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                       np.cos(phi)*np.cos(theta)]
        ])
        
        V_wind_earth = R_b2e @ V_wind_body
        
        # 3. Add the calculated gust to the control vector (Indices 5, 6, 7)
        current_control[5] += V_wind_earth[0] # Wind North
        current_control[6] += V_wind_earth[1] # Wind East
        current_control[7] += V_wind_earth[2] # Wind Down
        
    return current_control