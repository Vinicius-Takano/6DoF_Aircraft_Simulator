# control/script.py

import numpy as np

def apply_control_pulse(t, base_control, start_time, duration, channel_idx, pulse_value):
    """
    Applies a pulse/step input to a specific control channel.
    
    Args:
        t (float): Current simulation time from solve_ivp.
        base_control (np.array): The initial/trimmed control vector.
        start_time (float): Time (s) when the pulse begins.
        duration (float): Duration (s) of the pulse.
        channel_idx (int): Index of the control vector to modify 
                           (0: aileron, 1: elevator, 2: rudder, 3: throttle_1, 4: throttle_2).
        pulse_value (float): The magnitude to ADD to the base control (in degrees for aero surfaces).
    
    Returns:
        np.array: The modified control vector for time `t`.
    """
    # Copy the base control so we don't permanently alter the original trim state
    current_control = np.copy(base_control)
    
    # Check if the current solver time is within the active pulse window
    if start_time <= t <= (start_time + duration):
        
        # If it's an aerodynamic surface (indices 0, 1, 2), convert the pulse to radians
        if channel_idx in [0, 1, 2]:
            modifier = pulse_value * (np.pi / 180.0)
        else:
            # If it's a throttle (indices 3, 4), apply the percentage raw
            modifier = pulse_value 
            
        current_control[channel_idx] += modifier
        
    return current_control