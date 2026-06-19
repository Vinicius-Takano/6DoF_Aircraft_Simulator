# control/control_input.py

import numpy as np

# control/control_input.py
import numpy as np

import numpy as np

def apply_control_step(t, base_control, start_time, channel_idx, step_value):
    """
    Applies a step input. Once triggered, the control surface remains deflected.
    Accepts single integers or lists/arrays for multi-channel inputs (like wind vectors).
    """
    current_control = np.copy(base_control)
    
    if t >= start_time:
        # Check if the user passed a list/array of channels (e.g., [5, 6, 7])
        if isinstance(channel_idx, (list, tuple, np.ndarray)):
            for ch, val in zip(channel_idx, step_value):
                # Convert to radians only if it's an aero surface
                modifier = np.radians(val) if ch in [0, 1, 2] else val
                current_control[ch] += modifier
        else:
            # Standard single-channel logic
            modifier = np.radians(step_value) if channel_idx in [0, 1, 2] else step_value
            current_control[channel_idx] += modifier
            
    return current_control

def apply_control_doublet(t, base_control, start_time, duration, channel_idx, doublet_magnitude):
    """
    Applies a doublet input (positive pulse followed immediately by a negative pulse).
    `duration` is the length of ONE half of the doublet.
    If applied to a thruster (index 3 or 4), it affects BOTH thrusters symmetrically.
    """
    current_control = np.copy(base_control)
    
    # 1. Determine the magnitude based on the current time window
    if start_time <= t <= (start_time + duration):
        magnitude = doublet_magnitude        # First half: Positive
    elif (start_time + duration) < t <= (start_time + 2 * duration):
        magnitude = -doublet_magnitude       # Second half: Negative
    else:
        return current_control               # Outside doublet window, return base
        
    # 2. Apply the magnitude to the appropriate channels
    if channel_idx in [0, 1, 2]:
        modifier = np.radians(magnitude)
        current_control[channel_idx] += modifier
        
    else:
        modifier = magnitude
        # If the target is a throttle (index 3 or 4), apply to BOTH engines
        if channel_idx in [3, 4]:
            current_control[3] += modifier  # throttle_1
            current_control[4] += modifier  # throttle_2
        else:
            current_control[channel_idx] += modifier # Wind parameters
            
    return current_control

def apply_wind_pulse(t, base_control, start_time, duration, wind_vector):
    """
    Applies a temporary Earth-fixed wind gust.
    wind_vector should be a list or array: [Wind_N, Wind_E, Wind_D] in m/s.
    """
    current_control = np.copy(base_control)
    
    # Strict time window for a pulse
    if start_time <= t <= (start_time + duration):
        
        # Add the wind vector to channels 5, 6, and 7
        current_control[5] += wind_vector[0]
        current_control[6] += wind_vector[1]
        current_control[7] += wind_vector[2]
        
    return current_control