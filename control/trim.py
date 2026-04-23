# control/trim.py

import numpy as np
from scipy.optimize import minimize
from dynamics.system import full_dynamics

def calculate_trim(target_velocity, altitude, base_state, base_control, params):
    """
    Finds the straight and level trim condition for the aircraft.
    """
    print(f"--- Starting Trim Routine for V = {target_velocity} m/s ---")

    # Objective function for the optimizer
    def cost_function(trim_vars):
        # Unpack optimizer variables
        alpha, elevator, throttle = trim_vars

        # Update State Vector
        # For straight and level flight, pitch angle (theta) equals alpha
        # Velocity components are derived from target_velocity and alpha
        state = np.copy(base_state)
        state[0] = target_velocity * np.cos(alpha)  # u
        state[2] = target_velocity * np.sin(alpha)  # w
        state[7] = alpha                            # theta
        
        # Ensure angular rates are zero for steady flight
        state[3:6] = 0.0  # p, q, r
        
        # 2. Update Control Vector
        # Assuming symmetric flight (aileron=0, rudder=0)
        control = np.copy(base_control)
        control[0] = 0.0       # aileron
        control[1] = elevator  # elevator
        control[2] = 0.0       # rudder
        control[3] = throttle  # throttle_1
        control[4] = throttle  # throttle_2

        # 3. Calculate dynamics
        state_dot = full_dynamics(0, state, control, params)

        # 4. Extract accelerations we want to drive to zero
        du = state_dot[0]
        dw = state_dot[2]
        dq = state_dot[4]

        # 5. Cost is the sum of squared errors (we want this to be 0)
        # We weight them heavily to ensure the optimizer cares about all of them equally
        cost = (du**2) + (dw**2) + (dq**2 * 100) 
        
        return cost

    # Initial Guesses: [alpha (rad), elevator (rad), throttle (0-1)]
    initial_guess = [0.05, 0.0, 0.5] 

    # Bounds to keep the optimizer in realistic physics limits
    # alpha: -10 to 15 deg, elevator: -25 to 10 deg, throttle: 0 to 1
    bounds = (
        (-np.radians(10), np.radians(15)), 
        (-np.radians(25), np.radians(10)), 
        (0.0, 1.0)
    )

    # Run the optimizer
    res = minimize(
        cost_function, 
        initial_guess, 
        bounds=bounds, 
        method='SLSQP',
        options={'ftol': 1e-9, 'disp': False}
    )

    if res.success:
        alpha_trim, elev_trim, throt_trim = res.x
        
        # Build the final trimmed state and control arrays
        trim_state = np.copy(base_state)
        trim_state[0] = target_velocity * np.cos(alpha_trim)
        trim_state[2] = target_velocity * np.sin(alpha_trim)
        trim_state[7] = alpha_trim
        trim_state[3:6] = 0.0
        
        trim_control = np.copy(base_control)
        trim_control[0] = 0.0
        trim_control[1] = elev_trim
        trim_control[2] = 0.0
        trim_control[3] = throt_trim
        trim_control[4] = throt_trim

        print("Trim Successful!")
        print(f" Alpha:    {np.degrees(alpha_trim):.3f} deg")
        print(f" Elevator: {np.degrees(elev_trim):.3f} deg")
        print(f" Throttle: {throt_trim * 100:.2f} %")
        print("---------------------------------------------------")
        
        return trim_state, trim_control
    else:
        print("Trim Failed! The optimizer could not find an equilibrium.")
        print(res.message)
        # Return base arrays if it fails so the simulation can still try to run
        return base_state, base_control