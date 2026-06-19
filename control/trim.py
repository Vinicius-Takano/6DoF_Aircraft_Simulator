# control/trim.py

import numpy as np
from scipy.optimize import minimize
from dynamics.system import full_dynamics
from dynamics.kinematics import euler_to_DCM 

def calculate_trim(target_Va, altitude, base_state, base_control, params):
    """
    Finds the straight and level trim condition accounting for wind.
    Target velocity is now Airspeed (Va).
    """
    print("="*50)
    print(f"Starting Trim Routine for Va = {target_Va} m/s")
    print("="*50)

    def cost_function(trim_vars):
        alpha, elevator, throttle = trim_vars

        state = np.copy(base_state)
        control = np.copy(base_control)
        
        wind_earth = control[5:8] # wind_N, wind_E, wind_D
        
        # Update Euler Angles (pitch = alpha)
        state[6] = 0.0    # phi
        state[7] = alpha  # theta
        # state[8] (psi) doens't matter
        
        u_rel = target_Va * np.cos(alpha)
        v_rel = 0.0                                     # Assuming symmetric flight (beta = 0), so v_rel = 0
        w_rel = target_Va * np.sin(alpha)
        v_rel_body = np.array([u_rel, v_rel, w_rel])

        DCM_E2B = euler_to_DCM(state[6], state[7], state[8])
        wind_body = DCM_E2B @ wind_earth
        
        v_body_inertial = v_rel_body + wind_body        # V_body = V_relative + V_wind_body
        
        state[0:3] = v_body_inertial # u, v, w
        state[3:6] = 0.0             # p, q, r

        control[0] = 0.0       # aileron
        control[1] = elevator  # elevator
        control[2] = 0.0       # rudder
        control[3] = throttle  # throttle_1
        control[4] = throttle  # throttle_2
        control[7] = 0.0       # No horizontal gust for trim

        state_dot = full_dynamics(0, state, control, params)

        du = state_dot[0]
        dw = state_dot[2]
        dq = state_dot[4]

        return (du**2) + (dw**2) + (dq**2 * 100) 

    initial_guess = [0.05, 0.0, 0.5] 
    bounds = (
        (-np.radians(10), np.radians(15)), 
        (-np.radians(25), np.radians(10)), 
        (0.0, 1.0)
    )

    res = minimize(
        cost_function, 
        initial_guess, 
        bounds=bounds, 
        method='SLSQP',
        options={'ftol': 1e-9, 'disp': False}
    )

    if res.success:
        alpha_trim, elev_trim, throt_trim = res.x
        
        trim_state = np.copy(base_state)            # final state after trim
        trim_control = np.copy(base_control)
        
        trim_state[6] = 0.0
        trim_state[7] = alpha_trim
        trim_state[3:6] = 0.0
        
        DCM_E2B = euler_to_DCM(trim_state[6], trim_state[7], trim_state[8])
        v_rel_body = np.array([target_Va * np.cos(alpha_trim), 0.0, target_Va * np.sin(alpha_trim)])
        wind_body = DCM_E2B @ trim_control[5:8]
        
        trim_state[0:3] = v_rel_body + wind_body
        
        trim_control[1] = elev_trim
        trim_control[3] = throt_trim
        trim_control[4] = throt_trim

        print("Trim Successful!")
        print(f" Alpha:    {np.degrees(alpha_trim):.2f} deg")
        print(f" Elevator:   {np.degrees(elev_trim):.2f} deg")
        print(f" Throttle:   {throt_trim * 100:.1f} %")
        print("="*50)
        return trim_state, trim_control
    else:
        print("Trim Failed!")
        return base_state, base_control

def calculate_turn_trim(target_Va, turn_rate, base_state, base_control, params):
    """
    Finds trim conditions for a steady, coordinated turn at a specific turn rate (rad/s).
    """
    print("="*50)
    print(f"Trim Routine for Va = {target_Va} m/s | Turn Rate = {np.degrees(turn_rate):.2f} deg/s")
    print("="*50)

    def cost_function(trim_vars):
        alpha, phi, aileron, elevator, rudder, throttle = trim_vars

        state = np.copy(base_state)
        control = np.copy(base_control)
        
        # 1. Kinematic constraints for a steady turn
        state[6] = phi    
        state[7] = alpha  
        # psi state[8] doesn't affect aerodynamics, leave as base
        
        theta = alpha # Approximation for level turn
        
        # Angular rates required to maintain the turn
        state[3] = -turn_rate * np.sin(theta)               # p
        state[4] =  turn_rate * np.sin(phi) * np.cos(theta) # q
        state[5] =  turn_rate * np.cos(phi) * np.cos(theta) # r

        # 2. Velocities (Assuming coordinated turn, beta = 0)
        u_rel = target_Va * np.cos(alpha)
        v_rel = 0.0
        w_rel = target_Va * np.sin(alpha)
        v_rel_body = np.array([u_rel, v_rel, w_rel])
        
        DCM_E2B = euler_to_DCM(state[6], state[7], state[8])
        wind_earth = control[5:8]
        wind_body = DCM_E2B @ wind_earth
        
        state[0:3] = v_rel_body + wind_body

        # 3. Apply Controls
        control[0] = aileron
        control[1] = elevator
        control[2] = rudder
        control[3] = throttle
        control[4] = throttle

        # 4. Calculate Dynamics
        state_dot = full_dynamics(0, state, control, params)

        # 5. Full 6DoF Cost Function
        du, dv, dw = state_dot[0:3]
        dp, dq, dr = state_dot[3:6]

        # Heavy weights on rotational accelerations
        cost = (du**2 + dv**2 + dw**2) + 100 * (dp**2 + dq**2 + dr**2)
        return cost

    # Initial Guesses: [alpha, phi, aileron, elevator, rudder, throttle]
    # Safely cap the initial guess for bank angle so it never exceeds +/- 50 degrees
    phi_guess = np.clip(turn_rate * 5, -np.radians(50), np.radians(50))
    initial_guess = [0.05, phi_guess, 0.0, 0.0, 0.0, 0.6] 

    bounds = (
        (-np.radians(5), np.radians(15)),   # alpha
        (-np.radians(60), np.radians(60)),  # phi (up to 60 deg bank)
        (-np.radians(20), np.radians(20)),  # aileron
        (-np.radians(25), np.radians(15)),  # elevator
        (-np.radians(20), np.radians(20)),  # rudder
        (0.0, 1.0)                          # throttle
    )

    res = minimize(
        cost_function, 
        initial_guess, 
        bounds=bounds, 
        method='SLSQP',
        options={'ftol': 1e-9, 'disp': True}
    )

    if res.success:
        alpha_t, phi_t, ail_t, elev_t, rud_t, throt_t = res.x
        
        print("\nTurn Trim Successful!")
        print(f" Alpha:    {np.degrees(alpha_t):.2f} deg | Bank (Phi): {np.degrees(phi_t):.2f} deg")
        print(f" Aileron:  {np.degrees(ail_t):.2f} deg   | Elevator:   {np.degrees(elev_t):.2f} deg")
        print(f" Rudder:   {np.degrees(rud_t):.2f} deg   | Throttle:   {throt_t * 100:.1f} %")
        print("="*50)
        
        # Build the final trimmed state and control arrays
        trim_state = np.copy(base_state)
        trim_control = np.copy(base_control)
        
        # Set final Euler angles
        trim_state[6] = phi_t
        trim_state[7] = alpha_t
        
        theta_t = alpha_t
        
        # Set final angular rates
        trim_state[3] = -turn_rate * np.sin(theta_t)
        trim_state[4] =  turn_rate * np.sin(phi_t) * np.cos(theta_t)
        trim_state[5] =  turn_rate * np.cos(phi_t) * np.cos(theta_t)
        
        # Set final inertial velocities considering wind
        u_rel = target_Va * np.cos(alpha_t)
        v_rel = 0.0
        w_rel = target_Va * np.sin(alpha_t)
        v_rel_body = np.array([u_rel, v_rel, w_rel])
        
        DCM_E2B = euler_to_DCM(trim_state[6], trim_state[7], trim_state[8])
        wind_body = DCM_E2B @ trim_control[5:8]
        
        trim_state[0:3] = v_rel_body + wind_body
        
        # Set final control values
        trim_control[0] = ail_t
        trim_control[1] = elev_t
        trim_control[2] = rud_t
        trim_control[3] = throt_t
        trim_control[4] = throt_t
        
        return trim_state, trim_control
    else:
        print("Turn Trim Failed! The optimizer could not find an equilibrium.")
        print(res.message)
        # Return base arrays if it fails so the simulation can still try to run safely
        return base_state, base_control
    
def calculate_sideslip_trim(target_Va, target_beta, base_state, base_control, params):
    """
    Finds trim conditions for a straight steady flight with a constant sideslip angle (beta).
    Tracks due North with zero altitude loss.
    """
    print("="*50)
    print(f"Trim Routine for Va = {target_Va:.1f} m/s | Beta = {np.degrees(target_beta):.2f} deg")
    print("="*50)

    def cost_function(trim_vars):
        alpha, phi, theta, psi, aileron, elevator, rudder, throttle = trim_vars

        state = np.copy(base_state)
        control = np.copy(base_control)
        
        # 1. Update Kinematics
        state[6] = phi    
        state[7] = theta  
        state[8] = psi
        
        # Angular rates are zero for straight flight
        state[3:6] = 0.0

        # 2. Velocities derived from Va, alpha, and beta
        u_rel = target_Va * np.cos(alpha) * np.cos(target_beta)
        v_rel = target_Va * np.sin(target_beta)
        w_rel = target_Va * np.sin(alpha) * np.cos(target_beta)
        v_rel_body = np.array([u_rel, v_rel, w_rel])
        
        DCM_E2B = euler_to_DCM(state[6], state[7], state[8])
        wind_earth = control[5:8]
        wind_body = DCM_E2B @ wind_earth
        
        # Calculate required inertial body velocities
        v_body_inertial = v_rel_body + wind_body
        state[0:3] = v_body_inertial

        # 3. Calculate Earth-Frame velocities to penalize drift
        R_B2E = np.transpose(DCM_E2B)
        v_earth = R_B2E @ v_body_inertial
        v_N, v_E, v_D = v_earth # North, East, Down

        # 4. Apply Controls
        control[0] = aileron
        control[1] = elevator
        control[2] = rudder
        control[3] = throttle
        control[4] = throttle

        # 5. Calculate Dynamics
        state_dot = full_dynamics(0, state, control, params)

        # 6. Full 8-DoF Cost Function
        du, dv, dw = state_dot[0:3]
        dp, dq, dr = state_dot[3:6]

        # Heavy weights on rotational accelerations
        cost = (du**2 + dv**2 + dw**2) + 100 * (dp**2 + dq**2 + dr**2)
        
        # Heavy weights on lateral and vertical drift
        cost += (v_E**2 * 100)  # Forces the aircraft to track strictly North
        cost += (v_D**2 * 100)  # Forces the aircraft to maintain altitude

        return cost

    # Initial Guesses: [alpha, phi, theta, psi, aileron, elevator, rudder, throttle]
    initial_guess = [
        0.05,            # alpha
        -target_beta,    # phi (guess bank opposite to beta)
        0.05,            # theta (guess similar to alpha)
        -target_beta,    # psi (guess crab angle opposite to beta)
        0.0,             # aileron
        0.0,             # elevator
        target_beta,     # rudder
        0.6              # throttle
    ] 

    bounds = (
        (-np.radians(5), np.radians(15)),   # alpha
        (-np.radians(30), np.radians(30)),  # phi 
        (-np.radians(10), np.radians(20)),  # theta
        (-np.radians(45), np.radians(45)),  # psi (allow wide crab angles)
        (-np.radians(20), np.radians(20)),  # aileron
        (-np.radians(25), np.radians(15)),  # elevator
        (-np.radians(25), np.radians(25)),  # rudder
        (0.0, 1.0)                          # throttle
    )

    res = minimize(
        cost_function, 
        initial_guess, 
        bounds=bounds, 
        method='SLSQP',
        options={'ftol': 1e-9, 'disp': True}
    )

    if res.success:
        alpha_t, phi_t, theta_t, psi_t, ail_t, elev_t, rud_t, throt_t = res.x
        
        print("\nSideslip Trim Successful!")
        print(f" Alpha:    {np.degrees(alpha_t):.2f} deg | Bank (Phi): {np.degrees(phi_t):.2f} deg")
        print(f" Pitch:    {np.degrees(theta_t):.2f} deg | Heading:    {np.degrees(psi_t):.2f} deg")
        print(f" Aileron:  {np.degrees(ail_t):.2f} deg   | Elevator:   {np.degrees(elev_t):.2f} deg")
        print(f" Rudder:   {np.degrees(rud_t):.2f} deg   | Throttle:   {throt_t * 100:.1f} %")
        print("="*50)
        
        # Build the final trimmed state and control arrays
        trim_state = np.copy(base_state)
        trim_control = np.copy(base_control)
        
        trim_state[6] = phi_t
        trim_state[7] = theta_t
        trim_state[8] = psi_t
        trim_state[3:6] = 0.0
        
        u_rel = target_Va * np.cos(alpha_t) * np.cos(target_beta)
        v_rel = target_Va * np.sin(target_beta)
        w_rel = target_Va * np.sin(alpha_t) * np.cos(target_beta)
        v_rel_body = np.array([u_rel, v_rel, w_rel])
        
        DCM_E2B = euler_to_DCM(trim_state[6], trim_state[7], trim_state[8])
        wind_body = DCM_E2B @ trim_control[5:8]
        trim_state[0:3] = v_rel_body + wind_body
        
        trim_control[0] = ail_t
        trim_control[1] = elev_t
        trim_control[2] = rud_t
        trim_control[3] = throt_t
        trim_control[4] = throt_t
        
        return trim_state, trim_control
    else:
        print("Sideslip Trim Failed! The optimizer could not find an equilibrium.")
        print(res.message)
        return base_state, base_control