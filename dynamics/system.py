import numpy as np
import dynamics.rigid_body as rb

def full_dynamics(t, state, control, params):
    
    # --- Translational velocity ---
    du, dv, dw = rb.eom_translational_velocity(state, control, params)

    # --- Rotational velocity ---
    dp, dq, dr = rb.eom_rotational_angular_velocity(state, control, params)

    # --- Attitude ---
    dphi, dtheta, dpsi = rb.eom_rotational_angular_position(state, control, params)
    
    # --- Position ---
    dx, dy, dz = rb.eom_translational_position(state, control, params)

    return np.array([
        du, dv, dw,
        dp, dq, dr,
        dphi, dtheta, dpsi,
        dx, dy, dz
    ])