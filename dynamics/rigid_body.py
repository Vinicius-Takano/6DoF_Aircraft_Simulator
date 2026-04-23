# Ridig_body.py
import numpy as np
import dynamics.equations as eq
import dynamics.kinematics as kin


def eom_translational_velocity(state, control, params):
    # unpack state
    u, v, w = state[0:3]
    p, q, r = state[3:6]
    
    # forces & moments
    X, Y, Z = eq.forces(state, control, params)
    
    # translational dynamics
    du, dv, dw = (1/params["mass"])*np.array([X, Y, Z]) - np.cross(np.array([p, q, r]), np.array([u, v, w]))
    
    return [du, dv, dw]

def eom_rotational_angular_velocity(state, control, params):
    # unpack state
    p, q, r = state[3:6]
    
    # forces & moments
    L, M, N = eq.moments(state, control, params)
    
    I = eq.build_inertia_matrix(params["inertia"])
    
    # rotational dynamics
    dp, dq, dr = np.linalg.inv(I) @ (np.array([L, M, N]) - np.cross(np.array([p, q, r]), I @ np.array([p, q, r])))
    
    return [dp, dq, dr]

def eom_rotational_angular_position(state, control, params):
    # unpack state
    phi, theta, _ = state[6:9]
    p, q, r, = state[3:6]
    
    R = kin.euler_to_kinematic_matrix(phi, theta)
    
    dphi, dtheta, dpsi = R @ np.array([p, q, r])
    
    return [dphi, dtheta, dpsi]

def eom_translational_position(state, control, params):           
    # unpack state
    u, v, w = state[0:3]
    phi, theta, psi = state[6:9]
    
    R = np.transpose(kin.euler_to_DCM(phi, theta, psi)) # Rotation from body to inertial frame (DCM)
    
    dx, dy, dz = R @ np.array([u, v, w]) # V_n, V_e, V_d, where V_d is -h_dot
    
    return [dx, dy, dz]