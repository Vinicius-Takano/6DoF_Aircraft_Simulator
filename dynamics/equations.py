# Equations.py
import numpy as np
import models.aerodynamics as aero
import models.propulsion as prop

# Utility functions 
def build_state_vector(state):
    u, v, w = state["velocity"]

    p, q, r = np.deg2rad(state["angular_velocity"])
    phi, theta, psi = np.deg2rad(state["euler_angles"])

    p_n, p_e, p_d = state["position"]

    return np.array([u, v, w, p, q, r, phi, theta, psi, p_n, p_e, p_d])

def build_control_vector(control):
    aileron = control["aileron"] * np.pi / 180
    elevator = control["elevator"] * np.pi / 180
    rudder = control["rudder"] * np.pi / 180
    throttle_1 = control["throttle_1"] / 100
    throttle_2 = control["throttle_2"] / 100

    return np.array([aileron, elevator, rudder, throttle_1, throttle_2])        # convert to radians for control inputs

def build_inertia_matrix(inertia):
    Ixx = inertia["Ixx"]
    Iyy = inertia["Iyy"]
    Izz = inertia["Izz"]
    Ixy = inertia["Ixy"]
    Ixz = inertia["Ixz"]
    Iyz = inertia["Iyz"]

    return np.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])
    
def aicraft_geometry(geometry):
    S_w = geometry["S_w"]
    lht = geometry["lht"]
    c = geometry["c"]
    b = geometry["b"]
    x_eng_1 = geometry["x_eng_1"]
    x_eng_2 = geometry["x_eng_2"]
    y_eng_1 = geometry["y_eng_1"]
    y_eng_2 = geometry["y_eng_2"]
    z_eng_1 = geometry["z_eng_1"]
    z_eng_2 = geometry["z_eng_2"]
    engine_positions = np.array([
        [x_eng_1, x_eng_2],
        [y_eng_1, y_eng_2],
        [z_eng_1, z_eng_2]
    ])
    return S_w, lht, c, b, engine_positions

def forces(state, control, params):
    # Aerodynamic forces
    F_aero = aero.aero_force_Fb(state, control, params)

    # Propulsion forces
    F_prop, _ = prop.propulsive_effects(state, control, params)

    # Gravity (body frame)
    phi, theta, _ = state[6:9]
    g = params["atmosphere"]["g"]
    m = params["mass"]

    F_grav = np.array([
        -m * g * np.sin(theta),
        m * g * np.sin(phi) * np.cos(theta),
        m * g * np.cos(phi) * np.cos(theta)
    ])
    
    return F_aero + F_prop + F_grav


def moments(state, control, params):
    # Aerodynamic moments
    M_aero = aero.aero_moment_cg(state, control, params)

    # Propulsion moments
    _, M_prop = prop.propulsive_effects(state, control, params)

    return M_aero + M_prop
