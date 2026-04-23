# Kinematics.py
import numpy as np
import dynamics.equations as eq
import models.atmosphere as atm

def alpha_beta(state):
    u, v, w = state[0:3]
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / np.sqrt(u**2 + v**2 + w**2)) if np.sqrt(u**2 + v**2 + w**2) != 0 else 0
    return alpha, beta

def total_velocity(state):
    u, v, w = state[0:3]
    V_a = np.sqrt(u**2 + v**2 + w**2)
    return V_a

def dynamic_pressure(state, params):
    V_a = total_velocity(state)
    rho, _ = atm.compute_atmospheric_properties(state)
    q_inf = 0.5 * rho * V_a**2
    return q_inf

def rotation_matrix(alpha, beta, transformation=["body_to_stability", "stability_to_wind", "body_to_wind"]):
    # Rotation matrices for angle of attack (alpha) and sideslip (beta)
    if transformation == "body_to_wind":
        R = np.array([
            [np.cos(alpha)*np.cos(beta), np.sin(beta), np.sin(alpha)*np.cos(beta)],
            [-np.cos(alpha)*np.sin(beta), np.cos(beta), -np.sin(alpha)*np.sin(beta)],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
    elif transformation == "body_to_stability":
        R = np.array([
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
    elif transformation == "stability_to_wind":
        R = np.array([
            [np.cos(beta), np.sin(beta), 0],
            [-np.sin(beta), np.cos(beta), 0],
            [0, 0, 1]
        ])
    return R

def euler_to_kinematic_matrix(phi, theta):
    # Rotation matrix from body to inertial frame (H(euler angles)) from pg.34 Stevens & Lewis
    R = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])
    return R

def euler_to_DCM(phi, theta, psi):
    # Rotation matrix from inertial to body frame (DCM)
    R = np.array([
        [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
        [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
        [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)]
    ])
    return R