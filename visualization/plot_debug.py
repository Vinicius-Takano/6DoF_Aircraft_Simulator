# visualization/plot_debug.py

import numpy as np
import matplotlib.pyplot as plt

from dynamics.kinematics import alpha_beta, total_velocity, euler_to_DCM, rotation_matrix
from dynamics.equations import moments

import models.aerodynamics as aero
import models.propulsion as prop

def plot_debug(time, state_history, control, params):

    # =========================
    # 1. Allocate variables
    # =========================
    alpha, beta, V_T = [], [], []
    climb_angle, flight_path_angle, roc = [], [], []
    
    # New variables for Aero Coefficients
    CL_hist, CD_hist, Cm_hist = [], [], []

    F_aero_hist, F_prop_hist, F_grav_hist = [], [], []
    M_b_hist, M_cg_hist, M_prop_hist = [], [], []

    g = params["atmosphere"]["g"]
    m = params["mass"]

    # =========================
    # 2. Extract data
    # =========================
    for i, s in enumerate(state_history):
        
        # Grab the control vector specifically for this time step
        current_control = control[i]

        # --- Kinematics ---
        u, v, w = s[0:3]
        phi, theta, psi = s[6:9]
        
        a, b = alpha_beta(s)
        vt = total_velocity(s)
        
        alpha.append(np.rad2deg(a))
        beta.append(np.rad2deg(b))
        V_T.append(vt)

        # --- Navigation & Performance ---
        # Rotate body velocities to inertial frame to find climb performance
        R_body_to_inertial = np.transpose(euler_to_DCM(phi, theta, psi))
        u_v, v_v, w_v = R_body_to_inertial @ np.array([u, v, w])
        
        current_roc = -w_v
        roc.append(current_roc)
        
        # Climb angle (gamma) and Flight path track (chi)
        gamma = np.arcsin(np.clip(current_roc / vt, -1.0, 1.0)) if vt != 0 else 0
        chi = np.pi/2 - np.arctan2(u_v, v_v)
        
        climb_angle.append(np.rad2deg(gamma))
        flight_path_angle.append(np.rad2deg(chi))

        # --- Aerodynamic Coefficients ---
        # 1. Get body frame force coefficients
        C_Fb = aero.aero_force_coef_Fb(s, current_control, params)
        # 2. Rotate back to wind frame to isolate CL and CD
        R_bw = rotation_matrix(a, b, "body_to_wind")
        C_Fw = R_bw @ C_Fb 
        
        CD_hist.append(-C_Fw[0])
        CL_hist.append(-C_Fw[2])
        
        # 3. Get Pitching Moment (Cm) directly from the moment coefficients
        C_Mb = aero.aero_moment_coef_Mb(s, current_control, params) 
        Cm_hist.append(C_Mb[1]) # C_Mb = [Cr, Cm, Cn]

        # --- Forces ---
        F_aero = aero.aero_force_Fb(s, current_control, params)
        F_prop, M_prop = prop.propulsive_effects(s, current_control, params)
        
        F_grav = np.array([
            -m * g * np.sin(theta),
            m * g * np.sin(phi) * np.cos(theta),
            m * g * np.cos(phi) * np.cos(theta)
        ])
        
        F_aero_hist.append(F_aero)
        F_prop_hist.append(F_prop)
        F_grav_hist.append(F_grav)

        # --- Moments ---
        M_b = aero.aero_moment_Mb(s, current_control, params) 
        M_cg = moments(s, current_control, params)
        
        M_b_hist.append(M_b)
        M_cg_hist.append(M_cg)
        M_prop_hist.append(M_prop)

    # =========================
    # 3. Convert to numpy arrays
    # =========================
    F_aero_hist = np.array(F_aero_hist)
    F_prop_hist = np.array(F_prop_hist)
    F_grav_hist = np.array(F_grav_hist)

    M_b_hist = np.array(M_b_hist)
    M_cg_hist = np.array(M_cg_hist)
    M_prop_hist = np.array(M_prop_hist)
    
    CL_hist = np.array(CL_hist)
    CD_hist = np.array(CD_hist)
    Cm_hist = np.array(Cm_hist)

    # Helper variables for looping over 3x3 grids
    col_labels = ["X-axis", "Y-axis", "Z-axis"]

    # =========================
    # FIGURE 2 — FORCE COMPONENTS
    # =========================
    fig2, axs2 = plt.subplots(3, 3, figsize=(14, 10))
    fig2.suptitle("Figure 2: Force Components", fontsize=14)
    
    forces_data = [F_aero_hist, F_prop_hist, F_grav_hist]
    row_labels_F = ["F_aero", "F_prop", "F_grav"]

    for i in range(3):
        for j in range(3):
            axs2[i, j].plot(time, forces_data[i][:, j])
            axs2[i, j].grid(True)
            if i == 0: axs2[i, j].set_title(col_labels[j])
            if j == 0: axs2[i, j].set_ylabel(f"{row_labels_F[i]} (N)")
            if i == 2: axs2[i, j].set_xlabel("Time (s)")
            
    plt.tight_layout()

    # =========================
    # FIGURE 3 — MOMENT COMPONENTS
    # =========================
    fig3, axs3 = plt.subplots(3, 3, figsize=(14, 10))
    fig3.suptitle("Figure 3: Moment Components", fontsize=14)
    
    moments_data = [M_b_hist, M_cg_hist, M_prop_hist]
    row_labels_M = ["M_b (Aero AC)", "M_cg (Total)", "M_prop"]

    for i in range(3):
        for j in range(3):
            axs3[i, j].plot(time, moments_data[i][:, j])
            axs3[i, j].grid(True)
            if i == 0: axs3[i, j].set_title(col_labels[j])
            if j == 0: axs3[i, j].set_ylabel(f"{row_labels_M[i]} (N·m)")
            if i == 2: axs3[i, j].set_xlabel("Time (s)")

    plt.tight_layout()

    # =========================
    # FIGURE 4 — KINEMATICS & PERFORMANCE
    # =========================
    # Expanded to 3 columns (figsize widened slightly for better proportions)
    fig4, axs4 = plt.subplots(3, 3, figsize=(16, 10))
    fig4.suptitle("Figure 4: Kinematics, Performance & Aero Coefficients", fontsize=14)

    # Column 1: alpha, beta, V_T
    axs4[0, 0].plot(time, alpha); axs4[0, 0].set_ylabel("Alpha (deg)"); axs4[0, 0].grid()
    axs4[1, 0].plot(time, beta);  axs4[1, 0].set_ylabel("Beta (deg)");  axs4[1, 0].grid()
    axs4[2, 0].plot(time, V_T);   axs4[2, 0].set_ylabel("Total Velocity (m/s)"); axs4[2, 0].set_xlabel("Time (s)"); axs4[2, 0].grid()

    # Column 2: Climb angle, Flight path angle, RoC
    axs4[0, 1].plot(time, climb_angle);       axs4[0, 1].set_ylabel("Climb Angle (deg)"); axs4[0, 1].grid()
    axs4[1, 1].plot(time, flight_path_angle); axs4[1, 1].set_ylabel("Flight Path Angle/Track (deg)"); axs4[1, 1].grid()
    axs4[2, 1].plot(time, roc);               axs4[2, 1].set_ylabel("Rate of Climb (m/s)"); axs4[2, 1].set_xlabel("Time (s)"); axs4[2, 1].grid()

    # Column 3: CL, CD, Cm
    axs4[0, 2].plot(time, CL_hist); axs4[0, 2].set_ylabel("C_L"); axs4[0, 2].grid()
    axs4[1, 2].plot(time, CD_hist); axs4[1, 2].set_ylabel("C_D"); axs4[1, 2].grid()
    axs4[2, 2].plot(time, Cm_hist); axs4[2, 2].set_ylabel("C_m"); axs4[2, 2].set_xlabel("Time (s)"); axs4[2, 2].grid()

    plt.tight_layout()

    # Show all plots
    plt.show()