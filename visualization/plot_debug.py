# visualization/plot_debug.py

import numpy as np
import matplotlib.pyplot as plt

from dynamics.kinematics import alpha_beta, total_velocity, euler_to_DCM, rotation_matrix
from dynamics.equations import moments

import models.aerodynamics as aero
import models.propulsion as prop
from visualization.plot_states import stabilize_axis # Reusing the stabilizer we built earlier

def extract_simulation_data(time, state_history, control_history, params):
    """
    Processes the raw state and control arrays into a dictionary of 
    ready-to-plot variables, accounting for wind kinematics.
    """
    g = params["atmosphere"]["g"]
    m = params["mass"]

    data = {
        # Raw States
        "u": state_history[:, 0], "v": state_history[:, 1], "w": state_history[:, 2],
        "p": np.rad2deg(state_history[:, 3]), "q": np.rad2deg(state_history[:, 4]), "r": np.rad2deg(state_history[:, 5]),
        "Phi": np.rad2deg(state_history[:, 6]), "Theta": np.rad2deg(state_history[:, 7]), "Psi": np.rad2deg(state_history[:, 8]),
        "North": state_history[:, 9], "East": state_history[:, 10], "Altitude": -state_history[:, 11],
        
        # Raw Controls
        "Aileron": np.rad2deg(control_history[:, 0]),
        "Elevator": np.rad2deg(control_history[:, 1]),
        "Rudder": np.rad2deg(control_history[:, 2]),
        "Throttle": control_history[:, 3] * 100,
        "Wind N": control_history[:, 5],
        "Wind E": control_history[:, 6],
        "Wind D": control_history[:, 7],

        # Derived Variables (Pre-allocated lists)
        "Alpha": [], "Beta": [], "V_a": [], "V_g": [],
        "RoC": [], "Climb Angle": [], "Flight Path Angle": [],
        "CL": [], "CD": [], "Cm": []
    }

    # Pre-allocate 3D force/moment arrays
    F_aero_hist, F_prop_hist, F_grav_hist = [], [], []
    M_b_hist, M_cg_hist, M_prop_hist = [], [], []

    for i, s in enumerate(state_history):
        current_control = control_history[i]

        u, v, w = s[0:3]
        phi, theta, psi = s[6:9]
        
        # --- Kinematics (Updated for Wind) ---
        a, b = alpha_beta(s, current_control)
        va = total_velocity(s, current_control)
        
        data["Alpha"].append(np.rad2deg(a))
        data["Beta"].append(np.rad2deg(b))
        data["V_a"].append(va)

        # --- Navigation & Performance ---
        R_body_to_inertial = np.transpose(euler_to_DCM(phi, theta, psi))
        u_v, v_v, w_v = R_body_to_inertial @ np.array([u, v, w])
        
        # Inertial Ground speed
        vg = np.sqrt(u_v**2 + v_v**2 + w_v**2)
        data["V_g"].append(vg)
        
        current_roc = -w_v
        data["RoC"].append(current_roc)
        
        # Climb angle uses Ground Speed (vg), not Airspeed (va)
        gamma = np.arcsin(np.clip(current_roc / vg, -1.0, 1.0)) if vg != 0 else 0
        chi = np.pi/2 - np.arctan2(u_v, v_v)
        
        data["Climb Angle"].append(np.rad2deg(gamma))
        data["Flight Path Angle"].append(np.rad2deg(chi))

        # --- Aerodynamic Coefficients ---
        C_Fb = aero.aero_force_coef_Fb(s, current_control, params)
        R_bw = rotation_matrix(a, b, "body_to_wind")
        C_Fw = R_bw @ C_Fb 
        
        data["CD"].append(-C_Fw[0])
        data["CL"].append(-C_Fw[2])
        
        C_Mb = aero.aero_moment_coef_Mb(s, current_control, params) 
        data["Cm"].append(C_Mb[1]) 

        # --- Forces & Moments ---
        F_aero_hist.append(aero.aero_force_Fb(s, current_control, params))
        f_p, m_p = prop.propulsive_effects(s, current_control, params)
        F_prop_hist.append(f_p)
        M_prop_hist.append(m_p)
        
        F_grav_hist.append(np.array([
            -m * g * np.sin(theta),
            m * g * np.sin(phi) * np.cos(theta),
            m * g * np.cos(phi) * np.cos(theta)
        ]))

        M_b_hist.append(aero.aero_moment_Mb(s, current_control, params))
        M_cg_hist.append(moments(s, current_control, params))

    # Convert derived lists to arrays
    for key in ["Alpha", "Beta", "V_a", "V_g", "RoC", "Climb Angle", "Flight Path Angle", "CL", "CD", "Cm"]:
        data[key] = np.array(data[key])
        
    data["F_aero"] = np.array(F_aero_hist)
    data["F_prop"] = np.array(F_prop_hist)
    data["F_grav"] = np.array(F_grav_hist)
    data["M_b"] = np.array(M_b_hist)
    data["M_cg"] = np.array(M_cg_hist)
    data["M_prop"] = np.array(M_prop_hist)

    return data


def plot_report(time, state_history, control_history, params, var1, var2, var3, var4):
    """
    Plots a custom 2x2 grid of simulation variables, formatted for formal reports.
    """
    data = extract_simulation_data(time, state_history, control_history, params)
    
    # figsize=(10, 8) or (12, 9) ensures the 2x2 grid yields clean, uniform aspect ratios
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    variables = [var1, var2, var3, var4]
    axs = axs.flatten()
    
    for i, var in enumerate(variables):
        if var in data:
            # 1. 2pt thickness and a professional "simulation blue"
            axs[i].plot(time, data[var], linewidth=1.5, color="#0990FF") 
            
            # 2. Subplot titles only
            axs[i].set_title(var, fontsize=12, fontweight='bold')
            axs[i].set_xlabel("Time (s)", fontsize=10)
            
            # 3. Disable scientific notation and offsets (e.g., stops 1e2 + 5.03)
            axs[i].ticklabel_format(useOffset=False, style='plain', axis='both')
            
            # 4. Standard engineering grids (Major and Minor)
            axs[i].minorticks_on()
            axs[i].grid(which='major', color='#CCCCCC', linewidth=0.8, linestyle='-')
            axs[i].grid(which='minor', color='#E5E5E5', linewidth=0.5, linestyle=':')
            
            # Set a clean spine (border) thickness
            for spine in axs[i].spines.values():
                spine.set_linewidth(1.2)
            
            # Keep your stabilization logic
            stabilize_axis(axs[i], min_range=0.1) 
            
        else:
            axs[i].set_title(f"{var} (NOT FOUND)", color='red')
            print(f"Warning: '{var}' is not a valid variable key.")
            
    # Pad adds a little breathing room between the graphs for the axis labels
    plt.tight_layout(pad=2.0)
    plt.show()


def plot_debug(time, state_history, control_history, params):
    """
    Plots the full detailed aerodynamic and performance breakdown.
    """
    data = extract_simulation_data(time, state_history, control_history, params)
    
    col_labels = ["X-axis", "Y-axis", "Z-axis"]

    # =========================
    # FIGURE 2 — FORCE COMPONENTS
    # =========================
    fig2, axs2 = plt.subplots(3, 3, figsize=(14, 10))
    fig2.suptitle("Figure 2: Force Components", fontsize=14)
    
    forces_data = [data["F_aero"], data["F_prop"], data["F_grav"]]
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
    
    moments_data = [data["M_b"], data["M_cg"], data["M_prop"]]
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
    fig4, axs4 = plt.subplots(3, 3, figsize=(16, 10))
    fig4.suptitle("Figure 4: Kinematics, Performance & Aero Coefficients", fontsize=14)

    # Column 1
    axs4[0, 0].plot(time, data["Alpha"]); axs4[0, 0].set_ylabel("Alpha (deg)"); axs4[0, 0].grid()
    axs4[1, 0].plot(time, data["Beta"]);  axs4[1, 0].set_ylabel("Beta (deg)");  axs4[1, 0].grid()
    axs4[2, 0].plot(time, data["V_a"]);   axs4[2, 0].set_ylabel("Airspeed (m/s)"); axs4[2, 0].set_xlabel("Time (s)"); axs4[2, 0].grid()

    # Column 2
    axs4[0, 1].plot(time, data["Climb Angle"]);       axs4[0, 1].set_ylabel("Climb Angle (deg)"); axs4[0, 1].grid()
    axs4[1, 1].plot(time, data["Flight Path Angle"]); axs4[1, 1].set_ylabel("Track (deg)"); axs4[1, 1].grid()
    axs4[2, 1].plot(time, data["RoC"]);               axs4[2, 1].set_ylabel("Rate of Climb (m/s)"); axs4[2, 1].set_xlabel("Time (s)"); axs4[2, 1].grid()

    # Column 3
    axs4[0, 2].plot(time, data["CL"]); axs4[0, 2].set_ylabel("C_L"); axs4[0, 2].grid()
    axs4[1, 2].plot(time, data["CD"]); axs4[1, 2].set_ylabel("C_D"); axs4[1, 2].grid()
    axs4[2, 2].plot(time, data["Cm"]); axs4[2, 2].set_ylabel("C_m"); axs4[2, 2].set_xlabel("Time (s)"); axs4[2, 2].grid()

    plt.tight_layout()
    plt.show()