# visualization/plot_states.py
import matplotlib.pyplot as plt
import numpy as np

def stabilize_axis(ax, min_range=0.1):
    """
    Prevents matplotlib from zooming in on microscopic floating-point noise.
    If the data span is smaller than min_range, pads the y-limits.
    """
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    if span < min_range:
        midpoint = (y_max + y_min) / 2.0
        ax.set_ylim(midpoint - (min_range / 2.0), midpoint + (min_range / 2.0))

def plot_states(time, state_history, control_history):

    # Helper function to enforce the reference image style
    def apply_style(ax, x_data, y_data, xlabel, ylabel, legend_label, line_color='#1F2937'):
        ax.plot(x_data, y_data, color=line_color, linewidth=1.5) 
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, which='both', linestyle='-', color='#D1D5DB')
        ax.ticklabel_format(useOffset=False, style='plain')
        
        # Add the signature legend box in the top right
        ax.legend([legend_label], loc='upper right', frameon=True, 
                  edgecolor='black', facecolor='white', framealpha=1.0)

    # =========================
    # FIGURE 1: State Variables
    # =========================
    fig, axs = plt.subplots(4, 3, figsize=(16, 12)) 

    # --- Row 1: Velocities (u, v, w) ---
    labels = ["u", "v", "w"]
    for i in range(3):
        apply_style(axs[0, i], time, state_history[:, i], "Time - s", "Velocity - m/s", labels[i])
        stabilize_axis(axs[0, i], min_range=1.0)

    # --- Row 2: Angular Velocities (p, q, r) ---
    labels = ["p", "q", "r"]
    for i in range(3):
        apply_style(axs[1, i], time, np.degrees(state_history[:, i+3]), "Time - s", "Rate - deg/s", labels[i])
        stabilize_axis(axs[1, i], min_range=2.0)

    # --- Row 3: Euler Angles (phi, theta, psi) ---
    labels = [r"$\phi$", r"$\theta$", r"$\psi$"] 
    for i in range(3):
        apply_style(axs[2, i], time, np.degrees(state_history[:, i+6]), "Time - s", "Angle - deg", labels[i])
        stabilize_axis(axs[2, i], min_range=2.0)

    # --- Row 4: Alpha, Y vs X, Z vs X ---
    # 1. Alpha (Angle of Attack)
    u_vel = state_history[:, 0]
    w_vel = state_history[:, 2]
    alpha_deg = np.degrees(np.arctan2(w_vel, u_vel))
    apply_style(axs[3, 0], time, alpha_deg, "Time - s", "Angle - deg", "Alpha")
    stabilize_axis(axs[3, 0], min_range=2.0)

    # 2. Y vs X (East vs North)
    p_n = state_history[:, 9]  
    p_e = state_history[:, 10] 
    apply_style(axs[3, 1], p_n, p_e, "Distance - m", "Distance - m", "Y vs X")

    # 3. Z vs X (Altitude vs North)
    altitude = -state_history[:, 11] 
    apply_style(axs[3, 2], p_n, altitude, "Distance - m", "Height - m", "Z vs X")

    fig.tight_layout(pad=2.0)

    # =========================
    # FIGURE 2: Control Inputs
    # =========================
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle("Control Inputs", fontsize=14, fontweight='bold')
    
    # Create a 3-row, 6-column GridSpec
    gs = fig2.add_gridspec(3, 6)
    
    # Map axes to the grid (span 2 columns per plot)
    axs2_r1 = [fig2.add_subplot(gs[0, 0:2]), fig2.add_subplot(gs[0, 2:4]), fig2.add_subplot(gs[0, 4:6])]
    axs2_r2 = [fig2.add_subplot(gs[1, 1:3]), fig2.add_subplot(gs[1, 3:5])] # Centered!
    axs2_r3 = [fig2.add_subplot(gs[2, 0:2]), fig2.add_subplot(gs[2, 2:4]), fig2.add_subplot(gs[2, 4:6])]

    ctrl_color = '#B22222'

    # --- Row 1: Aero Surfaces (Aileron, Elevator, Rudder) ---
    labels_r1 = ["Aileron", "Elevator", "Rudder"]
    for i, ax in enumerate(axs2_r1):
        data = np.degrees(control_history[:, i])
        apply_style(ax, time, data, "Time - s", "Angle - deg", labels_r1[i], line_color=ctrl_color)
        stabilize_axis(ax, min_range=2.0)

    # --- Row 2: Throttles (Thrust 1, Thrust 2) ---
    labels_r2 = ["Throttle 1", "Throttle 2"]
    for i, ax in enumerate(axs2_r2):
        data = control_history[:, i+3]
        apply_style(ax, time, data, "Time - s", "Throttle - %", labels_r2[i], line_color=ctrl_color)
        stabilize_axis(ax, min_range=1.0)

    # --- Row 3: Wind (North, East, Down) ---
    labels_r3 = ["Wind North", "Wind East", "Wind Down"]
    for i, ax in enumerate(axs2_r3):
        data = control_history[:, i+5]
        apply_style(ax, time, data, "Time - s", "Velocity - m/s", labels_r3[i], line_color=ctrl_color)
        stabilize_axis(ax, min_range=2.0)
        
    fig2.tight_layout(pad=2.0)
    
    
def plot_3d_flight_path(state_history):
    """
    Plots a 3D trajectory of the aircraft's flight path.
    """
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')

    p_n = state_history[:, 9]   
    p_e = state_history[:, 10]  
    p_d = state_history[:, 11]  
    altitude = -p_d

    ax3.plot(p_e, p_n, altitude, color='#004C99', linewidth=2.5, label='Trajectory')

    ax3.scatter(p_e[0], p_n[0], altitude[0], color='green', s=100, label='Start', zorder=5)
    ax3.scatter(p_e[-1], p_n[-1], altitude[-1], color='red', s=100, label='End', zorder=5)

    ax3.set_xlabel('East (m)')
    ax3.set_ylabel('North (m)')
    ax3.set_zlabel('Altitude (m)')
    
    x_min, x_max = ax3.get_xlim3d()
    y_min, y_max = ax3.get_ylim3d()
    z_min, z_max = ax3.get_zlim3d()
    
    if (x_max - x_min) < 10.0:
        mid = (x_max + x_min) / 2.0
        ax3.set_xlim3d(mid - 5.0, mid + 5.0)
    if (y_max - y_min) < 10.0:
        mid = (y_max + y_min) / 2.0
        ax3.set_ylim3d(mid - 5.0, mid + 5.0)
    if (z_max - z_min) < 10.0:
        mid = (z_max + z_min) / 2.0
        ax3.set_zlim3d(mid - 5.0, mid + 5.0)

    ax3.legend()
    fig3.tight_layout()
    
    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()
    except AttributeError:
        pass 
    
    plt.show()