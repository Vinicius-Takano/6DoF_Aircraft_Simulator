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

    # =========================
    # FIGURE 1: State Variables
    # =========================
    fig, axs = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle("Aircraft State Variables", fontsize=14)

    # --- Row 1: Velocities ---
    labels = ["u", "v", "w"]
    for i in range(3):
        axs[0, i].plot(time, state_history[:, i])
        axs[0, i].set_title(labels[i])
        axs[0, i].ticklabel_format(useOffset=False)
        axs[0, i].grid(True)
        stabilize_axis(axs[0, i], min_range=1.0) # Minimum 1 m/s span
    axs[0, 0].set_ylabel("Velocity (m/s)")

    # --- Row 2: Angular velocities ---
    labels = ["p", "q", "r"]
    for i in range(3):
        axs[1, i].plot(time, state_history[:, i+3])
        axs[1, i].set_title(labels[i])
        axs[1, i].ticklabel_format(useOffset=False) 
        axs[1, i].grid(True)
        stabilize_axis(axs[1, i], min_range=0.05) # Minimum 0.05 rad/s span
    axs[1, 0].set_ylabel("Angular Velocity (rad/s)")

    # --- Row 3: Euler angles ---
    labels = ["phi", "theta", "psi"]
    for i in range(3):
        axs[2, i].plot(time, state_history[:, i+6])
        axs[2, i].set_title(labels[i])
        axs[2, i].ticklabel_format(useOffset=False)
        axs[2, i].grid(True)
        stabilize_axis(axs[2, i], min_range=0.05) # Minimum 0.05 rad span
    axs[2, 0].set_ylabel("Euler Angles (rad)")

    # --- Row 4: NED Positions ---
    labels = ["p_n (North)", "p_e (East)", "p_d (Down)"]
    for i in range(3):
        axs[3, i].plot(time, state_history[:, i+9])
        axs[3, i].set_title(labels[i])
        axs[3, i].ticklabel_format(useOffset=False)
        axs[3, i].grid(True)
        axs[3, i].set_xlabel("Time (s)")
        stabilize_axis(axs[3, i], min_range=2.0) # Minimum 2 m span
    axs[3, 0].set_ylabel("Position (m)")

    fig.tight_layout()

    # =========================
    # FIGURE 2: Control Inputs
    # =========================
    fig2, axs2 = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
    fig2.suptitle("Control Inputs", fontsize=14)
    
    control_labels = [
        "Aileron (rad)", 
        "Elevator (rad)", 
        "Rudder (rad)", 
        "Throttle 1 (%)", 
        "Throttle 2 (%)",
        "Wind X (m/s)",
        "Wind Y (m/s)",
        "Wind Z (m/s)"
    ]

    # Define appropriate minimum ranges for each control channel
    control_min_ranges = [0.05, 0.05, 0.05, 1, 1, 2.0, 2.0, 2.0]

    for i in range(8):
        axs2[i].plot(time, control_history[:, i], color='red')
        axs2[i].set_ylabel(control_labels[i])
        axs2[i].grid(True)
        stabilize_axis(axs2[i], min_range=control_min_ranges[i])
        
    axs2[-1].set_xlabel("Time (s)")
    
    fig2.tight_layout()
    
    
def plot_3d_flight_path(state_history):
    """
    Plots a 3D trajectory of the aircraft's flight path.
    """
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    fig3.suptitle("3D Flight Path", fontsize=14)

    # Extract positions from the state history
    p_n = state_history[:, 9]   # North
    p_e = state_history[:, 10]  # East
    p_d = state_history[:, 11]  # Down

    altitude = -p_d

    # Plot the continuous flight path
    ax3.plot(p_e, p_n, altitude, color='blue', linewidth=2.5, label='Trajectory')

    # Add markers for the Start and End points to indicate flight direction
    ax3.scatter(p_e[0], p_n[0], altitude[0], color='green', s=100, label='Start', zorder=5)
    ax3.scatter(p_e[-1], p_n[-1], altitude[-1], color='red', s=100, label='End', zorder=5)

    ax3.set_xlabel('East (m)')
    ax3.set_ylabel('North (m)')
    ax3.set_zlabel('Altitude (m)')
    
    # Optional: Stabilize 3D axes to prevent extreme zooming on straight flights
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
        pass # Silently pass if the backend doesn't support full screen
    
    plt.show()