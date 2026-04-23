# visualization/plot_states.py
import matplotlib.pyplot as plt

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
    axs[0, 0].set_ylabel("Velocity (m/s)")

    # --- Row 2: Angular velocities ---
    labels = ["p", "q", "r"]
    for i in range(3):
        axs[1, i].plot(time, state_history[:, i+3])
        axs[1, i].set_title(labels[i])
        axs[1, i].ticklabel_format(useOffset=False) 
        axs[1, i].grid(True)
    axs[1, 0].set_ylabel("Angular Velocity (rad/s)")

    # --- Row 3: Euler angles ---
    labels = ["phi", "theta", "psi"]
    for i in range(3):
        axs[2, i].plot(time, state_history[:, i+6])
        axs[2, i].set_title(labels[i])
        axs[2, i].ticklabel_format(useOffset=False)
        axs[2, i].grid(True)
    axs[2, 0].set_ylabel("Euler Angles (rad)")

    # --- Row 4: NED Positions ---
    labels = ["p_n (North)", "p_e (East)", "p_d (Down)"]
    for i in range(3):
        axs[3, i].plot(time, state_history[:, i+9])
        axs[3, i].set_title(labels[i])
        axs[3, i].ticklabel_format(useOffset=False)
        axs[3, i].grid(True)
        axs[3, i].set_xlabel("Time (s)")
    axs[3, 0].set_ylabel("Position (m)")

    fig.tight_layout()

    # =========================
    # FIGURE 2: Control Inputs
    # =========================
    fig2, axs2 = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig2.suptitle("Control Inputs", fontsize=14)
    
    control_labels = [
        "Aileron (rad)", 
        "Elevator (rad)", 
        "Rudder (rad)", 
        "Throttle 1 (%)", 
        "Throttle 2 (%)"
    ]

    for i in range(5):
        axs2[i].plot(time, control_history[:, i], color='red')
        axs2[i].set_ylabel(control_labels[i])
        axs2[i].grid(True)
        
    axs2[-1].set_xlabel("Time (s)")
    
    fig2.tight_layout()
    plt.show()
    
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
    
    ax3.legend()
    fig3.tight_layout()
    
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    
    plt.show()
