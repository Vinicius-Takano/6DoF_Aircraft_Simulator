# main.py
import numpy as np
import yaml

from dynamics.equations import build_state_vector, build_control_vector
from dynamics.system import full_dynamics
from control.control_input import apply_control_doublet, apply_control_pulse, apply_alpha_beta_gust
from control.trim import calculate_trim, calculate_turn_trim, calculate_sideslip_trim
from scipy.integrate import solve_ivp
from visualization.plot_states import plot_states, plot_3d_flight_path
from visualization.plot_debug import plot_debug, plot_report
 
# ------------------------------------------------------------------
# 1. Extract parameters and initial conditions
# ------------------------------------------------------------------
cfg_path = 'config/aircraft.yaml'
with open(cfg_path, 'r') as file:
    data = yaml.safe_load(file)

raw_state = build_state_vector(data["state"])
raw_control = build_control_vector(data["control"])
params = data

# ==================================================================
# 2. SIMULATION CONFIGURATION
# ==================================================================
# Trim Modes: "straight", "turn", "sideslip", or "none"
TRIM_MODE = "turn" 

# Target parameters
TARGET_V = np.linalg.norm(raw_state[0:3]) 
TARGET_ALTITUDE = -raw_state[11]          # Used for straight trim
TARGET_TURN_RATE = np.radians(1)          # Used for turn trim (rad/s)
TARGET_BETA = np.radians(0)               # Used for sideslip trim (rad)

INPUT_TYPE = "none"   # Options: "pulse", "doublet", "gust", "none"
MANEUVER_START = 5.0
MANEUVER_DURATION = 1  # For doublet, this is the length of ONE half
CHANNEL = 1              # 1 for Elevator
MAGNITUDE = -1         # Degrees

# Gust parameters
GUST_DELTA_ALPHA = 2.0   # +2 degrees of Alpha
GUST_DELTA_BETA = 0.0

# ------------------------------------------------------------------
# 3. Trim the Aircraft
# ------------------------------------------------------------------
if TRIM_MODE == "straight":
    state, base_control = calculate_trim(TARGET_V, TARGET_ALTITUDE, raw_state, raw_control, params)
elif TRIM_MODE == "turn":
    state, base_control = calculate_turn_trim(TARGET_V, TARGET_TURN_RATE, raw_state, raw_control, params)
elif TRIM_MODE == "sideslip":
    state, base_control = calculate_sideslip_trim(TARGET_V, TARGET_BETA, raw_state, raw_control, params)
else:
    print("--- Skipping Trim Routine ---")
    state = raw_state
    base_control = raw_control

# ------------------------------------------------------------------
# 4. Simulate dynamics
# ------------------------------------------------------------------
def ode(t, current_state):                                          
    if INPUT_TYPE == "pulse":
        current_control = apply_control_pulse(
            t, base_control, MANEUVER_START, MANEUVER_DURATION, CHANNEL, MAGNITUDE
        )
    elif INPUT_TYPE == "doublet":
        current_control = apply_control_doublet(
            t, base_control, MANEUVER_START, MANEUVER_DURATION, CHANNEL, MAGNITUDE
        )
    elif INPUT_TYPE == "gust":
        current_control = apply_alpha_beta_gust(
            t, current_state, base_control, MANEUVER_START, MANEUVER_DURATION, GUST_DELTA_ALPHA, GUST_DELTA_BETA
        )
    else:
        current_control = base_control
        
    return full_dynamics(t, current_state, current_control, params)

t_span = (0, 10)
t_eval = np.arange(0, 10, 0.01) 

sol = solve_ivp(
    ode,
    t_span,
    state,
    t_eval=t_eval,
    method="RK45"
)

# ------------------------------------------------------------------
# 5. Plot results
# ------------------------------------------------------------------

# 1. Reconstruct the control history exactly as the solver saw it
control_history = []

for i, t in enumerate(sol.t):
    current_state = sol.y[:, i] # Extract state at this specific time step
    
    if INPUT_TYPE == "pulse":
        ctrl = apply_control_pulse(t, base_control, MANEUVER_START, MANEUVER_DURATION, CHANNEL, MAGNITUDE)
    elif INPUT_TYPE == "doublet":
        ctrl = apply_control_doublet(t, base_control, MANEUVER_START, MANEUVER_DURATION, CHANNEL, MAGNITUDE)
    elif INPUT_TYPE == "gust":
        ctrl = apply_alpha_beta_gust(t, current_state, base_control, MANEUVER_START, MANEUVER_DURATION, GUST_DELTA_ALPHA, GUST_DELTA_BETA)
    else:
        ctrl = base_control
        
    control_history.append(ctrl)

control_history = np.array(control_history)

# 2. Generate Plots

# Standard generic state plots
plot_states(sol.t, sol.y.T, control_history)
plot_3d_flight_path(sol.y.T)

# Custom dynamic report
plot_report(sol.t, sol.y.T, control_history, params, 'Alpha', 'Beta', 'V_a', 'V_g')