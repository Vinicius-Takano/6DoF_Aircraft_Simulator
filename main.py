# main.py
import numpy as np
import yaml

from dynamics.equations import build_state_vector, build_control_vector
from dynamics.system import full_dynamics
from control.control_input import apply_control_doublet, apply_control_step, apply_wind_pulse
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
# --- A. Trim Settings ---
TRIM_MODE = "straight"  # Options: "straight", "turn", "sideslip", "none"

TARGET_V = np.linalg.norm(raw_state[0:3]) 
TARGET_ALTITUDE = -raw_state[11]          
TARGET_TURN_RATE = np.radians(0)          
TARGET_BETA = np.radians(0)               

# --- B. Maneuver Input Settings
INPUT_TYPE = "none"  # Options: "step", "doublet", "wind_pulse", "none"
MANEUVER_START = 0 
MANEUVER_DURATION = 2  
CHANNEL = [5, 6, 7]              
MAGNITUDE = [0, -6.978, 0]      
WIND_VECTOR = [0, 0, 0]  

# --- C. State Perturbation Settings (For Stability Analysis) ---
PERTURB_STATE = True

# 2d) Pure Roll: Delta p = 2 deg/s
DELTA_P_DEG = 20    # Roll Rate (p)

# 2e) Spiral: Delta phi = 10 deg
DELTA_PHI_DEG = 0.0 # Bank Angle (phi) - Set to 10.0 for simulation 2e

# --- D. Time Settings ---
SIM_TIME = 5 
t_span = (0, SIM_TIME)
t_eval = np.arange(0, SIM_TIME, 0.005)

# --- E. Plotting Settings ---
REPORT_VARIABLES = ['V_a', 'Alpha', 'Beta', 'Altitude'] # Good variables for lateral modes

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
    print("\n--- Skipping Trim Routine ---")
    state = raw_state
    base_control = raw_control

# ==================================================================
# 3.5 Apply Initial State Perturbations
# ==================================================================
if PERTURB_STATE:
    # Add Delta p (convert to radians)
    if DELTA_P_DEG != 0.0:
        state[3] += np.radians(DELTA_P_DEG)
        
    # Add Delta phi (convert to radians)
    if DELTA_PHI_DEG != 0.0:
        state[6] += np.radians(DELTA_PHI_DEG)

# ------------------------------------------------------------------
# 4. Console Confirmation
# ------------------------------------------------------------------
print("\n" + "="*50)
print(" SIMULATION CONFIGURATION SUMMARY ")
print("="*50)
print(f" Trim Mode : {TRIM_MODE.upper()} at {TARGET_V:.1f} m/s")

if PERTURB_STATE and (DELTA_P_DEG != 0.0 or DELTA_PHI_DEG != 0.0):
    print(f" Perturb   : FREE RESPONSE (Natural Modes)")
    if DELTA_P_DEG != 0.0: print(f"   -> Delta p   = {DELTA_P_DEG} deg/s")
    if DELTA_PHI_DEG != 0.0: print(f"   -> Delta phi = {DELTA_PHI_DEG} deg")
else:
    print(f" Input Type: {INPUT_TYPE.upper()} starting at t={MANEUVER_START}s")
    # ... [Keep your existing input prints here] ...

print(f" Sim Time  : {SIM_TIME} seconds")
print("="*50 + "\n")
print("Executing Integration (RK45)...")

# ------------------------------------------------------------------
# 5. Simulate dynamics
# ------------------------------------------------------------------
def ode(t, current_state):                                          
    if INPUT_TYPE == "step":
        current_control = apply_control_step(
            t, base_control, MANEUVER_START, CHANNEL, MAGNITUDE
        )
    elif INPUT_TYPE == "doublet":
        current_control = apply_control_doublet(
            t, base_control, MANEUVER_START, MANEUVER_DURATION, CHANNEL, MAGNITUDE
        )
    elif INPUT_TYPE == "wind_pulse":
        current_control = apply_wind_pulse(
            t, base_control, MANEUVER_START, MANEUVER_DURATION, WIND_VECTOR
        )
    else:
        current_control = base_control
        
    return full_dynamics(t, current_state, current_control, params)

sol = solve_ivp(
    ode,
    t_span,
    state,
    t_eval=t_eval,
    method="RK45"
)

print("Integration Complete! Preparing plots...")

# ------------------------------------------------------------------
# 6. Plot results
# ------------------------------------------------------------------
control_history = []

for i, t in enumerate(sol.t):
    current_state = sol.y[:, i]
    
    if INPUT_TYPE == "step":
        ctrl = apply_control_step(t, base_control, MANEUVER_START, CHANNEL, MAGNITUDE)
    elif INPUT_TYPE == "doublet":
        ctrl = apply_control_doublet(t, base_control, MANEUVER_START, MANEUVER_DURATION, CHANNEL, MAGNITUDE)
    elif INPUT_TYPE == "wind_pulse":
        ctrl = apply_wind_pulse(t, base_control, MANEUVER_START, MANEUVER_DURATION, WIND_VECTOR)
    else:
        ctrl = base_control
        
    control_history.append(ctrl)

control_history = np.array(control_history)

# Standard generic state plots
plot_states(sol.t, sol.y.T, control_history)
#plot_3d_flight_path(sol.y.T)

# Custom report using the variables defined at the top
plot_report(sol.t, sol.y.T, control_history, params, REPORT_VARIABLES[0], REPORT_VARIABLES[1], REPORT_VARIABLES[2], REPORT_VARIABLES[3])