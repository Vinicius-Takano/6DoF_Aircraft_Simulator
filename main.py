import numpy as np
import yaml

from dynamics.equations import build_state_vector, build_control_vector
from dynamics.system import full_dynamics
from control.control_input import apply_control_pulse
from control.trim import calculate_trim

from scipy.integrate import solve_ivp
 
# ------------------------------------------------------------------
# 1. Extract parameters and initial conditions
# ------------------------------------------------------------------
cfg_path = 'config/aircraft.yaml'
with open(cfg_path, 'r') as file:
    data = yaml.safe_load(file)

raw_state = build_state_vector(data["state"])
raw_control = build_control_vector(data["control"])
params = data

# ------------------------------------------------------------------
# 2. Trim the Aircraft
# ------------------------------------------------------------------

if params["trim"]:
    target_V = np.linalg.norm(raw_state[0:3]) 
    altitude = -raw_state[11] # -p_d
    state, base_control = calculate_trim(target_V, altitude, raw_state, raw_control, params)

# ------------------------------------------------------------------
# 3. Define Maneuver Parameters
# ------------------------------------------------------------------
pulse_start = 15
pulse_duration = 1
target_channel = 1    
pulse_magnitude = -5 # degrees

# ------------------------------------------------------------------
# 4. Simulate dynamics
# ------------------------------------------------------------------
def ode(t, current_state):                                                  
    current_control = apply_control_pulse(
        t, base_control, pulse_start, pulse_duration, target_channel, pulse_magnitude
    )
    return full_dynamics(t, current_state, current_control, params)

t_span = (0, 60)
t_eval = np.arange(0, 60, 0.01) 

sol = solve_ivp(
    ode,
    t_span,
    state, # Now using the trimmed state
    t_eval=t_eval,
    method="RK45"
)

# ------------------------------------------------------------------
# 5. Plot results
# ------------------------------------------------------------------
control_history = np.array([
    apply_control_pulse(t, base_control, pulse_start, pulse_duration, target_channel, pulse_magnitude) 
    for t in sol.t
])

from visualization.plot_states import plot_states, plot_3d_flight_path
from visualization.plot_debug import plot_debug

plot_states(sol.t, sol.y.T, control_history)
#plot_3d_flight_path(sol.y.T)                     
#plot_debug(sol.t, sol.y.T, control_history, params) 


