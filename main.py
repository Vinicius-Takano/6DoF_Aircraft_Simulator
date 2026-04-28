"""
Main Aircraft Simulation Script
"""

import numpy as np
from core.aircraft import Aircraft

# ------------------------------------------------------------------
# 1. Create and Configure Aircraft
# ------------------------------------------------------------------

cfg_path = 'config/aircraft.yaml'
aircraft = Aircraft(cfg_path)

print("Aircraft Configuration:")
print(aircraft)
print()

# ------------------------------------------------------------------
# 2. Trim the Aircraft (if enabled in config)
# ------------------------------------------------------------------

if aircraft.params.get("trim", False):
    target_V = aircraft.state.total_velocity
    altitude = aircraft.state.altitude
    aircraft.trim(target_V, altitude)
else:
    print("Trim disabled in configuration, skipping.")

print()

# ------------------------------------------------------------------
# 3. Simulate Dynamics with Control Pulse
# ------------------------------------------------------------------

# Define maneuver parameters
maneuver_params = {
    'pulse_start': 10,  # seconds
    'pulse_duration': 1,  # seconds
    'target_channel': 1, # 0=aileron, 1=elevator, 2=rudder, 3/4=throttle
    'pulse_magnitude': -5,  # degrees for surfaces, % for throttle
    't_start': 0.0,
    't_end': 100,
    'dt': 0.05
}

# Run simulation
time, state_history = aircraft.simulate(
    t_start=maneuver_params['t_start'],
    t_end=maneuver_params['t_end'],
    dt=maneuver_params['dt'],
    pulse_start=maneuver_params['pulse_start'],
    pulse_duration=maneuver_params['pulse_duration'],
    target_channel=maneuver_params['target_channel'],
    pulse_magnitude=maneuver_params['pulse_magnitude']
)

print()

# ------------------------------------------------------------------
# 4. Plot Results
# ------------------------------------------------------------------

aircraft.plot_results()

print()
print("Simulation complete!")
