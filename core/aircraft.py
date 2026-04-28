"""
Aircraft Class

Main entry point for aircraft simulation, providing encapsulation of state,
control, and parameters with a OOP interface.
"""

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from typing import Optional, Tuple

from core.state import State
from core.control import Control
from dynamics.equations import build_state_vector, build_control_vector
from dynamics.system import full_dynamics
from control.trim import calculate_trim
from control.control_input import apply_control_pulse


class Aircraft:
    """
    Aircraft simulation class encapsulating state, control, parameters, and simulation logic.

    The Aircraft class provides a object-oriented interface for:
    - Loading and managing configuration
    - State and control vector management
    - Trim condition calculation
    - Simulation execution
    - Result visualization
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Aircraft instance.

        Args:
            config_path: Path to YAML configuration file. If None, creates empty Aircraft.
        """
        self._state: Optional[State] = None
        self._control: Optional[Control] = None
        self._params: dict = {}
        self._trimmed: bool = False
        self._cache: dict = {}

        # Simulation results
        self._time: Optional[np.ndarray] = None
        self._state_history: Optional[np.ndarray] = None
        self._control_history: Optional[np.ndarray] = None

        if config_path:
            self.load_configuration(config_path)

    # --- Configuration ---

    def load_configuration(self, config_path: str):
        """
        Load aircraft configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If required fields are missing
        """
        try:
            with open(config_path, 'r') as file:
                self._params = yaml.safe_load(file)

            # Validate required sections
            required_sections = ['state', 'control', 'mass', 'inertia']
            for section in required_sections:
                if section not in self._params:
                    raise ValueError(f"Missing required section '{section}' in configuration")

            # Initialize state and control from config
            state_array = build_state_vector(self._params["state"])
            control_array = build_control_vector(self._params["control"])

            self._state = State(state_array)
            self._control = Control(control_array)
            self._trimmed = False

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

    @property
    def params(self) -> dict:
        """Get configuration parameters."""
        return self._params

    # --- State Management ---

    @property
    def state(self) -> State:
        """Get current aircraft state."""
        if self._state is None:
            raise ValueError("State not initialized. Call load_configuration() first.")
        return self._state

    @state.setter
    def state(self, value: State):
        """Set aircraft state."""
        if not isinstance(value, State):
            raise TypeError("State must be State instance")
        self._clear_cache()
        self._state = value

    def set_state_from_array(self, state_array: np.ndarray):
        """Set state from a 12-element numpy array."""
        self.state = State(state_array)

    # --- Control Management ---

    @property
    def control(self) -> Control:
        """Get current control settings."""
        if self._control is None:
            raise ValueError("Control not initialized. Call load_configuration() first.")
        return self._control

    @control.setter
    def control(self, value: Control):
        """Set control settings."""
        if not isinstance(value, Control):
            raise TypeError("Control must be Control instance")
        self._clear_cache()
        self._control = value

    def set_control_from_array(self, control_array: np.ndarray):
        """Set control from a 5-element numpy array."""
        self.control = Control(control_array)

    # --- Trim ---

    @property
    def trimmed(self) -> bool:
        """Get trim status."""
        return self._trimmed

    def trim(self, target_velocity: Optional[float] = None, altitude: Optional[float] = None):
        """
        Calculate trim condition for straight and level flight.

        Args:
            target_velocity: Target velocity (m/s). If None, uses current velocity magnitude.
            altitude: Target altitude (m). If None, uses current altitude.

        Returns:
            tuple: (trimmed_state, trimmed_control)
        """
        if not self._params:
            raise ValueError("Configuration not loaded. Call load_configuration() first.")

        if self._params.get("trim", False) is False:
            print("Trim disabled in configuration, skipping.")
            return

        # Use current values if not specified
        if target_velocity is None:
            target_velocity = self.state.total_velocity
        if altitude is None:
            altitude = self.state.altitude

        print(f"--- Starting Trim Routine for V = {target_velocity:.2f} m/s ---")

        # Calculate trim using existing function (will be refactored later)
        trimmed_state_array, trimmed_control_array = calculate_trim(
            target_velocity=target_velocity,
            altitude=altitude,
            base_state=self.state.data,
            base_control=self.control.data,
            params=self.params
        )

        # Update with trimmed values
        self.set_state_from_array(trimmed_state_array)
        self.set_control_from_array(trimmed_control_array)
        self._trimmed = True

        print("Trim Successful!")
        print(f" Alpha: {np.degrees(self.state.theta):.3f} deg")
        print(f" Elevator: {np.degrees(self.control.elevator):.3f} deg")
        print(f" Throttle: {self.control.throttle * 100:.2f} %")
        print("---------------------------------------------------")

        return self.state, self.control

    # --- Simulation ---

    def simulate(self,
                 t_start: float = 0.0,
                 t_end: float = 60.0,
                 dt: float = 0.01,
                 pulse_start: float = 15.0,
                 pulse_duration: float = 10.0,
                 target_channel: int = 0,
                 pulse_magnitude: float = -15.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run flight simulation with a control pulse input.

        Args:
            t_start: Simulation start time (s)
            t_end: Simulation end time (s)
            dt: Time step for evaluation (s)
            pulse_start: When pulse begins (s)
            pulse_duration: Pulse duration (s)
            target_channel: Control channel to pulse (0-4)
            pulse_magnitude: Pulse magnitude (degrees for surfaces, % for throttle)

        Returns:
            tuple: (time_array, state_history) where state_history has shape (time_steps, 12)
        """
        if self._state is None or self._control is None:
            raise ValueError("State and control must be initialized before simulation")

        print(f"\nStarting simulation: t=[{t_start}, {t_end}] s")
        print(f"Control pulse: channel={target_channel}, magnitude={pulse_magnitude}, "
              f"t=[{pulse_start}, {pulse_start + pulse_duration}] s")
        print(f"Initial state: {self.state}\n")

        t_span = (t_start, t_end)
        t_eval = np.arange(t_start, t_end, dt)

        # ODE function that uses current aircraft control
        base_control = self.control.data.copy()

        def ode(t, current_state):
            current_control = apply_control_pulse(
                t, base_control, pulse_start, pulse_duration, target_channel, pulse_magnitude
            )
            return full_dynamics(t, current_state, current_control, self.params)

        # Run simulation
        sol = solve_ivp(
            ode,
            t_span,
            self.state.data,
            t_eval=t_eval,
            method="RK45"
        )

        # Store results
        self._time = sol.t
        self._state_history = sol.y.T

        # Generate control history for analysis
        self._control_history = np.array([
            apply_control_pulse(t, base_control, pulse_start, pulse_duration, target_channel, pulse_magnitude)
            for t in sol.t
        ])

        print(f"Simulation complete. Time steps: {len(sol.t)}")
        return self._time, self._state_history

    # --- Results Analysis ---

    @property
    def time(self) -> Optional[np.ndarray]:
        """Get simulation time array (time_steps,)."""
        return self._time

    @property
    def state_history(self) -> Optional[np.ndarray]:
        """Get state history array (time_steps, 12)."""
        return self._state_history

    @property
    def control_history(self) -> Optional[np.ndarray]:
        """Get control history array (time_steps, 5)."""
        return self._control_history

    def get_state_at_time(self, t: float) -> Optional[State]:
        """
        Get state at specific time using nearest interpolation.

        Args:
            t: Time in seconds

        Returns:
            State instance or None if simulation hasn't run
        """
        if self._time is None or self._state_history is None:
            return None

        # Find nearest time index
        idx = np.argmin(np.abs(self._time - t))
        return State(self._state_history[idx])

    # --- Visualization ---

    def plot_results(self):
        """
        Plot simulation results.

        Raises:
            ValueError: If simulation hasn't been run yet
        """
        if self._time is None or self._state_history is None:
            raise ValueError("No simulation results to plot. Call simulate() first.")

        print("\nGenerating plots...")

        # Import here to avoid import errors if visualization not needed
        from visualization.plot_states import plot_states, plot_3d_flight_path
        # from visualization.plot_debug import plot_debug

        plot_states(self._time, self._state_history, self._control_history)
        plot_3d_flight_path(self._state_history)
        # plot_debug(self._time, self._state_history, self._control_history, self.params)

        print("Plotting complete.")

    # --- Cache Management ---

    def _clear_cache(self):
        """Clear cached computed values."""
        self._cache.clear()

    def __str__(self) -> str:
        """String representation of aircraft."""
        trim_status = "TRIMMED" if self._trimmed else "NOT TRIMMED"
        return (
            f"Aircraft({trim_status})\n"
            f"  State: {self.state}\n"
            f"  Control: {self.control}"
        )

    def __repr__(self) -> str:
        return f"Aircraft(trimmed={self._trimmed})"
