"""
Unit tests for Aircraft class
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.aircraft import Aircraft
from core.state import State
from core.control import Control


class TestAircraft(unittest.TestCase):
    """Test cases for Aircraft class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "state": {
                "velocity": [100, 0, 0],
                "angular_velocity": [0, 0, 0],
                "euler_angles": [0, 0, 0],
                "position": [0, 0, -1000]
            },
            "trim": False,
            "control": {
                "aileron": 0,
                "elevator": 0,
                "rudder": 0,
                "throttle_1": 30,
                "throttle_2": 30
            },
            "mass": 120000,
            "inertia": {
                "Ixx": 5550000, "Iyy": 9720000, "Izz": 14510000,
                "Ixy": 0, "Ixz": -33000, "Iyz": 0
            },
            "geometry": {
                "S_w": 260, "c": 6.6, "b": 44,
                "x_eng_1": 0, "x_eng_2": 0,
                "y_eng_1": 8, "y_eng_2": -8,
                "z_eng_1": 0, "z_eng_2": 0,
                "engine_i": 1, "lht": 25
            },
            "r_cg": [0.30, 0, 0],
            "r_ac": [0.25, 0, 0],
            "propulsion": {"T_max": 120000},
            "aerodynamics": {
                "CL_0": 0.5, "CL_alpha": 4.982, "CL_delta_e": 0.435, "CL_q": -0.7,
                "CD0": 0.0175, "k": 0.06,
                "Cm_0": -0.025, "Cm_q": -15, "Cm_delta_e": -1.46, "Cm_alpha": -1.246,
                "Cy_beta": -1.5, "Cy_delta_a": 0.05, "Cy_delta_r": 0.3,
                "Cr_beta": -1.3, "Cr_e": -13, "Cr_r": 2.9,
                "Cr_delta_a": -0.33, "Cr_delta_r": 0.25,
                "Cn_beta": 1.75, "Cn_e": -1.5, "Cn_r": -7.5,
                "Cn_delta_a": -0.125, "Cn_delta_r": -1.0
            }
            "atmosphere": {"rho": 1.225, "g": 9.81},
            "aerodynamics": {
                "CL_0": 0.5, "CL_alpha": 4.982, "CL_delta_e": 0.435, "CL_q": -0.7,
                "CD0": 0.0175, "k": 0.06,
                "Cm_0": -0.025, "Cm_q": -15, "Cm_delta_e": -1.46, "Cm_alpha": -1.246,
                "Cy_beta": -1.5, "Cy_delta_a": 0.05, "Cy_delta_r": 0.3,
                "Cr_beta": -1.3, "Cr_e": -13, "Cr_r": 2.9,
                "Cr_delta_a": -0.33, "Cr_delta_r": 0.25,
                "Cn_beta": 1.75, "Cn_e": -1.5, "Cn_r": -7.5,
                "Cn_delta_a": -0.125, "Cn_delta_r": -1.0
            }
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            self.config_path = f.name

        self.aircraft = Aircraft(self.config_path)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.config_path and os.path.exists(self.config_path):
            os.remove(self.config_path)

    # --- Initialization ---

    def test_aircraft_initialization(self):
        """Test Aircraft initialization with config file."""
        self.assertIsNotNone(self.aircraft)
        self.assertIsInstance(self.aircraft.state, State)
        self.assertIsInstance(self.aircraft.control, Control)
        self.assertEqual(self.aircraft.trimmed, False)

    def test_aircraft_no_config(self):
        """Test Aircraft initialization without config path."""
        aircraft = Aircraft()
        self.assertIsNotNone(aircraft)
        self.assertIsNone(aircraft._state)
        self.assertIsNone(aircraft._control)

    def test_aircraft_params(self):
        """Test params property returns configuration dictionary."""
        params = self.aircraft.params
        self.assertIsInstance(params, dict)
        self.assertEqual(params["mass"], 120000)
        self.assertIn("state", params)

    # --- State and Control Management ---

    def test_aircraft_state_property(self):
        """Test Aircraft.state property."""
        state = self.aircraft.state
        self.assertIsInstance(state, State)
        self.assertEqual(state.u, 100.0)

    def test_aircraft_control_property(self):
        """Test Aircraft.control property."""
        control = self.aircraft.control
        self.assertIsInstance(control, Control)
        self.assertEqual(control.throttle_1, 0.3)

    def test_aircraft_set_state_from_array(self):
        """Test Aircraft.set_state_from_array method."""
        new_state = np.array([200.0, 5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1000.0])
        self.aircraft.set_state_from_array(new_state)
        self.assertEqual(self.aircraft.state.u, 200.0)
        self.assertEqual(self.aircraft.state.v, 5.0)

    def test_aircraft_set_control_from_array(self):
        """Test Aircraft.set_control_from_array method."""
        new_control = np.array([np.radians(5.0), np.radians(10.0), np.radians(-2.0), 0.8, 0.8])
        self.aircraft.set_control_from_array(new_control)
        self.assertEqual(self.aircraft.control.aileron, np.radians(5.0))
        self.assertEqual(self.aircraft.control.throttle_1, 0.8)

    def test_aircraft_state_setter(self):
        """Test Aircraft.state setter."""
        new_state = State(np.ones(12))
        self.aircraft.state = new_state
        self.assertEqual(self.aircraft.state.u, 1.0)

    def test_aircraft_control_setter(self):
        """Test Aircraft.control setter."""
        new_control = Control(np.ones(5))
        self.aircraft.control = new_control
        self.assertEqual(self.aircraft.control.aileron, 1.0)

    # --- Trim Configuration ---

    def test_trim_disabled(self):
        """Test trim does nothing when disabled in config."""
        aircraft = Aircraft(self.config_path)
        result = aircraft.trim()
        self.assertIsNone(result)  # Should return None when trim is disabled
        self.assertEqual(aircraft.trimmed, False)

    def test_trim_enabled(self):
        """Test trim functionality when enabled."""
        self.test_config["trim"] = True
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            aircraft = Aircraft(config_path)
            aircraft.trim()
            # If trim is successful, trimmed flag should be set
            self.assertTrue(aircraft.trimmed)

            # Trim values should have changed from initial values
            self.assertNotEqual(aircraft.control.throttle, 0.3)
        finally:
            if config_path and os.path.exists(config_path):
                os.remove(config_path)

    def test_trim_without_velocity(self):
        """Test trim uses current velocity if not specified."""
        self.test_config["trim"] = True
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            aircraft = Aircraft(config_path)
            v_before = aircraft.state.total_velocity
            aircraft.trim()  # No velocity specified = use current
            v_after = aircraft.state.total_velocity
            self.assertAlmostEqual(v_before, v_after, places=5)
        finally:
            if config_path and os.path.exists(config_path):
                os.remove(config_path)

    # --- Simulation ---

    def test_simulation_before_initialization_fails(self):
        """Test simulation fails if state not initialized."""
        aircraft = Aircraft()
        with self.assertRaises(ValueError):
            aircraft.simulate()

    def test_simulation_run(self):
        """Test simulation can run successfully (short simulation)."""
        # Run a short simulation for testing
        t_end = 1.0  # 1 second
        dt = 0.1     # Large timestep for speed
        time, state_history = self.aircraft.simulate(
            t_start=0.0,
            t_end=t_end,
            dt=dt,
            pulse_start=0.5,
            pulse_duration=0.3,
            target_channel=1,
            pulse_magnitude=-5.0
        )

        # Verify output format
        self.assertIsInstance(time, np.ndarray)
        self.assertEqual(len(time), 11)  # 0.0, 0.1, ..., 1.0
        self.assertEqual(state_history.shape, (11, 12))

        # Verify results stored
        self.assertIsNotNone(self.aircraft.time)
        self.assertIsNotNone(self.aircraft.state_history)
        self.assertIsNotNone(self.aircraft.control_history)

    def test_simulation_no_pulse(self):
        """Test simulation with no control pulse."""
        time, state_history = self.aircraft.simulate(
            t_start=0.0,
            t_end=0.5,
            dt=0.1,
            pulse_start=10.0,  # Pulse after simulation ends
            pulse_duration=0.1,
            target_channel=0,
            pulse_magnitude=0.0
        )

        # Should complete successfully
        self.assertEqual(len(time), 6)

    def test_simulation_full_params(self):
        """Test simulation with all parameters specified."""
        time, state_history = self.aircraft.simulate(
            t_start=0.0,
            t_end=0.5,
            dt=0.1,
            pulse_start=0.2,
            pulse_duration=0.2,
            target_channel=1,
            pulse_magnitude=-10.0
        )

        # Verify physical results make sense
        # State should have changed from initial
        initial_state = self.aircraft.state
        final_state_values = state_history[-1]
        final_state = State(final_state_values)

        # Final state should not equal initial (even if slightly)
        self.assertNotEqual(final_state.u, initial_state.u)

    def test_simulation_result_storage(self):
        """Test simulation results are properly stored in aircraft."""
        self.aircraft.simulate(
            t_start=0.0,
            t_end=0.5,
            dt=0.1,
            pulse_start=0.3,
            pulse_duration=0.2,
            target_channel=0,
            pulse_magnitude=-5.0
        )

        # Check results accessible
        self.assertIsNotNone(self.aircraft._time)
        self.assertIsNotNone(self.aircraft._state_history)
        self.assertIsNotNone(self.aircraft._control_history)

        # Check property access
        self.assertIsNotNone(self.aircraft.time)
        self.assertIsNotNone(self.aircraft.state_history)
        self.assertIsNotNone(self.aircraft.control_history)

    # --- Results Analysis ---

    def test_time_property(self):
        """Test Aircraft.time property."""
        # Initially None
        self.assertIsNone(self.aircraft.time)

        # After simulation
        self.aircraft.simulate(t_start=0.0, t_end=0.3, dt=0.1)
        self.assertIsNotNone(self.aircraft.time)

    def test_state_history_property(self):
        """Test Aircraft.state_history property."""
        # Initially None
        self.assertIsNone(self.aircraft.state_history)

        # After simulation
        self.aircraft.simulate(t_start=0.0, t_end=0.3, dt=0.1)
        self.assertIsNotNone(self.aircraft.state_history)

    def test_control_history_property(self):
        """Test Aircraft.control_history property."""
        # Initially None
        self.assertIsNone(self.aircraft.control_history)

        # After simulation
        self.aircraft.simulate(t_start=0.0, t_end=0.3, dt=0.1)
        self.assertIsNotNone(self.aircraft.control_history)

    def test_get_state_at_time(self):
        """Test Aircraft.get_state_at_time method."""
        time, state_history = self.aircraft.simulate(
            t_start=0.0, t_end=0.5, dt=0.1
        )

        # Get state at specific times
        state_t0 = self.aircraft.get_state_at_time(0.0)
        state_t2 = self.aircraft.get_state_at_time(0.2)
        state_t5 = self.aircraft.get_state_at_time(0.5)

        self.assertIsInstance(state_t0, State)
        self.assertIsInstance(state_t2, State)
        self.assertIsInstance(state_t5, State)

    def test_get_state_at_time_before_simulation(self):
        """Test get_state_at_time returns None if simulation not run."""
        state = self.aircraft.get_state_at_time(10.0)
        self.assertIsNone(state)

    # --- Visualization ---

    def test_plot_results_no_simulation_fails(self):
        """Test plot_results raises error if simulation not run."""
        with self.assertRaises(ValueError):
            self.aircraft.plot_results()

    def test_plot_results_after_simulation_runs(self):
        """Test plot_results runs without error after simulation."""
        self.aircraft.simulate(t_start=0.0, t_end=0.3, dt=0.1)
        try:
            self.aircraft.plot_results()
            # If we get here, plotting succeeded
        except Exception as e:
            # Some plotting errors are acceptable (e.g., GUI issues)
            if "Failed to import" in str(e) or "backend" in str(e):
                self.skipTest("Plotting backend not available")
            else:
                raise

    # --- String Representation ---

    def test_str_representation_not_trimmed(self):
        """Test __str__ representation when not trimmed."""
        aircraft = Aircraft(self.config_path)
        s = str(aircraft)
        self.assertIn("NOT TRIMMED", s)
        self.assertIn("State", s)
        self.assertIn("Control", s)

    def test_str_representation_trimmed(self):
        """Test __str__ representation when trimmed."""
        self.test_config["trim"] = True
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name

        try:
            aircraft = Aircraft(config_path)
            aircraft.trim()
            s = str(aircraft)
            self.assertIn("TRIMMED", s)
        finally:
            if config_path and os.path.exists(config_path):
                os.remove(config_path)

    def test_repr_representation(self):
        """Test __repr__ representation."""
        r = repr(self.aircraft)
        self.assertTrue(r.startswith("Aircraft("))

    # --- Cache Management ---

    def test_clear_cache_internal(self):
        """Test _clear_cache clears the internal cache."""
        # Initialize something that might get cached
        _ = self.aircraft.state.u
        self.assertIsInstance(self.aircraft._cache, dict)

    # --- Integration Test ---

    def test_full_workflow(self):
        """Test complete workflow: init -> simulate -> plot."""
        # 1. Initialize
        self.assertIsInstance(self.aircraft, Aircraft)

        # 2. Run short simulation
        time, state_history = self.aircraft.simulate(
            t_start=0.0, t_end=0.4, dt=0.1
        )
        self.assertIsNotNone(time)
        self.assertIsNotNone(state_history)

        # 3. Verify results can be accessed
        state_at_time = self.aircraft.get_state_at_time(0.2)
        self.assertIsNotNone(state_at_time)

    def test_simulation_stability(self):
        """Test simulation produces stable, repeatable results."""
        # Run simulation twice with same parameters
        results1 = self.aircraft.simulate(
            t_start=0.0, t_end=0.3, dt=0.1
        )

        time1, history1 = results1

        # Reset aircraft to same initial state
        self.aircraft = Aircraft(self.config_path)

        results2 = self.aircraft.simulate(
            t_start=0.0, t_end=0.3, dt=0.1
        )

        time2, history2 = results2

        # Results should be numerically identical
        np.testing.assert_array_equal(time1, time2)
        np.testing.assert_array_almost_equal(history1, history2)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
