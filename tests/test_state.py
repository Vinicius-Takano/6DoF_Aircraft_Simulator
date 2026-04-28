"""
Unit tests for State class
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.state import State


class TestState(unittest.TestCase):
    """Test cases for State class."""

    def setUp(self):
        """Set up test fixtures."""
        self.zero_state = State()
        self.test_array = np.array([
            100.0, 0.0, 0.0,  # velocities
            0.1, 0.2, 0.3,    # angular velocities
            0.0, 0.1, 0.0,    # euler angles
            0.0, 0.0, -1000.0  # position
        ])
        self.test_state = State(self.test_array)

    def test_init_default(self):
        """Test State initialization with default (zero) values."""
        state = State()
        self.assertEqual(len(state.data), 12)
        np.testing.assert_array_equal(state.data, np.zeros(12))

    def test_init_with_array(self):
        """Test State initialization with custom array."""
        np.testing.assert_array_equal(self.test_state.data, self.test_array)

    def test_init_invalid_length(self):
        """Test State initialization with invalid array length raises error."""
        with self.assertRaises(ValueError):
            State(np.array([1, 2, 3]))

    def test_init_preserves_array(self):
        """Test that State makes a copy of input array."""
        original = self.test_array.copy()
        state = State(self.test_array)
        self.test_array[0] = 999.0
        self.assertEqual(state.u, original[0])  # State should have original value

    def test_from_array(self):
        """Test State.from_array class method."""
        state = State.from_array(self.test_array)
        self.assertIsInstance(state, State)
        np.testing.assert_array_equal(state.data, self.test_array)

    def test_to_array(self):
        """Test State.to_array method."""
        arr = self.test_state.to_array()
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, self.test_array)

    def test_data_property(self):
        """Test State.data property."""
        arr = self.test_state.data[0:3]
        np.testing.assert_array_equal(arr, [100.0, 0.0, 0.0])

    # --- Velocity Properties ---

    def test_u_property(self):
        """Test u property getter and setter."""
        self.assertEqual(self.test_state.u, 100.0)
        self.test_state.u = 150.0
        self.assertEqual(self.test_state.u, 150.0)

    def test_v_property(self):
        """Test v property getter and setter."""
        self.assertEqual(self.test_state.v, 0.0)
        self.test_state.v = 10.0
        self.assertEqual(self.test_state.v, 10.0)

    def test_w_property(self):
        """Test w property getter and setter."""
        self.assertEqual(self.test_state.w, 0.0)
        self.test_state.w = 5.0
        self.assertEqual(self.test_state.w, 5.0)

    def test_velocity_vector(self):
        """Test velocity_vector property."""
        np.testing.assert_array_equal(self.test_state.velocity_vector, [100.0, 0.0, 0.0])

    # --- Angular Velocity Properties ---

    def test_p_property(self):
        """Test p property getter and setter."""
        self.assertEqual(self.test_state.p, 0.1)
        self.test_state.p = 0.5
        self.assertEqual(self.test_state.p, 0.5)

    def test_q_property(self):
        """Test q property getter and setter."""
        self.assertEqual(self.test_state.q, 0.2)
        self.test_state.q = 0.3
        self.assertEqual(self.test_state.q, 0.3)

    def test_r_property(self):
        """Test r property getter and setter."""
        self.assertEqual(self.test_state.r, 0.3)
        self.test_state.r = 0.4
        self.assertEqual(self.test_state.r, 0.4)

    def test_angular_velocity_vector(self):
        """Test angular_velocity_vector property."""
        np.testing.assert_array_equal(self.test_state.angular_velocity_vector, [0.1, 0.2, 0.3])

    # --- Euler Angles ---

    def test_phi_property(self):
        """Test phi property getter and setter."""
        self.assertEqual(self.test_state.phi, 0.0)
        self.test_state.phi = np.radians(45.0)
        self.assertAlmostEqual(self.test_state.phi, np.radians(45.0))

    def test_theta_property(self):
        """Test theta property getter and setter."""
        self.assertEqual(self.test_state.theta, 0.1)
        self.test_state.theta = np.radians(30.0)
        self.assertAlmostEqual(self.test_state.theta, np.radians(30.0))

    def test_psi_property(self):
        """Test psi property getter and setter."""
        self.assertEqual(self.test_state.psi, 0.0)
        self.test_state.psi = np.radians(90.0)
        self.assertAlmostEqual(self.test_state.psi, np.radians(90.0))

    def test_euler_angles(self):
        """Test euler_angles property."""
        np.testing.assert_array_equal(self.test_state.euler_angles, [0.0, 0.1, 0.0])

    # --- Position Properties ---

    def test_p_n_property(self):
        """Test p_n property getter and setter."""
        self.assertEqual(self.test_state.p_n, 0.0)
        self.test_state.p_n = 100.0
        self.assertEqual(self.test_state.p_n, 100.0)

    def test_p_e_property(self):
        """Test p_e property getter and setter."""
        self.assertEqual(self.test_state.p_e, 0.0)
        self.test_state.p_e = 200.0
        self.assertEqual(self.test_state.p_e, 200.0)

    def test_p_d_property(self):
        """Test p_d property getter and setter."""
        self.assertEqual(self.test_state.p_d, -1000.0)
        self.test_state.p_d = -500.0
        self.assertEqual(self.test_state.p_d, -500.0)

    def test_position_vector(self):
        """Test position property."""
        np.testing.assert_array_equal(self.test_state.position, [0.0, 0.0, -1000.0])

    def test_altitude_property(self):
        """Test altitude property."""
        self.assertEqual(self.test_state.altitude, 1000.0)
        self.test_state.p_d = -2000.0
        self.assertEqual(self.test_state.altitude, 2000.0)

    # --- Cache Tests ---

    def test_cache_clear_on_state_change(self):
        """Test that cache is cleared when state changes."""
        initial_V = self.test_state.total_velocity

        # Access total_velocity (caches it)
        self.assertEqual(self.test_state.total_velocity, initial_V)

        # Store current cache dictionary id
        first_cache = self.test_state._cache

        # Change velocity
        self.test_state.u = 200.0

        # Cache should be emptied (new dict or cleared)
        self.assertEqual(len(self.test_state._cache), 0)

        # New calculated velocity should reflect change
        new_V = self.test_state.total_velocity
        self.assertNotEqual(new_V, initial_V)

    # --- Total Velocity ---

    def test_total_velocity(self):
        """Test total_velocity property with correct calculation."""
        expected = np.sqrt(100.0**2 + 0.0**2 + 0.0**2)
        self.assertEqual(self.test_state.total_velocity, expected)

    def test_total_velocity_property_caching(self):
        """Test that total_velocity is cached after first access."""
        # First access should calculate
        v1 = self.test_state.total_velocity
        # Second access should use cache
        v2 = self.test_state.total_velocity
        self.assertEqual(v1, v2)
        # Cache should have entry
        self.assertIn('V_a', self.test_state._cache)

    def test_total_velocity_after_velocity_change(self):
        """Test total_velocity reflects velocity changes."""
        self.assertEqual(self.test_state.total_velocity, 100.0)
        self.test_state.u = 50.0
        self.test_state.v = 50.0
        self.test_state.w = 50.0
        expected = np.sqrt(50.0**2 + 50.0**2 + 50.0**2)
        self.assertAlmostEqual(self.test_state.total_velocity, expected)

    # --- String Representation ---

    def test_str_representation(self):
        """Test __str__ representation."""
        s = str(self.test_state)
        self.assertIn("u=100.00", s)
        self.assertIn("v=0.00", s)
        self.assertIn("V=100.00", s)
        self.assertIn("altitude=1000.00", s)

    def test_repr_representation(self):
        """Test __repr__ representation."""
        r = repr(self.test_state)
        self.assertTrue(r.startswith("State("))
        self.assertIn("1.e+02", r)

    # --- Round Trip Tests ---

    def test_array_to_state_to_array(self):
        """Test round trip from array to State and back to array."""
        state = State.from_array(self.test_array)
        result_array = state.to_array()
        np.testing.assert_array_equal(result_array, self.test_array)

    def test_state_creation_various_values(self):
        """Test State creation with various realistic values."""
        # High speed
        fast = State(np.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -10000.0]))
        self.assertEqual(fast.u, 300.0)
        self.assertEqual(fast.altitude, 10000.0)

        # High altitude
        high = State(np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -15000.0]))
        self.assertEqual(high.altitude, 15000.0)

        # Steep climb
        climb = State(np.array([100.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -5000.0]))
        self.assertAlmostEqual(climb.total_velocity, np.sqrt(100**2 + 50**2))


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
