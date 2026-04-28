"""
Unit tests for Control class
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.control import Control


class TestControl(unittest.TestCase):
    """Test cases for Control class."""

    def setUp(self):
        """Set up test fixtures."""
        self.zero_control = Control()
        self.test_array = np.array([
            np.radians(5.0),   # aileron
            np.radians(10.0),  # elevator
            np.radians(-5.0), # rudder
            0.5,                # throttle_1
            0.5                 # throttle_2
        ])
        self.test_control = Control(self.test_array)

    def test_init_default(self):
        """Test Control initialization with default (zero) values."""
        control = Control()
        self.assertEqual(len(control.data), 5)
        np.testing.assert_array_equal(control.data, np.zeros(5))

    def test_init_with_array(self):
        """Test Control initialization with custom array."""
        np.testing.assert_array_equal(self.test_control.data, self.test_array)

    def test_init_invalid_length(self):
        """Test Control initialization with invalid array length raises error."""
        with self.assertRaises(ValueError):
            Control(np.array([1, 2]))

    def test_init_preserves_array(self):
        """Test that Control makes a copy of input array."""
        original = self.test_array.copy()
        control = Control(self.test_array)
        self.test_array[0] = -999.0
        self.assertNotEqual(control.data[0], -999.0)

    def test_from_array(self):
        """Test Control.from_array class method."""
        control = Control.from_array(self.test_array)
        self.assertIsInstance(control, Control)
        np.testing.assert_array_equal(control.data, self.test_array)

    def test_to_array(self):
        """Test Control.to_array method."""
        arr = self.test_control.to_array()
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, self.test_array)

    def test_data_property(self):
        """Test Control.data property."""
        arr = self.test_control.data[0:3]
        expected = [np.radians(5.0), np.radians(10.0), np.radians(-5.0)]
        np.testing.assert_array_almost_equal(arr, expected)

    # --- Control Surface Properties ---

    def test_aileron_property(self):
        """Test aileron property getter and setter."""
        self.assertEqual(self.test_control.aileron, np.radians(5.0))
        self.test_control.aileron = np.radians(15.0)
        self.assertEqual(self.test_control.aileron, np.radians(15.0))

    def test_elevator_property(self):
        """Test elevator property getter and setter."""
        self.assertEqual(self.test_control.elevator, np.radians(10.0))
        self.test_control.elevator = np.radians(-5.0)
        self.assertEqual(self.test_control.elevator, np.radians(-5.0))

    def test_rudder_property(self):
        """Test rudder property getter and setter."""
        self.assertEqual(self.test_control.rudder, np.radians(-5.0))
        self.test_control.rudder = np.radians(10.0)
        self.assertEqual(self.test_control.rudder, np.radians(10.0))

    # --- Throttle Properties ---

    def test_throttle_1_property(self):
        """Test throttle_1 property getter and setter."""
        self.assertEqual(self.test_control.throttle_1, 0.5)
        self.test_control.throttle_1 = 0.75
        self.assertEqual(self.test_control.throttle_1, 0.75)

    def test_throttle_2_property(self):
        """Test throttle_2 property getter and setter."""
        self.assertEqual(self.test_control.throttle_2, 0.5)
        self.test_control.throttle_2 = 0.9
        self.assertEqual(self.test_control.throttle_2, 0.9)

    def test_throttle_property(self):
        """Test throttle property getter and setter."""
        # Test getter (average of throttle_1 and throttle_2)
        self.assertEqual(self.test_control.throttle, 0.5)

        # Test setter (sets both)
        self.test_control.throttle = 0.8
        self.assertEqual(self.test_control.throttle_1, 0.8)
        self.assertEqual(self.test_control.throttle_2, 0.8)

    def test_throttle_clipping_low(self):
        """Test throttle value is clipped to minimum of 0.0."""
        self.test_control.throttle = -0.5
        self.assertEqual(self.test_control.throttle_1, 0.0)
        self.assertEqual(self.test_control.throttle_2, 0.0)

    def test_throttle_clipping_high(self):
        """Test throttle value is clipped to maximum of 1.0."""
        self.test_control.throttle = 1.5
        self.assertEqual(self.test_control.throttle_1, 1.0)
        self.assertEqual(self.test_control.throttle_2, 1.0)

    # --- Helpers ---

    def test_in_degrees(self):
        """Test in_degrees method converts surfaces to degrees."""
        deg_values = self.test_control.in_degrees()
        expected = np.array([5.0, 10.0, -5.0, 0.5, 0.5])
        np.testing.assert_array_almost_equal(deg_values, expected)
        self.assertNotEqual(deg_values[0], self.test_control.aileron)  # Should be different

    def test_apply_pulse_to_surface(self):
        """Test apply_pulse for aileron surface (degrees)."""
        initial = self.test_control.aileron
        self.test_control.apply_pulse(0, 5.0)  # Add 5 degrees
        expected = initial + np.radians(5.0)
        self.assertEqual(self.test_control.aileron, expected)

    def test_apply_pulse_to_elevator_surface(self):
        """Test apply_pulse for elevator surface (degrees)."""
        initial = self.test_control.elevator
        self.test_control.apply_pulse(1, -10.0)  # Add -10 degrees
        expected = initial + np.radians(-10.0)
        self.assertEqual(self.test_control.elevator, expected)

    def test_apply_pulse_to_rudder_surface(self):
        """Test apply_pulse for rudder surface (degrees)."""
        initial = self.test_control.rudder
        self.test_control.apply_pulse(2, 15.0)  # Add 15 degrees
        expected = initial + np.radians(15.0)
        self.assertEqual(self.test_control.rudder, expected)

    def test_apply_pulse_to_throttle(self):
        """Test apply_pulse for throttle (percentage)."""
        initial = self.test_control.throttle_1
        self.test_control.apply_pulse(3, 0.2)  # Add 20% throttle
        expected = initial + 0.2
        self.assertEqual(self.test_control.throttle_1, expected)

    def test_apply_pulse_to_throttle_2(self):
        """Test apply_pulse for throttle_2 (percentage)."""
        initial = self.test_control.throttle_2
        self.test_control.apply_pulse(4, -0.1)  # Reduce throttle by 10%
        expected = initial - 0.1
        self.assertEqual(self.test_control.throttle_2, expected)

    # --- String Representation ---

    def test_str_representation(self):
        """Test __str__ representation."""
        s = str(self.test_control)
        self.assertIn("aileron=5.00", s)
        self.assertIn("elevator=10.00", s)
        self.assertIn("rudder=-5.00", s)
        self.assertIn("throttle_1=50.0%", s)
        self.assertIn("throttle_2=50.0%", s)

    def test_repr_representation(self):
        """Test __repr__ representation."""
        r = repr(self.test_control)
        self.assertTrue(r.startswith("Control("))

    # --- Round Trip Tests ---

    def test_array_to_control_to_array(self):
        """Test round trip from array to Control and back to array."""
        control = Control.from_array(self.test_array)
        result_array = control.to_array()
        np.testing.assert_array_equal(result_array, self.test_array)

    def test_control_creation_various_values(self):
        """Test Control creation with various realistic values."""
        # Neutral control
        neutral = Control(np.array([0.0, 0.0, 0.0, 0.3, 0.3]))
        self.assertEqual(neutral.aileron, 0.0)
        self.assertEqual(neutral.throttle_1, 0.3)

        # Maximum control deflections
        extreme = Control(np.array([
            np.radians(25.0),   # aileron
            np.radians(10.0),   # elevator (max up)
            np.radians(-30.0),  # rudder
            1.0,                # full throttle
            1.0
        ]))
        self.assertEqual(extreme.aileron, np.radians(25.0))
        self.assertEqual(extreme.elevator, np.radians(10.0))
        self.assertEqual(extreme.rudder, np.radians(-30.0))
        self.assertEqual(extreme.throttle, 1.0)

        # Mixed controls
        mixed = Control(np.array([
            np.radians(-5.0),  # left aileron
            np.radians(-2.0), # elevator down
            np.radians(3.0),  # rudder right
            0.6,              # medium throttle
            0.8
        ]))
        self.assertEqual(mixed.aileron, np.radians(-5.0))
        self.assertEqual(mixed.throttle_1, 0.6)
        self.assertEqual(mixed.throttle_2, 0.8)

    # --- Edge Cases ---

    def test_aileron_max_deflection(self):
        """Test at maximum aileron deflection."""
        max_aileron = np.radians(25.0)
        control = Control()
        control.aileron = max_aileron
        self.assertEqual(control.aileron, max_aileron)

    def test_elevator_max_up(self):
        """Test at maximum elevator up deflection."""
        max_up = np.radians(10.0)
        control = Control()
        control.elevator = max_up
        self.assertEqual(control.elevator, max_up)

    def test_elevator_max_down(self):
        """Test at maximum elevator down deflection."""
        max_down = np.radians(-25.0)
        control = Control()
        control.elevator = max_down
        self.assertEqual(control.elevator, max_down)

    def test_rudder_max_deflection(self):
        """Test at maximum rudder deflection."""
        max_rudder = np.radians(-30.0)
        control = Control()
        control.rudder = max_rudder
        self.assertEqual(control.rudder, max_rudder)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
