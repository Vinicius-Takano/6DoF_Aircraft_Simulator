"""
Aircraft Control Management

Provides a class-based interface to aircraft control vectors.
"""

import numpy as np


class Control:
    """
    Encapsulates the 5-element aircraft control vector with named properties.

    Control vector structure:
    [0] - aileron (rad)
    [1] - elevator (rad)
    [2] - rudder (rad)
    [3] - throttle_1 (0-1)
    [4] - throttle_2 (0-1)
    """

    # Define indices as class constants
    AILERON = 0
    ELEVATOR = 1
    RUDDER = 2
    THROTTLE_1 = 3
    THROTTLE_2 = 4

    def __init__(self, control_array=None):
        """
        Initialize Control from a 5-element numpy array.

        Args:
            control_array: Optional 5-element numpy array. If None, initializes to zeros.
        """
        if control_array is None:
            self._data = np.zeros(5, dtype=np.float64)
        else:
            if len(control_array) != 5:
                raise ValueError(f"Control array must have 5 elements, got {len(control_array)}")
            self._data = np.array(control_array, dtype=np.float64, copy=True)

    # --- Vector Access ---

    @property
    def data(self) -> np.ndarray:
        """Get the underlying numpy array."""
        return self._data

    def to_array(self) -> np.ndarray:
        """Convert control to array (alias for data property)."""
        return self._data

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Create Control instance from numpy array."""
        return cls(arr)

    # --- Control Surfaces ---

    @property
    def aileron(self) -> float:
        """Aileron deflection (rad). Positive is right aileron down."""
        return self._data[self.AILERON]

    @aileron.setter
    def aileron(self, value: float):
        self._data[self.AILERON] = float(value)

    @property
    def elevator(self) -> float:
        """Elevator deflection (rad). Positive is trailing edge down."""
        return self._data[self.ELEVATOR]

    @elevator.setter
    def elevator(self, value: float):
        self._data[self.ELEVATOR] = float(value)

    @property
    def rudder(self) -> float:
        """Rudder deflection (rad). Positive is trailing edge left."""
        return self._data[self.RUDDER]

    @rudder.setter
    def rudder(self, value: float):
        self._data[self.RUDDER] = float(value)

    # --- Throttle ---

    @property
    def throttle_1(self) -> float:
        """Engine 1 throttle setting (0-1)."""
        return self._data[self.THROTTLE_1]

    @throttle_1.setter
    def throttle_1(self, value: float):
        self._data[self.THROTTLE_1] = np.clip(float(value), 0.0, 1.0)

    @property
    def throttle_2(self) -> float:
        """Engine 2 throttle setting (0-1)."""
        return self._data[self.THROTTLE_2]

    @throttle_2.setter
    def throttle_2(self, value: float):
        self._data[self.THROTTLE_2] = np.clip(float(value), 0.0, 1.0)

    @property
    def throttle(self) -> float:
        """Average throttle setting (0-1)."""
        return (self.throttle_1 + self.throttle_2) / 2.0

    @throttle.setter
    def throttle(self, value: float):
        clipped = np.clip(float(value), 0.0, 1.0)
        self.throttle_1 = clipped
        self.throttle_2 = clipped

    # --- Helpers ---

    def in_degrees(self) -> np.ndarray:
        """Return control surface deflections in degrees."""
        arr = np.copy(self._data)
        arr[0:3] = np.degrees(arr[0:3])
        return arr

    def apply_pulse(self, channel: int, value: float):
        """
        Apply a pulse to a specific control channel.

        Args:
            channel: Control channel index (0-4)
            value: Pulse magnitude (degrees for surfaces, percentage for throttle)
        """
        if channel == self.AILERON or channel == self.ELEVATOR or channel == self.RUDDER:
            modifier = value * np.pi / 180.0
        else:
            modifier = value

        self._data[channel] += modifier

    # --- String Representation ---

    def __str__(self) -> str:
        """String representation showing control values in degrees."""
        deg = self.in_degrees()
        return (
            f"Control(aileron={deg[0]:.2f}°, elevator={deg[1]:.2f}°, "
            f"rudder={deg[2]:.2f}°, throttle_1={self.throttle_1*100:.1f}%, "
            f"throttle_2={self.throttle_2*100:.1f}%)"
        )

    def __repr__(self) -> str:
        return f"Control({self._data})"
