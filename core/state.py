"""
Aircraft State Management

Provides a class-based interface to aircraft state vectors, eliminating magic array indices.
"""

import numpy as np


class State:
    """
    Encapsulates the 12-element aircraft state vector with named properties.

    State vector structure:
    [0:3] - velocities: u, v, w (m/s)
    [3:6] - angular velocities: p, q, r (rad/s)
    [6:9] - euler angles: phi, theta, psi (rad)
    [9:12] - position: p_n, p_e, p_d (m)
    """

    # Define indices as class constants
    U, V, W = 0, 1, 2  # Velocities
    P, Q, R = 3, 4, 5  # Angular velocities
    PHI, THETA, PSI = 6, 7, 8  # Euler angles
    P_N, P_E, P_D = 9, 10, 11  # Position

    def __init__(self, state_array=None):
        """
        Initialize State from a 12-element numpy array or zeros.

        Args:
            state_array: Optional 12-element numpy array. If None, initializes to zeros.
        """
        if state_array is None:
            self._data = np.zeros(12, dtype=np.float64)
        else:
            if len(state_array) != 12:
                raise ValueError(f"State array must have 12 elements, got {len(state_array)}")
            self._data = np.array(state_array, dtype=np.float64, copy=True)

        # Cache for computed values
        self._cache = {}

    # --- Vector Access ---

    @property
    def data(self) -> np.ndarray:
        """Get the underlying numpy array."""
        return self._data

    def to_array(self) -> np.ndarray:
        """Convert state to array (alias for data property)."""
        return self._data

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Create State instance from numpy array."""
        return cls(arr)

    # --- Velocities (u, v, w) ---

    @property
    def u(self) -> float:
        """Forward velocity (m/s)."""
        return self._data[self.U]

    @u.setter
    def u(self, value: float):
        self._clear_cache()
        self._data[self.U] = float(value)

    @property
    def v(self) -> float:
        """Lateral velocity (m/s)."""
        return self._data[self.V]

    @v.setter
    def v(self, value: float):
        self._clear_cache()
        self._data[self.V] = float(value)

    @property
    def w(self) -> float:
        """Vertical velocity (m/s)."""
        return self._data[self.W]

    @w.setter
    def w(self, value: float):
        self._clear_cache()
        self._data[self.W] = float(value)

    # --- Angular Velocities (p, q, r) ---

    @property
    def p(self) -> float:
        """Roll rate (rad/s)."""
        return self._data[self.P]

    @p.setter
    def p(self, value: float):
        self._clear_cache()
        self._data[self.P] = float(value)

    @property
    def q(self) -> float:
        """Pitch rate (rad/s)."""
        return self._data[self.Q]

    @q.setter
    def q(self, value: float):
        self._clear_cache()
        self._data[self.Q] = float(value)

    @property
    def r(self) -> float:
        """Yaw rate (rad/s)."""
        return self._data[self.R]

    @r.setter
    def r(self, value: float):
        self._clear_cache()
        self._data[self.R] = float(value)

    # --- Euler Angles (phi, theta, psi) ---

    @property
    def phi(self) -> float:
        """Roll angle (rad)."""
        return self._data[self.PHI]

    @phi.setter
    def phi(self, value: float):
        self._clear_cache()
        self._data[self.PHI] = float(value)

    @property
    def theta(self) -> float:
        """Pitch angle (rad)."""
        return self._data[self.THETA]

    @theta.setter
    def theta(self, value: float):
        self._clear_cache()
        self._data[self.THETA] = float(value)

    @property
    def psi(self) -> float:
        """Yaw angle (rad)."""
        return self._data[self.PSI]

    @psi.setter
    def psi(self, value: float):
        self._clear_cache()
        self._data[self.PSI] = float(value)

    # --- Position (p_n, p_e, p_d) ---

    @property
    def p_n(self) -> float:
        """North position (m)."""
        return self._data[self.P_N]

    @p_n.setter
    def p_n(self, value: float):
        self._clear_cache()
        self._data[self.P_N] = float(value)

    @property
    def p_e(self) -> float:
        """East position (m)."""
        return self._data[self.P_E]

    @p_e.setter
    def p_e(self, value: float):
        self._clear_cache()
        self._data[self.P_E] = float(value)

    @property
    def p_d(self) -> float:
        """Down position (m). Negative indicates altitude above sea level."""
        return self._data[self.P_D]

    @p_d.setter
    def p_d(self, value: float):
        self._clear_cache()
        self._data[self.P_D] = float(value)

    # --- Cached Computed Properties ---

    @property
    def velocity_vector(self) -> np.ndarray:
        """Velocity vector [u, v, w] (m/s)."""
        return self._data[0:3]

    @property
    def angular_velocity_vector(self) -> np.ndarray:
        """Angular velocity vector [p, q, r] (rad/s)."""
        return self._data[3:6]

    @property
    def euler_angles(self) -> np.ndarray:
        """Euler angles [phi, theta, psi] (rad)."""
        return self._data[6:9]

    @property
    def position(self) -> np.ndarray:
        """Position vector [p_n, p_e, p_d] (m)."""
        return self._data[9:12]

    @property
    def total_velocity(self) -> float:
        """Total velocity magnitude (m/s). Cached for performance."""
        if 'V_a' not in self._cache:
            self._cache['V_a'] = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        return self._cache['V_a']

    @property
    def altitude(self) -> float:
        """Altitude above sea level (m)."""
        return -self.p_d

    def _clear_cache(self):
        """Clear cached computed values when state changes."""
        self._cache.clear()

    # --- String Representation ---

    def __str__(self) -> str:
        """String representation showing key state variables."""
        return (
            f"State(u={self.u:.2f} m/s, v={self.v:.2f} m/s, w={self.w:.2f} m/s, "
            f"V={self.total_velocity:.2f} m/s, "
            f"phi={np.degrees(self.phi):.2f}°, theta={np.degrees(self.theta):.2f}°, "
            f"psi={np.degrees(self.psi):.2f}°, "
            f"altitude={self.altitude:.2f} m)"
        )

    def __repr__(self) -> str:
        return f"State({self._data})"
