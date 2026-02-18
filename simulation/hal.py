import numpy as np
import time

class KineticTable:
    def __init__(self, grid_size=(30, 30), max_height=100.0, max_speed=50.0, update_rate=60):
        """
        Initializes the Kinetic Table Hardware Abstraction Layer.

        Args:
            grid_size (tuple): Dimensions of the pin grid (rows, cols). Default is (30, 30).
            max_height (float): Maximum height of a pin in mm. Default is 100.0.
            max_speed (float): Maximum speed of a pin in mm/s. Default is 50.0.
            update_rate (int): Frequency of updates in Hz (for simulation logic).
        """
        self.rows, self.cols = grid_size
        self.max_height = max_height
        self.max_speed = max_speed  # mm per second
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate # Time step per update

        # State Matrices
        # Using float32 for performance and sufficient precision
        self.current_heights = np.zeros(grid_size, dtype=np.float32)
        self.target_heights = np.zeros(grid_size, dtype=np.float32)

        # Simulation Time Keeping
        self.last_update_time = time.time()

    def set_target(self, target_matrix):
        """
        Sets the target heights for the pins.
        
        Args:
            target_matrix (np.ndarray): A 2D array of target heights. 
                                        Must match grid_size. 
                                        Values will be clamped to [0, max_height].
        """
        if target_matrix.shape != (self.rows, self.cols):
            raise ValueError(f"Target matrix shape {target_matrix.shape} does not match grid size {(self.rows, self.cols)}")
        
        # Clamp values to valid range
        self.target_heights = np.clip(target_matrix, 0, self.max_height).astype(np.float32)

    def update(self, dt=None):
        """
        Updates the physical simulation of the pins based on elapsed time.
        The pins move towards their targets at a maximum speed.

        Args:
            dt (float, optional): Time delta in seconds. If None, calculated automatically.
        """
        if dt is None:
            now = time.time()
            dt = now - self.last_update_time
            self.last_update_time = now
        else:
            # If manual dt is provided, still update the last_update_time to keep consistent
            self.last_update_time = time.time()

        # Calculate the distance to move
        diff = self.target_heights - self.current_heights
        
        # Determine the maximum distance allowed in this time step
        max_move_dist = self.max_speed * dt
        
        # Calculate the step to take (clamped by max speed)
        # We want to move towards target but not overshoot if close
        # np.sign(diff) gives direction
        # np.minimum(abs(diff), max_move_dist) gives magnitude limited by speed
        
        move_step = np.sign(diff) * np.minimum(np.abs(diff), max_move_dist)
        
        # Update current heights
        self.current_heights += move_step

        # Ensure we stay within bounds (floating point errors can accumulate)
        self.current_heights = np.clip(self.current_heights, 0, self.max_height)

    def get_display_matrix(self):
        """
        Returns the current heights of the pins.
        """
        return self.current_heights.copy()

    def reset(self):
        """Resets the table to flat state immediately."""
        self.current_heights.fill(0)
        self.target_heights.fill(0)
