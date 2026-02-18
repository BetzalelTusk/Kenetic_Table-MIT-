"""
TableVisualizer — Real-time 3D Surface Renderer
=================================================
Uses matplotlib's mplot3d to render the pin grid as a terrain surface.
Height 0 → dark (valley), Height 100 → bright (peak).
Includes slow auto-rotation for a cinematic demo feel.
"""

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
import numpy as np


class TableVisualizer:
    def __init__(self, grid_size=(30, 30), max_height=100.0, **kwargs):
        self.rows, self.cols = grid_size
        self.max_height = max_height

        # ── Dark theme ────────────────────────────────────────────────── #
        plt.style.use("dark_background")
        plt.ion()

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.manager.set_window_title("Kinetic Table Simulator — 3D")

        # ── Coordinate grids ─────────────────────────────────────────── #
        self.X, self.Y = np.meshgrid(
            np.arange(self.cols),
            np.arange(self.rows),
        )

        # ── Fixed colour-bar (won't redraw every frame) ──────────────── #
        norm = mcolors.Normalize(vmin=0, vmax=max_height)
        mappable = cm.ScalarMappable(norm=norm, cmap="plasma")
        mappable.set_array([])
        self.fig.colorbar(mappable, ax=self.ax, label="Height (mm)",
                          shrink=0.55, pad=0.12)

        # ── Axis styling ─────────────────────────────────────────────── #
        self.ax.set_zlim(0, max_height)
        self.ax.set_xlabel("Column", labelpad=10)
        self.ax.set_ylabel("Row", labelpad=10)
        self.ax.set_zlabel("Height (mm)", labelpad=10)
        self.ax.set_title("Kinetic Table  —  Live Pin Heights",
                          fontsize=15, pad=20, color="white")

        # ── Camera ───────────────────────────────────────────────────── #
        self._base_azim = -60
        self.ax.view_init(elev=35, azim=self._base_azim)

        # ── Status bar text ──────────────────────────────────────────── #
        self._mood_text = self.fig.text(
            0.02, 0.02, "", fontsize=10, color="cyan", alpha=0.85,
            family="monospace",
        )

        # ── Internal state ───────────────────────────────────────────── #
        self._surf = None
        self._frame = 0
        self._open = True
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.fig.tight_layout()

    # ------------------------------------------------------------------ #
    def _on_close(self, _event):
        self._open = False

    # ------------------------------------------------------------------ #
    def update_pins(self, height_matrix, mood_text=""):
        """
        Redraw the 3D surface.  Returns False when the window is closed.
        """
        if not self._open:
            return False

        # Remove previous surface
        if self._surf is not None:
            try:
                self._surf.remove()
            except ValueError:
                pass

        # Draw new surface
        self._surf = self.ax.plot_surface(
            self.X, self.Y, height_matrix,
            cmap="plasma",
            vmin=0,
            vmax=self.max_height,
            alpha=0.92,
            rstride=1,
            cstride=1,
            antialiased=False,   # faster rendering for real-time
        )

        # Keep axis limits fixed
        self.ax.set_zlim(0, self.max_height)

        # Slow cinematic rotation (~0.75 °/s at 15 fps)
        self._frame += 1
        azim = self._base_azim + self._frame * 0.05
        self.ax.view_init(elev=35, azim=azim)

        # Mood indicator
        if mood_text:
            self._mood_text.set_text(f"AI Mood: {mood_text}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        return self._open

    # ------------------------------------------------------------------ #
    def close(self):
        plt.close(self.fig)
