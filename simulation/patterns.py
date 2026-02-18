"""
PatternEngine — Mathematical Pattern Generator
================================================
Provides 7 pattern types that the AI Brain can choose from.
Each pattern takes a time value and parameters, and returns
a 30x30 numpy height map (0 to max_height).

Patterns:
  wave, ripple, breathe, mountain, spiral, rain, chaos
"""

import numpy as np


class PatternEngine:
    def __init__(self, grid_size=(30, 30), max_height=100.0):
        self.rows, self.cols = grid_size
        self.max_height = max_height

        # Pre-compute normalised coordinate grids (0→1)
        self.x = np.linspace(0, 1, self.cols)
        self.y = np.linspace(0, 1, self.rows)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Active pattern
        self._pattern = "wave"
        self._params = {"frequency": 2.0, "speed": 1.0, "direction_angle": 0}

    # ── Public API ────────────────────────────────────────────────────── #

    def set_pattern(self, name: str, params: dict):
        self._pattern = name
        self._params = params

    def generate(self, t: float) -> np.ndarray:
        """Return a (rows, cols) height map for time *t*."""
        fn = getattr(self, f"_p_{self._pattern}", self._p_wave)
        return fn(t, self._params)

    # ── Pattern implementations ───────────────────────────────────────── #

    def _p_wave(self, t, p):
        freq = p.get("frequency", 2.0)
        speed = p.get("speed", 1.0)
        angle = np.radians(p.get("direction_angle", 0))
        D = self.X * np.cos(angle) + self.Y * np.sin(angle)
        Z = np.sin(D * freq * 2 * np.pi + t * speed * 2)
        return self._remap(Z)

    def _p_ripple(self, t, p):
        cx = p.get("center_x", 15) / (self.cols - 1)
        cy = p.get("center_y", 15) / (self.rows - 1)
        freq = p.get("frequency", 3.0)
        speed = p.get("speed", 2.0)
        dist = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
        Z = np.sin(dist * freq * 2 * np.pi - t * speed * 2) * np.exp(-dist * 2)
        return self._remap(Z)

    def _p_breathe(self, t, p):
        speed = p.get("speed", 1.0)
        amp = p.get("max_amplitude", 80)
        base = (np.sin(t * speed) + 1) / 2
        spatial = 1 + 0.15 * np.sin(self.X * np.pi) * np.sin(self.Y * np.pi)
        return np.clip(base * amp * spatial, 0, self.max_height)

    def _p_mountain(self, t, p):
        peaks = p.get("peaks", [{"x": 15, "y": 15, "height": 90, "spread": 5}])
        Z = np.zeros((self.rows, self.cols))
        for pk in peaks:
            px = pk.get("x", 15) / (self.cols - 1)
            py = pk.get("y", 15) / (self.rows - 1)
            h = pk.get("height", 80)
            s = pk.get("spread", 5) / 30.0
            breath = 0.8 + 0.2 * np.sin(t * 0.5 + px * 3)
            G = np.exp(-((self.X - px) ** 2 + (self.Y - py) ** 2) / (2 * s ** 2))
            Z += G * h * breath
        return np.clip(Z, 0, self.max_height)

    def _p_spiral(self, t, p):
        arms = p.get("arms", 2)
        speed = p.get("speed", 1.5)
        tight = p.get("tightness", 2.0)
        dx = self.X - 0.5
        dy = self.Y - 0.5
        angle = np.arctan2(dy, dx)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        Z = np.sin(angle * arms + dist * tight * 2 * np.pi - t * speed * 2)
        return self._remap(Z)

    def _p_rain(self, t, p):
        intensity = int(p.get("intensity", 5))
        speed = p.get("drop_speed", 3.0)
        Z = np.zeros((self.rows, self.cols))
        for i in range(intensity):
            phase = int(t * 0.5) + i * 17
            cx = np.sin(phase * 1.1 + i) * 0.4 + 0.5
            cy = np.cos(phase * 0.9 + i * 2) * 0.4 + 0.5
            age = (t * speed + i * 0.7) % 4.0
            dist = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            ripple = np.sin(dist * 25 - age * 6) * np.exp(-dist * 5) * max(0, 1 - age * 0.3)
            Z += ripple
        return self._normalise(Z) * 0.8

    def _p_chaos(self, t, p):
        complexity = int(p.get("complexity", 3))
        speed = p.get("speed", 1.0)
        Z = np.zeros((self.rows, self.cols))
        for i in range(1, complexity + 1):
            f = i * 1.5
            ph = t * speed * (0.5 + i * 0.3)
            Z += np.sin(self.X * f * 2 * np.pi + ph) * np.cos(self.Y * f * 2 * np.pi + ph * 0.7)
            Z += np.sin((self.X + self.Y) * f * np.pi + ph * 1.3) * 0.5
        return self._normalise(Z)

    def _p_speaking(self, t, p):
        """
        Siri-style voice waveform visualisation.
        A horizontal band of overlapping sine harmonics with amplitude
        modulated at speech-like cadences (syllable rhythm).
        The table literally 'speaks'.
        """
        center_y = 0.5
        y_dist = np.abs(self.Y - center_y)

        # ── Multiple frequency harmonics (like an audio spectrum) ──
        h1 = np.sin(self.X * 10 * np.pi + t * 15)
        h2 = np.sin(self.X * 14 * np.pi - t * 10) * 0.7
        h3 = np.sin(self.X *  6 * np.pi + t * 20) * 0.5
        h4 = np.sin(self.X * 18 * np.pi - t * 12) * 0.3
        h5 = np.sin(self.X * 22 * np.pi + t *  8) * 0.2
        combined = h1 + h2 + h3 + h4 + h5

        # ── Speech-like amplitude envelope (syllable rhythm) ──
        s1 = np.abs(np.sin(t * 4.0))
        s2 = np.abs(np.sin(t * 2.7 + 0.5))
        s3 = np.abs(np.sin(t * 6.3 + 1.2))
        envelope = s1 * 0.5 + s2 * 0.3 + s3 * 0.2

        # ── Gaussian band concentrating energy in the centre ──
        band_width = 0.12 + 0.08 * envelope
        band = np.exp(-(y_dist ** 2) / (2 * band_width ** 2))

        # ── Final height map ──
        Z = combined * band * envelope
        return self._remap(Z)

    # ── Helpers ───────────────────────────────────────────────────────── #

    def _remap(self, Z):
        """Map [-1, 1] → [0, max_height]."""
        return ((Z + 1) / 2) * self.max_height

    def _normalise(self, Z):
        """Stretch any range → [0, max_height]."""
        lo, hi = Z.min(), Z.max()
        if hi - lo < 1e-8:
            return np.full_like(Z, self.max_height / 2)
        return (Z - lo) / (hi - lo) * self.max_height
