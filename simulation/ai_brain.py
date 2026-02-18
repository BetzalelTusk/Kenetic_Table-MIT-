"""
AIBrain — Ollama-powered Creative Director
============================================
Runs a background thread that asks a local LLM (via Ollama) to choose
moods and pattern parameters every few seconds.

If Ollama is not running, the brain gracefully falls back to cycling
through built-in default patterns so the table is never idle.

Communicates with Ollama via its REST API (no extra pip packages needed).
"""

import json
import urllib.request
import urllib.error
import threading
import time

# ── System prompt that defines the AI's "personality" ─────────────────── #
SYSTEM_PROMPT = """\
You are the creative soul of a Kinetic Table — a physical art installation
made of a 30×30 grid of motorized pins, each rising from 0 mm to 100 mm.

Every few seconds you choose what the table should express.
Respond ONLY with a single JSON object (no markdown, no extra text).

Available pattern types and their parameters:

  wave      — frequency (0.5-4), speed (0.5-3), direction_angle (0-360)
  ripple    — center_x (0-29), center_y (0-29), frequency (1-5), speed (1-3)
  breathe   — speed (0.3-2), max_amplitude (20-100)
  mountain  — peaks: list of {x, y, height, spread} (1-4 peaks)
  spiral    — arms (1-4), speed (0.5-3), tightness (0.5-3)
  rain      — intensity (1-10), drop_speed (1-5)
  chaos     — complexity (1-5), speed (0.5-3)

Example response:
{"pattern":"wave","params":{"frequency":2,"speed":1.5,"direction_angle":45},"mood":"calm ocean breeze"}
"""

# What we use when the AI hasn't spoken yet or Ollama is offline
DEFAULT_SEQUENCE = [
    {"pattern": "wave",    "params": {"frequency": 2.0, "speed": 1.0, "direction_angle": 0},   "mood": "gentle startup"},
    {"pattern": "ripple",  "params": {"center_x": 15, "center_y": 15, "frequency": 3, "speed": 2}, "mood": "drop in a pond"},
    {"pattern": "spiral",  "params": {"arms": 3, "speed": 1.5, "tightness": 2},                "mood": "cosmic swirl"},
    {"pattern": "breathe", "params": {"speed": 0.7, "max_amplitude": 70},                      "mood": "deep breath"},
    {"pattern": "chaos",   "params": {"complexity": 3, "speed": 1.2},                          "mood": "electric dreams"},
]


class AIBrain:
    def __init__(self, model="llama3.2:3b", ollama_url="http://localhost:11434",
                 think_interval=12):
        self.model = model
        self.ollama_url = ollama_url
        self.think_interval = think_interval

        self._current = DEFAULT_SEQUENCE[0].copy()
        self._connected = False
        self._running = True
        self._fallback_idx = 0

        self._thread = threading.Thread(target=self._think_loop, daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────── #

    @property
    def is_connected(self):
        return self._connected

    def get_current(self):
        return self._current.copy()

    def stop(self):
        self._running = False

    # ── Background thinking loop ──────────────────────────────────────── #

    def _think_loop(self):
        print("[AI Brain] Starting creative thinking loop...")

        prompts = [
            "Choose a pattern. Express something beautiful.",
            "Show something new. Surprise the viewer.",
            "Create a moment of calm.",
            "Express energy and excitement.",
            "The table is feeling playful. What fits?",
            "Think of nature — ocean, mountains, rain. Translate it.",
            "Be abstract. Create a feeling, not a picture.",
        ]
        idx = 0

        while self._running:
            try:
                response_text = self._call_ollama(prompts[idx % len(prompts)])
                idx += 1
                parsed = self._parse(response_text)

                if parsed and "pattern" in parsed:
                    self._current = parsed
                    self._connected = True
                    mood = parsed.get("mood", "—")
                    print(f'[AI Brain] New mood: "{mood}" → {parsed["pattern"]}')
                else:
                    print("[AI Brain] Could not parse response — keeping current pattern.")

            except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as exc:
                if not self._connected:
                    print(f"[AI Brain] Ollama not reachable ({type(exc).__name__}). "
                          f"Cycling built-in patterns.")
                self._connected = False
                # Cycle through defaults so the table stays alive
                self._fallback_idx = (self._fallback_idx + 1) % len(DEFAULT_SEQUENCE)
                self._current = DEFAULT_SEQUENCE[self._fallback_idx].copy()
                mood = self._current.get("mood", "—")
                print(f'[AI Brain] Fallback → "{mood}" ({self._current["pattern"]})')

            time.sleep(self.think_interval)

    # ── Ollama REST call ──────────────────────────────────────────────── #

    def _call_ollama(self, prompt):
        body = json.dumps({
            "model":  self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.9},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "")

    # ── JSON extractor ────────────────────────────────────────────────── #

    @staticmethod
    def _parse(text):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None
