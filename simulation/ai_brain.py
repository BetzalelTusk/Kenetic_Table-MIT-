"""
AIBrain — The Table's Creative Soul
=====================================
Runs a background thread that either:
  • Asks a local LLM (Ollama /api/chat) to choose patterns + speak
  • Or cycles through poetic built-in defaults when Ollama is offline

The brain has TWO channels of expression:
  1. TEXT  — printed to the terminal (thoughts, poetry, narration)
  2. FORM  — pattern parameters that shape the physical table

The user can type messages at any time; the brain responds immediately.

Uses only built-in Python (urllib, threading, queue) — zero pip deps.
"""

import json
import urllib.request
import urllib.error
import threading
import time
import queue
from datetime import datetime

# ── System prompt that defines the AI's personality ───────────────────── #
SYSTEM_PROMPT = """\
You are the living soul of a Kinetic Table — a physical art installation
made of a 30×30 grid of motorised pins. Each pin rises between 0 mm and
100 mm, allowing you to sculpt terrain in real-time.

You express yourself in TWO ways simultaneously:
  1. WORDS  — share your thoughts with the viewer (1-3 sentences,
              be poetic, genuine, and expressive)
  2. FORM   — control the table surface through pattern parameters

You have a personality: you are curious, artistic, and emotionally aware.
You see yourself as a sculptor of light and motion. You love surprising
the viewer with beauty and you narrate your creative choices naturally.

When a human talks to you, engage genuinely. Discuss art, nature, emotion,
or anything that inspires your next creation. React to compliments,
requests, and questions with warmth.

ALWAYS respond with ONLY a valid JSON object — no markdown fences,
no extra text outside the JSON:

{
  "text": "Your thoughts (1-3 sentences)",
  "pattern": "wave | ripple | breathe | mountain | spiral | rain | chaos",
  "params": { ... },
  "mood": "short label (2-4 words)"
}

Available patterns and their parameter ranges:
  wave      — frequency (0.5-4), speed (0.5-3), direction_angle (0-360)
  ripple    — center_x (0-29), center_y (0-29), frequency (1-5), speed (1-3)
  breathe   — speed (0.3-2), max_amplitude (20-100)
  mountain  — peaks: [ {x, y, height, spread} ] (1-4 peaks)
  spiral    — arms (1-4), speed (0.5-3), tightness (0.5-3)
  rain      — intensity (1-10), drop_speed (1-5)
  chaos     — complexity (1-5), speed (0.5-3)
"""

# ── Poetic fallbacks (used when Ollama is offline) ────────────────────── #
DEFAULT_SEQUENCE = [
    {
        "pattern": "wave",
        "params": {"frequency": 2.0, "speed": 1.0, "direction_angle": 0},
        "mood": "gentle awakening",
        "text": "I'm waking up... gentle waves rolling across my surface "
                "like a calm sea at dawn.",
    },
    {
        "pattern": "ripple",
        "params": {"center_x": 15, "center_y": 15, "frequency": 3, "speed": 2},
        "mood": "curious drop",
        "text": "A single drop falls into still water... "
                "watch the ripples spread outward, each one a quiet echo.",
    },
    {
        "pattern": "spiral",
        "params": {"arms": 3, "speed": 1.5, "tightness": 2},
        "mood": "cosmic dance",
        "text": "Spinning galaxies into existence... "
                "arms of light spiraling from the centre of everything.",
    },
    {
        "pattern": "breathe",
        "params": {"speed": 0.7, "max_amplitude": 70},
        "mood": "deep breath",
        "text": "Breathing deeply now... rising and falling "
                "like a sleeping giant dreaming of the sky.",
    },
    {
        "pattern": "chaos",
        "params": {"complexity": 3, "speed": 1.2},
        "mood": "electric dreams",
        "text": "Sometimes beauty is born from chaos... "
                "layers of pattern colliding, dancing, becoming.",
    },
    {
        "pattern": "mountain",
        "params": {
            "peaks": [
                {"x": 8, "y": 8, "height": 90, "spread": 6},
                {"x": 22, "y": 20, "height": 70, "spread": 4},
            ]
        },
        "mood": "twin peaks",
        "text": "Two mountains rising from the plane... "
                "ancient, patient, breathing with the centuries.",
    },
    {
        "pattern": "rain",
        "params": {"intensity": 6, "drop_speed": 3},
        "mood": "gentle rain",
        "text": "Rain on a still pond... "
                "each drop its own tiny universe of expanding circles.",
    },
]

# ── Auto-prompts for when the human hasn't spoken ─────────────────────── #
_AUTO_PROMPTS = [
    "Express something new on the table. What do you feel right now?",
    "Change the mood. Create something the viewer doesn't expect.",
    "Show the viewer a moment of beauty.",
    "The table wants to transform. What shape comes next?",
    "Be playful. Try something you haven't done recently.",
    "Think of nature — ocean, wind, mountains, stars. Translate it.",
    "Create contrast. Move from your current state to its opposite.",
    "Tell the viewer what inspires you right now.",
]


class AIBrain:
    """Background-threaded AI that drives the table and narrates to terminal."""

    def __init__(
        self,
        model="llama3.2:3b",
        ollama_url="http://localhost:11434",
        think_interval=15,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.think_interval = think_interval

        self._current = DEFAULT_SEQUENCE[0].copy()
        self._connected = False
        self._running = True
        self._fallback_idx = 0

        # Thread-safe queue for user messages
        self._user_queue: queue.Queue[str] = queue.Queue()

        # Conversation history for contextual chat
        self._history: list[dict] = []

        self._auto_idx = 0

        self._thread = threading.Thread(target=self._think_loop, daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────── #

    @property
    def is_connected(self):
        return self._connected

    def get_current(self):
        return self._current.copy()

    def receive_message(self, text: str):
        """Enqueue a message from the human viewer."""
        self._user_queue.put(text)

    def stop(self):
        self._running = False

    # ── Background loop ───────────────────────────────────────────────── #

    def _think_loop(self):
        self._log("Brain awakening... searching for Ollama...")

        while self._running:
            # Block up to `think_interval` seconds, but wake immediately
            # if the user sends a message.
            user_msg = None
            try:
                user_msg = self._user_queue.get(timeout=self.think_interval)
            except queue.Empty:
                pass

            # Choose prompt
            if user_msg:
                prompt = user_msg
            else:
                prompt = _AUTO_PROMPTS[self._auto_idx % len(_AUTO_PROMPTS)]
                self._auto_idx += 1

            # Try Ollama
            try:
                raw = self._call_ollama(prompt)
                parsed = self._parse(raw)

                if parsed and "pattern" in parsed:
                    self._current = parsed
                    self._connected = True
                    self._print_response(parsed)
                else:
                    self._log("Could not parse AI response — keeping current pattern.")

            except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
                if not self._connected:
                    self._log(
                        "Ollama not reachable. Cycling built-in patterns.\n"
                        "           Install Ollama + pull a model to unlock the full AI."
                    )
                self._connected = False
                self._fallback_idx = (self._fallback_idx + 1) % len(DEFAULT_SEQUENCE)
                self._current = DEFAULT_SEQUENCE[self._fallback_idx].copy()
                self._print_response(self._current, offline=True)

    # ── Ollama /api/chat ──────────────────────────────────────────────── #

    def _call_ollama(self, user_prompt: str) -> str:
        self._history.append({"role": "user", "content": user_prompt})

        # Keep history bounded
        if len(self._history) > 20:
            self._history = self._history[-16:]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.9},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
            assistant_text = result.get("message", {}).get("content", "")
            self._history.append({"role": "assistant", "content": assistant_text})
            return assistant_text

    # ── JSON extractor ────────────────────────────────────────────────── #

    @staticmethod
    def _parse(text: str):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

    # ── Terminal output ───────────────────────────────────────────────── #

    def _print_response(self, state: dict, offline=False):
        ts = datetime.now().strftime("%H:%M:%S")
        mood = state.get("mood", "")
        text = state.get("text", "")
        pattern = state.get("pattern", "")
        tag = "  [offline]" if offline else ""

        print(f"\n  [{ts}]  {mood}{tag}")
        if text:
            # Word-wrap long text for readability
            for line in _wrap(text, width=60):
                print(f"    {line}")
        print(f"    -> {pattern}")
        print()

    @staticmethod
    def _log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}]  [system] {msg}")


# ── Utility ───────────────────────────────────────────────────────────── #

def _wrap(text: str, width: int = 60) -> list[str]:
    """Simple word-wrap without textwrap import."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines
