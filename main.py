"""
Kinetic Table Simulator — Main Entry Point
============================================
Connects four subsystems:

  1. KineticTable  (HAL)       — simulates motor physics
  2. PatternEngine             — math-driven pattern library (7 types)
  3. AIBrain                   — LLM creative director (Ollama) + terminal voice
  4. TableVisualizer           — 3D matplotlib renderer

The user can type messages in the terminal at any time.
The AI responds immediately with new patterns AND spoken text.

Run:    python main.py

Requires:
    pip install numpy matplotlib

Optional (full AI mode):
    1. Install Ollama  →  https://ollama.com
    2. Pull a model    →  ollama pull llama3.2:3b
    3. Run simulator   →  python main.py
"""

import sys
import time
import threading
import numpy as np

from simulation.hal import KineticTable
from simulation.visualizer import TableVisualizer
from simulation.patterns import PatternEngine
from simulation.ai_brain import AIBrain

# ── Configuration ─────────────────────────────────────────────────────── #
GRID_SIZE    = (30, 30)
MAX_HEIGHT   = 100.0      # mm
MAX_SPEED    = 50.0       # mm/s  (motor speed limit)
AI_MODEL     = "llama3.2:3b"
AI_INTERVAL  = 15         # seconds between autonomous AI thoughts


# ── User Input Thread ─────────────────────────────────────────────────── #
def _input_loop(ai: AIBrain):
    """
    Background thread that reads user input and forwards it to the AI.
    Runs as a daemon — killed automatically when the main thread exits.
    """
    while True:
        try:
            text = input("You > ")
            if text.strip():
                ai.receive_message(text.strip())
        except (EOFError, KeyboardInterrupt):
            break


# ── Main Loop ─────────────────────────────────────────────────────────── #
def main():
    # 1 ── Hardware Abstraction Layer ──
    table = KineticTable(
        grid_size=GRID_SIZE,
        max_height=MAX_HEIGHT,
        max_speed=MAX_SPEED,
    )

    # 2 ── Pattern Engine ──
    patterns = PatternEngine(
        grid_size=GRID_SIZE,
        max_height=MAX_HEIGHT,
    )

    # 3 ── AI Brain (creative director + terminal voice) ──
    ai = AIBrain(
        model=AI_MODEL,
        think_interval=AI_INTERVAL,
    )

    # 4 ── 3D Visualizer ──
    visualizer = TableVisualizer(
        grid_size=GRID_SIZE,
        max_height=MAX_HEIGHT,
    )

    # 5 ── User input thread ──
    input_thread = threading.Thread(target=_input_loop, args=(ai,), daemon=True)
    input_thread.start()

    start_time = time.time()
    running = True

    print(
        "\n"
        "  ╔════════════════════════════════════════════════════════════╗\n"
        "  ║            Kinetic Table Simulator  v2.0                  ║\n"
        "  ╠════════════════════════════════════════════════════════════╣\n"
       f"  ║  Grid : {GRID_SIZE[0]}x{GRID_SIZE[1]}   "
       f"|  Height : 0-{int(MAX_HEIGHT)} mm  "
       f"|  Motor : {int(MAX_SPEED)} mm/s    ║\n"
        "  ║  AI   : {:<49}║\n".format(f"{AI_MODEL} via Ollama") +
        "  ╠════════════════════════════════════════════════════════════╣\n"
        "  ║  The AI speaks in the terminal and sculpts the table.     ║\n"
        "  ║  Type a message below to talk to it at any time.          ║\n"
        "  ║  Close the 3D window or press Ctrl+C to quit.             ║\n"
        "  ╚════════════════════════════════════════════════════════════╝\n"
    )

    try:
        while running:
            elapsed = time.time() - start_time

            # ── Read AI's current creative direction ──
            ai_state = ai.get_current()

            # ── When the AI is "speaking", override with voice waveform ──
            if ai.is_speaking:
                patterns.set_pattern("speaking", {})
                mood = "Speaking..."
            else:
                patterns.set_pattern(
                    ai_state.get("pattern", "wave"),
                    ai_state.get("params", {}),
                )
                mood = ai_state.get("mood", "")

            # ── Generate target heights from pattern engine ──
            target = patterns.generate(elapsed)
            table.set_target(target)

            # ── Simulate motor physics ──
            table.update()

            # ── Render 3D view ──
            if not visualizer.update_pins(table.current_heights, mood_text=mood):
                running = False

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        ai.stop()
        visualizer.close()
        print("  Simulator closed. Goodbye.\n")


# ── Entry Point ───────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()
