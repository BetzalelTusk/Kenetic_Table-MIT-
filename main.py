"""
Kinetic Table Simulator — Main Entry Point
============================================
Connects four subsystems:

  1. KineticTable (HAL)    — simulates motor physics
  2. PatternEngine         — math-driven pattern library
  3. AIBrain               — LLM creative director (Ollama)
  4. TableVisualizer       — 3D matplotlib renderer

Run:
    python main.py

Requirements:
    pip install numpy matplotlib

Optional (for AI-driven patterns):
    1. Install Ollama        →  https://ollama.com
    2. Pull a model          →  ollama pull llama3.2:3b
    3. The simulator auto-detects Ollama and switches to AI mode.
"""

import time
import numpy as np

from simulation.hal import KineticTable
from simulation.visualizer import TableVisualizer
from simulation.patterns import PatternEngine
from simulation.ai_brain import AIBrain

# ── Configuration ─────────────────────────────────────────────────────── #
GRID_SIZE   = (30, 30)
MAX_HEIGHT  = 100.0     # mm
MAX_SPEED   = 50.0      # mm/s  (motor speed limit)
AI_MODEL    = "llama3.2:3b"
AI_INTERVAL = 12        # seconds between AI "thoughts"


# ── Main Loop ─────────────────────────────────────────────────────────── #
def main():
    # 1 ── Hardware Abstraction Layer (the "digital twin") ──
    table = KineticTable(
        grid_size=GRID_SIZE,
        max_height=MAX_HEIGHT,
        max_speed=MAX_SPEED,
    )

    # 2 ── Pattern Engine (the math behind the art) ──
    patterns = PatternEngine(
        grid_size=GRID_SIZE,
        max_height=MAX_HEIGHT,
    )

    # 3 ── AI Brain (creative director — optional) ──
    ai = AIBrain(
        model=AI_MODEL,
        think_interval=AI_INTERVAL,
    )

    # 4 ── 3D Visualizer ──
    visualizer = TableVisualizer(
        grid_size=GRID_SIZE,
        max_height=MAX_HEIGHT,
    )

    start_time = time.time()
    running = True

    print("══════════════════════════════════════════════════")
    print("  Kinetic Table Simulator  v2.0")
    print(f"  Grid : {GRID_SIZE[0]}×{GRID_SIZE[1]}   "
          f"|  Max Height : {int(MAX_HEIGHT)} mm  "
          f"|  Motor Speed : {int(MAX_SPEED)} mm/s")
    print(f"  AI Model : {AI_MODEL}  (via Ollama)")
    print("  Close the window or press Ctrl+C to quit.")
    print("══════════════════════════════════════════════════")

    try:
        while running:
            elapsed = time.time() - start_time

            # ── Ask the AI what to express ──
            ai_state = ai.get_current()
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
        print("\nInterrupted by user.")
    finally:
        ai.stop()
        visualizer.close()
        print("Simulator closed.")


# ── Entry Point ───────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()
