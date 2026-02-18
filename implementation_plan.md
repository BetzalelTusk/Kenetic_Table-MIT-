# Implementation Plan - Kinetic Table Simulator

This plan outlines the steps to build the Hardware Abstraction Layer (HAL) and Visual Simulator for the Kinetic Table project.

## User Requirements
- **Hardware Simulation**: Grid of motorized pins (e.g., 30x30).
- **Actuation**: Linear actuators, 0-100mm height.
- **Physics**: Pins move at a finite speed (no teleportation).
- **Visualization**: 3D representation of the grid (using Ursina Engine for best performance/ease of use).
- **Capability**: Support for organic shapes ("hills"), not just binary on/off.

## Proposed Architecture
1.  **`KineticTable` Class (HAL)**:
    - Manages the state of the table (Current Heights vs Target Heights).
    - Handles the "physics" update tick (moving pins towards targets based on speed limits).
    - Uses `numpy` for efficient matrix operations.

2.  **`TableVisualizer` (UI)**:
    - Uses `ursina` to render the grid.
    - Updates the visual representation of pins (scaling/positioning) based on the HAL's current state.

3.  **Controller/AI Placeholder**:
    - A simple script to generate patterns (waves or noise) that feed target heights into the simulation.

## Steps
- [ ] Create `requirements.txt`
- [ ] Implement `simulation/hal.py` containing the `KineticTable` class.
- [ ] Implement `simulation/visualizer.py` containing the Ursina application.
- [ ] Create a `main.py` entry point that ties the HAL, Visualizer, and a demo pattern generator together.
