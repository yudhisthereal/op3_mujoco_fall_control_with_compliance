# OP3 Humanoid Fall Control: Robust RL on Morphologically Dynamic Robots

A MuJoCo + Gymnasium framework for studying compliant fall recovery in the ROBOTIS OP3 humanoid robot. This project explores how **adaptive joint stiffness/damping** can improve impact mitigation and energy absorption during forward falls.

## Overview

When a humanoid robot loses balance, rigid control can cause dangerous impact spikes. This framework investigates **compliance-after-goal-reaching**: after the robot drives its arms to a protective pose, it temporarily softens joint stiffness and increases damping to absorb fall impact more gracefully.

### Key Features
- **20 ms control timestep** with MuJoCo physics and Gymnasium API
- **Gain interface**: scale stiffness (Kp) and damping (Kd) with clamped ranges
- **Impact detection**: spike detection on net disturbance torque (subtracting gravity/Coriolis bias)
- **Goal-reaching gate**: moderate threshold (mean error < 0.12 rad, max error < 0.20 rad for 5 consecutive steps)
- **Gain restoration**: automatic linear ramp-back to stiff gains after loads subside
- **10-run experiments**: alternating stiff vs. compliant-after-goal trials
- **Multi-metric logging**: tracks `qfrc_actuator`, `qfrc_constraint`, `actuator_force`, and net disturbance
- **Envelope plots**: cohort comparison (stiff vs. compliant) with median ± percentile bands

## Project Structure

```
.
├── main.py                          # Main experiment runner (20ms control loop, 10 runs, plotting)
├── robotis_op3/
│   ├── op3.xml                      # OP3 model (20 actuators, floating base)
│   ├── scene.xml                    # Scene with camera setup (azimuth=160°, elevation=-20°)
│   └── assets/simplified_convex/    # Collision meshes
├── tutorial.ipynb                   # Reference notebook (camera setup, basic simulation)
├── abstraction.txt                  # High-level project roadmap
└── outputs/                         # Generated plots (created at runtime)
    ├── run_00_loads.png             # Per-run torque time-series + event markers
    ├── run_01_loads.png
    ├── ...
    └── envelope_stiff_vs_compliant.png  # Stiff vs. compliant cohort envelopes
```

## Setup: Obtaining the OP3 Model

The `robotis_op3/` folder contains the MJCF model, XML scene, and collision mesh assets. You can obtain these from the **Google DeepMind MuJoCo Menagerie** repository in two ways:

### Option 1: Clone Entire Repo & Copy Folder (Simpler)
```bash
# Clone the full menagerie repo (may be large)
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# Copy just the OP3 folder to your project root
cp -r mujoco_menagerie/robotis_op3 /path/to/op3_mujoco_arm_to_fall_direction/
```

### Option 2: Sparse Checkout (Efficient, Only Downloads OP3)
```bash
# Create a new workspace directory
mkdir mujoco_menagerie_sparse
cd mujoco_menagerie_sparse

# Initialize git repo with sparse-checkout
git init
git config core.sparsecheckout true

# Add remote (don't fetch yet)
git remote add origin https://github.com/google-deepmind/mujoco_menagerie.git

# Set sparse checkout path
echo "robotis_op3/" >> .git/info/sparse-checkout

# Fetch only the robotis_op3 folder
git fetch --depth 1 origin main

# Checkout
git checkout origin/main

# Copy to your project
cp -r robotis_op3 /path/to/op3_mujoco_arm_to_fall_direction/
```

> **Tip**: Option 2 downloads ~50 MB instead of the full ~500+ MB repo. Use this if bandwidth is a concern.

After setup, verify the structure:
```bash
ls robotis_op3/
# Expected output: CHANGELOG.md, LICENSE, README.md, op3.xml, scene.xml, assets/
```

## Running the Experiment

### Prerequisites
```bash
pip install mujoco gymnasium matplotlib numpy
```

### Execute
```bash
python main.py
```

**Duration**: ~5 seconds per run × 10 runs ≈ 50–60 seconds total (plus plot generation).

**Output**: All plots saved to `outputs/` directory.

## What Happens in Each Run

1. **Robot Reset** at standing pose with floating base.
2. **Initial Push** (0.08 s, 35 N forward force on head) → simulates unexpected fall trigger.
3. **Goal Tracking** (stiff gains, Kp_scale=1.0, Kd_scale=1.0):
   - All controlled joints drive toward `goal_angles` (protective arm pose).
   - Tracked: `qfrc_actuator`, `qfrc_constraint`, `actuator_force` (net of gravity/Coriolis bias).
4. **Goal-Reached Gate** (moderate criterion):
   - If 5 consecutive steps satisfy: mean error ≤ 0.12 rad, max error ≤ 0.20 rad, mean speed ≤ 0.8 rad/s → **GOAL REACHED**.
5. **Compliance Switch** (odd runs only, e.g., runs 1, 3, 5, 7, 9):
   - Kp drops to 0.45, Kd rises to 1.8 (softer, damped joints).
   - Even runs (0, 2, 4, 6, 8) stay stiff throughout.
6. **Impact Detection**:
   - Rolling window (10 steps) computes mean/std of net disturbance.
   - Spike = load > μ + 4σ AND jump ratio > 1.25 (with 5-step refractory).
7. **Gain Restoration** (compliant runs only):
   - Triggered when no spike for 20 steps AND load < 1.2× baseline for 10 consecutive steps.
   - Linear ramp over 50 steps (1.0 s) back to stiff gains.

## Metrics & Interpretation

### Net Disturbance Torque
$$\tau_{dist} = | \text{qfrc\_constraint} | + | \text{qfrc\_actuator} - \text{qfrc\_bias} |$$

- **qfrc_constraint**: external & contact forces transmitted to joints.
- **qfrc_actuator - qfrc_bias**: actuator demand minus gravity/Coriolis; isolates active and impact-driven loads.
- **qfrc_bias** (subtracted): static balance torque; not a "disturbance."

### Per-Run Plots
Four stacked time-series (timestep on x-axis):
1. `qfrc_actuator (L1)` — blue
2. `qfrc_constraint (L1)` — orange
3. `actuator_force (L1)` — green
4. `total_load` — red

**Vertical event markers**:
- **Black dashed (–)**: goal reached
- **Purple dotted (:)**: compliance ON
- **Brown dash-dot (-.)**: gain restoration started

### Envelope Plots
A 4×2 grid comparing stiff vs. compliant-after-goal cohorts:
- **Rows**: one metric each (`qfrc_actuator`, `qfrc_constraint`, `actuator_force`, `total_load`)
- **Columns**: stiff (left, blue), compliant (right, purple)
- **Bands**: median line + shaded P10–P90 percentile range
- **Aligned y-scales** within each metric row for direct visual comparison

### Terminal Summary
```
Saved plots to: /path/to/outputs
Median peak total_load (stiff/compliant): X / Y
Median AUC total_load (stiff/compliant): A / B
```

## Configuration

Edit these dataclasses in [main.py](main.py) to tune behavior:

```python
@dataclass
class GainConfig:
    kp_scale_stiff: float = 1.0           # Stiff stiffness scale
    kd_scale_stiff: float = 1.0           # Stiff damping scale
    kp_scale_compliant: float = 0.45      # Compliant stiffness scale
    kd_scale_compliant: float = 1.8       # Compliant damping scale
    kp_scale_min: float = 0.2             # Min clamp
    kp_scale_max: float = 1.5             # Max clamp
    kd_scale_min: float = 0.5
    kd_scale_max: float = 3.0

@dataclass
class GoalGateConfig:
    mean_err_thr: float = 0.12            # Mean error threshold
    max_err_thr: float = 0.20             # Max error threshold
    mean_qvel_thr: float = 0.8            # Mean speed threshold
    consecutive_steps: int = 5            # Steps to hold before "reached"

@dataclass
class ImpactConfig:
    rolling_window: int = 10              # Baseline window size
    sigma_multiplier: float = 4.0         # Spike threshold (μ + k*σ)
    jump_ratio: float = 1.25              # Load jump ratio
    refractory_steps: int = 5             # Refractory period

@dataclass
class RestoreConfig:
    no_spike_steps: int = 20              # Steps without spike before restore
    low_load_steps: int = 10              # Steps in low-load regime
    baseline_multiplier: float = 1.2      # Baseline threshold
    ramp_steps: int = 50                  # Restoration ramp duration
```

## Future Roadmap

### Phase 2: Gymnasium Environments
- `Op3FallControlArms` (arms only, scope v1)
- `Op3FallControlArmsLegs` (arms + legs)
- `Op3FallControlRoll` (lateral fall)
- Register in gymnasium with `gymnasium.make('Op3FallControl-v0')`

### Phase 3: Robust RL
- **Domain randomization**: limb mass perturbations, joint offsets, friction
- **Curriculum learning**: phases with incremental noise/randomization
- **Parallel population training**: multiple agents, shared policy

### Phase 4: Advanced Features
- IMU noise perturbation
- Protective pose learning (vs. fixed `goal_angles`)
- Post-impact recovery policy
- Fall direction estimation (F/B/L/R/diagonal)

## Camera Setup

Matches [tutorial.ipynb](tutorial.ipynb) and [robotis_op3/scene.xml](robotis_op3/scene.xml):
- **Azimuth**: 160° (slightly rotated toward right shoulder)
- **Elevation**: –20° (tilted down)
- **Distance**: 1.5 m from lookat point
- **Lookat**: (0, 0, 0.2) — chest height

Human render mode uses MuJoCo's passive viewer with real-time camera sync.

## References

- **OP3 Model**: ROBOTIS OP3 humanoid (20 1-DoF joints + floating base)
- **MuJoCo**: Physics simulation with position actuators (Kp=21.1 default)
- **Gymnasium**: Standard RL environment API (v0.26+)

## Notes

- **RNG Seeds**: Deterministic per run (`base_seed + run_idx`)
- **Rendering**: `render_mode=None` in code (set to `"human"` for live viewing; slows execution ~2–3×)
- **Gain application**: Scales position-actuator gainprm + dof_damping arrays at runtime
- **Impact detection**: Adaptive threshold per run; refractory period avoids repeated false positives

---

**Status**: Core experiment framework complete. Ready for Phase 2 (Gymnasium registration) and Phase 3 (RL policy training).
