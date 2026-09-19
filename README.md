# Intelligent Propagation-Aware Adaptive Satellite Communication Systems

**Physics-Aware RF Propagation Modeling → LEO Link Analysis → Adaptive Communication → Reinforcement Learning → Hybrid Predictive Control**

This repository contains a six-stage research progression investigating how propagation-aware modeling and intelligent control can be used to improve the adaptability of dynamic Low Earth Orbit (LEO) satellite communication links.

The work combines classical satellite communication and RF propagation concepts with machine learning, adaptive link-margin estimation, reinforcement learning, and model predictive control.

---

## Research Objective

Satellite communication links experience continuously changing propagation conditions due to:

* Satellite motion and changing elevation angle
* Slant-range variation
* Free-space path loss
* Atmospheric and environmental attenuation
* Frequency-dependent propagation effects
* Time-varying link margin
* Power and spectral-efficiency constraints

The central research question is:

> **How can satellite communication links be modeled and controlled intelligently while preserving physical interpretability, robustness, and operational constraints?**

The research therefore follows a progressive architecture rather than treating each project as an isolated experiment.

---

# Research Progression

```text
                 LEO Satellite Geometry
                         │
                         ▼
                  Slant Range / Time
                         │
                         ▼
              Free-Space Path Loss
                         │
                         +
                         │
                         ▼
                Excess Path Loss
                         │
                         ▼
             ML Propagation Modeling
                         │
                         ▼
              Multiband S/C/X Model
                         │
                         ▼
                  Link Budget
                         │
                         ▼
                    Received Power
                         │
                         ▼
                        SNR
                         │
                         ▼
                   Link Margin
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Rule-Based MCS          RL-Based MCS
              │                     │
              └──────────┬──────────┘
                         ▼
                   Hybrid MPC-RL
                         │
                         ▼
             Adaptive Satellite Link
```

---

# Six-Project Research Structure

## Project 1 — ML-Based Excess Path Loss Modeling

**Directory:** `project_1_rf_propagation_study`

### Objective

Develop a physics-aware machine-learning approach for predicting **excess path loss** while retaining the deterministic free-space path-loss component.

Instead of asking a machine-learning model to learn the complete propagation loss directly:

```text
Total Loss = FSPL + Excess Loss
```

FSPL is calculated analytically, while the ML model estimates the excess component.

### Main variables

* Elevation angle
* Slant range
* Carrier frequency
* Time index

### Models investigated

* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression
* Feedforward Neural Network

### Output

Predicted excess path loss and associated prediction error.

### Why it matters

The predicted excess-loss component becomes a propagation-aware input for the later satellite-link adaptation stages.

---

# Project 2 — LEO Spatio-Temporal Propagation Modeling

**Directory:** `project_2_leo_satellite_propagation_ml`

### Objective

Extend the propagation model from individual/static conditions to the changing geometry of a LEO satellite pass.

A LEO communication link does not remain constant. As the satellite moves:

```text
Elevation changes
       ↓
Slant range changes
       ↓
Propagation loss changes
       ↓
Link margin changes
       ↓
Communication conditions change
```

### State/features

The research models a time-dependent state involving:

* Elevation angle
* Slant range
* Carrier frequency
* Time
* Slant-range rate

### Output

A time-varying propagation representation of a LEO satellite pass.

### Why it matters

This provides the temporal/geometric foundation required for predictive link adaptation and later reinforcement-learning control.

---

# Project 3 — Multiband LEO Propagation Modeling

**Directory:** `project_3_leo_multiband_propagation_ml`

### Objective

Extend the propagation analysis across:

* S-band
* C-band
* X-band

The research investigates how frequency and environmental conditions influence excess propagation loss during LEO communication.

### Inputs

* Elevation angle
* Slant range
* Carrier frequency
* Rain rate
* Relative humidity
* Ambient temperature
* Time

### ML approach

Random Forest regression is used to model nonlinear relationships between propagation conditions and excess loss.

### Outputs

* Excess-loss prediction
* Prediction error
* Cross-band comparison
* Feature importance

### System-level purpose

The resulting propagation estimates can be used by later adaptive communication stages for decisions involving modulation/coding and frequency-dependent link behavior.

---

# Project 4 — Adaptive Satellite Link Margin and Rule-Based MCS

**Directory:** `project_4_adaptive_satellite_link_system`

### Objective

Move from propagation prediction to **communication-link decision making**.

The predicted propagation loss is incorporated into a simplified satellite link budget:

```text
Pr = Pt + Gt + Gr
     - LFSPL
     - Lexcess
     - Lsystem
```

Then:

```text
SNR = Pr - Pnoise
```

and:

```text
Link Margin = SNR - Required SNR
```

### Link states

The simulation classifies the link into:

* Good
* Degraded
* Outage

### Rule-based modulation selection

The link margin is mapped to communication states such as:

```text
High margin      → 64-QAM
Moderate margin  → 16-QAM
Low margin       → QPSK
Very low margin  → Link-Off
```

### Why this stage exists

This provides a transparent, deterministic baseline before introducing reinforcement learning.

---

# Project 5 — Reinforcement Learning-Based Adaptive MCS

**Directory:** `project_5_rl_adaptive_satellite_link`

### Objective

Investigate whether reinforcement learning can learn adaptive modulation decisions under changing satellite-link conditions instead of relying exclusively on fixed thresholds.

### RL formulation

The problem is represented as a Markov Decision Process:

```text
State
  ↓
Action
  ↓
Communication outcome
  ↓
Reward
  ↓
Updated policy
```

### State

The instantaneous link margin is discretized into three operating conditions:

* Low-margin / outage-prone
* Medium-margin / degraded
* High-margin / good

### Actions

* QPSK
* 16-QAM
* 64-QAM
* Link-Off

### Algorithm

**Tabular Q-learning**

### Evaluation

The research evaluates:

* MCS selection behavior
* Outage frequency
* Effective throughput
* Behavior on unseen simulated passes

### Why this stage matters

Project 5 transforms the problem from:

> "What modulation should a fixed rule select?"

into:

> "Can an agent learn a policy for selecting the communication state over time?"

---

# Project 6 — Hybrid MPC-RL Adaptive Satellite Link Control

**Directory:** `project_6_hybrid_mpc_rl_satellite_link`

### Objective

Combine:

* Model Predictive Control
* Reinforcement Learning

into a hierarchical adaptive-link architecture.

The motivation is to combine predictive model-based reasoning with data-driven adaptation while restricting unsafe actions.

### Architecture

```text
LEO Geometry
     │
     ▼
Propagation Model
     │
     ▼
Future Link-Margin Prediction
     │
     ▼
       MPC
     │
     ▼
Safe / Feasible Action Set
     │
     ▼
       RL
     │
     ▼
MCS Decision
```

### MPC role

The predictive controller evaluates future link conditions and applies constraints such as:

* Minimum required SNR
* Predicted link feasibility
* Conservative operation during predicted degradation

### RL role

The reinforcement-learning policy selects a performance-oriented action from the actions permitted by the predictive constraint layer.

### Evaluation

The hybrid approach is studied against:

* Rule-based adaptation
* Pure RL adaptation

using metrics including:

* MCS behavior
* Link-margin evolution
* Outage behavior
* Stability

---

# End-to-End Research Flow

The six projects can therefore be understood as one continuous engineering progression:

```text
PROJECT 1
Excess Path Loss Prediction
          │
          ▼
PROJECT 2
LEO Time + Geometry
          │
          ▼
PROJECT 3
Multiband Propagation
          │
          ▼
PROJECT 4
Link Budget + Link Margin
          │
          ▼
Rule-Based MCS
          │
          ▼
PROJECT 5
RL-Based MCS
          │
          ▼
PROJECT 6
MPC-Constrained RL
          │
          ▼
Adaptive Satellite Communication
```

The individual projects are therefore not independent ML exercises. Each stage supplies concepts or outputs used by the next stage.

---

# Research Scope and Validation Status

This repository represents **simulation-based research and algorithmic prototyping**.

The work uses simulated/semi-empirical propagation data and simplified communication/control models.

It should **not** be interpreted as:

* Flight-qualified software
* Mission-qualified satellite software
* A flight-tested communication controller
* Hardware-validated modem firmware
* An operational ISRO communication system
* A certified aerospace control system

Hardware-in-the-loop, SDR implementation, real atmospheric measurements, continuous-state reinforcement learning, uncertainty-aware robust MPC, and multi-satellite coordination are identified as directions for future work.

---

# Technical Areas

* Satellite Communications
* RF Propagation
* Link Budget Analysis
* Free-Space Path Loss
* Excess Path Loss Modeling
* LEO Satellite Geometry
* S/C/X-Band Propagation
* Machine Learning
* Regression
* Adaptive Modulation
* Link Margin Estimation
* Reinforcement Learning
* Q-Learning
* Model Predictive Control
* Hybrid Intelligent Control
* Aerospace Communication Systems

---

# Research Applications

The framework is investigated in the context of potential applications such as:

* Satellite telemetry and communications
* Payload data downlinks
* Ground-station adaptive scheduling
* Small-satellite communication systems
* LEO satellite constellations
* Propagation-aware link adaptation

These are research/application areas rather than claims of deployment on any specific operational spacecraft.

---

# Repository Organization

```text
Intelligent-propagation-aware-satcom/
│
├── project_1_rf_propagation_study/
│
├── project_2_leo_satellite_propagation_ml/
│
├── project_3_leo_multiband_propagation_ml/
│
├── project_4_adaptive_satellite_link_system/
│
├── project_5_rl_adaptive_satellite_link/
│
└── project_6_hybrid_mpc_rl_satellite_link/
```

Each project directory contains the corresponding implementation, analysis, and research artifacts for that stage.

---

# Relationship Between Physics and Machine Learning

A central design principle throughout the research is to avoid replacing known physical relationships unnecessarily.

For example:

```text
Known physics
     │
     ├── FSPL
     ├── LEO geometry
     ├── Link-budget relationships
     └── Communication constraints
             │
             ▼
       ML / RL / MPC
             │
             ▼
     Adaptive decision making
```

The intention is therefore not "AI instead of communication theory."

It is:

> **Communication theory + physical modeling + machine learning + adaptive control.**

---

# Research Limitations

The current research has several limitations:

1. Propagation data are simulated/semi-empirical rather than a comprehensive real-world measurement campaign.
2. The RL formulation uses a simplified discrete state space.
3. The MCS action space represents modulation choices rather than a complete standardized coding/modulation implementation.
4. The MPC formulation is simplified.
5. Hardware and SDR validation have not yet been performed.
6. Real atmospheric/beacon datasets have not yet been integrated into the complete framework.
7. The work does not constitute flight or mission qualification.

These limitations define the boundary between the current research prototype and a future hardware/operational implementation.

---

# Future Research Directions

Potential extensions include:

* Continuous-state deep reinforcement learning
* Uncertainty-aware robust MPC
* Real atmospheric and beacon datasets
* Software-defined-radio implementation
* Hardware-in-the-loop validation
* Multi-satellite coordination
* Multi-ground-station optimization
* Inter-satellite-link integration
* Optical communication extensions

---

## Author

**SaiPrabha C Y**

Electronics & Communication Engineering
Independent Research — Satellite Communications, RF Propagation, Machine Learning and Adaptive Control

---

## Disclaimer

This repository is an independent research and simulation project.

References to satellite missions, space organizations, or potential aerospace applications describe the technical context or possible applicability of the research and do not imply employment, collaboration, endorsement, mission deployment, or access to proprietary information from those organizations.
