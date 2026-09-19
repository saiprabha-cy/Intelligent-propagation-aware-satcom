# Project 6 — Hybrid MPC-RL Adaptive Satellite Link Control

## Objective

Combine predictive model-based control with reinforcement
learning while constraining the available actions.

## Architecture

Propagation / Geometry Model
          ↓
       MPC Layer
          ↓
   Safe Action Set
          ↓
       RL Policy
          ↓
     MCS Decision

## MPC role

- Predict future link margin
- Evaluate feasibility
- Enforce communication constraints
- Mask unsafe actions

## RL role

Select a performance-oriented action from the
MPC-approved action set.

## Evaluation

Compared against:

- Rule-based control
- Pure RL

Metrics:

- MCS distribution
- Outage rate
- Stability
- Link-margin behavior

## Scope

Simulation-based research prototype.