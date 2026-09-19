# Project 5 — Reinforcement Learning-Based Adaptive MCS

## Objective

Investigate whether reinforcement learning can learn
adaptive modulation decisions under changing link conditions.

## State

Discretized link-margin state:

- S0 — low margin
- S1 — medium margin
- S2 — high margin

## Actions

- QPSK
- 16-QAM
- 64-QAM
- Link-Off

## Algorithm

Tabular Q-learning

## Reward

The reward balances:

- successful high-throughput operation
- robust operation under degraded conditions
- outage avoidance

## Evaluation

- MCS distribution
- outage frequency
- effective throughput
- behavior on unseen passes