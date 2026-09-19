# Project 2 — LEO Spatio-Temporal Propagation Modeling

## Objective

Model how satellite communication-link geometry and
propagation conditions evolve during a LEO pass.

## Key variables

- Elevation angle
- Slant range
- Carrier frequency
- Time
- Slant-range rate

## Model

x(t) = [θ(t), d(t), fc, t, d_dot(t)]

## Main outputs

- Time-varying elevation
- Time-varying slant range
- Propagation variation
- Excess-loss estimates

## Why it matters

Unlike a static link budget, a LEO link changes continuously
during a satellite pass.