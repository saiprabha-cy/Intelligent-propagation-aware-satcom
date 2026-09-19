# Project 1 — ML-Based Excess Path Loss Modeling

## Objective

Develop a physics-aware ML model that predicts excess
propagation loss while calculating free-space path loss
analytically.

## Core formulation

L_total = L_FSPL + L_excess

The ML model predicts:

L_excess = f(elevation, frequency, slant_range, time)

## Inputs

- Elevation angle
- Slant range
- Carrier frequency
- Time index

## Approach

1. Calculate FSPL analytically
2. Construct excess-loss target
3. Normalize features
4. Train regression models
5. Evaluate prediction error
6. Compare behavior across elevation/frequency

## Models

- Linear Regression
- Random Forest
- Gradient Boosting
- Feedforward Neural Network

## Output

Predicted excess path loss and associated prediction error.

## Relevance

The predicted excess loss becomes an input to subsequent
adaptive satellite-link modeling and control stages.