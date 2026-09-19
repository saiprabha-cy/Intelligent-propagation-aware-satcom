# Project 4 — Adaptive Link Margin and Rule-Based MCS

## Objective

Use predicted propagation loss inside a link budget to
estimate link margin and adapt the modulation state.

## Link budget

Pr = Pt + Gt + Gr - LFSPL - Lexcess - Lsys

SNR = Pr - Pn

Margin = SNR - SNRreq

## Link states

- Good
- Degraded
- Outage

## Adaptive modulation

Link margin is mapped to:

- 64-QAM
- 16-QAM
- QPSK
- Link-Off

## Purpose

Establish a transparent rule-based baseline before introducing
reinforcement learning.