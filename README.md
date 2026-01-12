# Robotic Pick-and-Place with Imitation Learning and Offline Reinforcement Learning

This repository contains the code and experimental setup for a robotic manipulation study
based on the **robosuite** simulation platform and **MuJoCo** physics engine.  
The project investigates how difficult it is to train a robotic arm to perform a simple
pick-and-place task in a controlled simulation environment using different learning paradigms.

## Overview

- **Environment**: robosuite Pick-and-Place task (Panda robot)
- **Simulation**: MuJoCo
- **Task**: Pick a bread object from a source bin and place it into a target bin
- **Control**: Operational Space Control (OSC\_POSE)
- **Learning methods**:
  - Behavioral Cloning (MLP)
  - Behavioral Cloning with temporal modeling (GRU)
  - Offline Reinforcement Learning (BCQ)

The project focuses on analyzing:
- The role of demonstration quality and diversity
- The effect of temporal context in imitation learning
- Differences between imitation learning and offline reinforcement learning under limited data

## Repository Structure


## Expert Demonstrations

- **Imitation Learning**:  
  102 successful expert demonstration trajectories collected via keyboard teleoperation.

- **Offline Reinforcement Learning (BCQ)**:  
  An additional 25 trajectories are collected, including diverse end-effector approaches and
  minor corrective behaviors, to improve dataset coverage.

## Training and Evaluation

- Policies are trained using fixed datasets without online interaction.
- Evaluation is performed via rollout-based testing.
- Each trained policy is evaluated over **100 rollouts**, and success rates are reported.
- Ablation studies are conducted on:
  - Observation space composition
  - Network capacity
  - Sequence length for GRU policies
  - Training hyperparameters

## Requirements

- Python 3.8+
- MuJoCo
- robosuite
- PyTorch
- NumPy

(Exact dependencies may vary depending on your local setup.)

## Notes

- The project is designed for analysis and comparison rather than achieving state-of-the-art performance.
- All experiments are conducted in simulation without access to privileged simulator states during policy execution.

## Acknowledgements

This project builds upon prior work in imitation learning, offline reinforcement learning,
and robot manipulation research, including BCQ and robosuite.
