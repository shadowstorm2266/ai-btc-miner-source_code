# 🧠 AI-Based Bitcoin Miner using Deep Reinforcement Learning

Welcome to the official source code for **AI-BTC-Miner**, an experimental project that explores the use of **Deep Q-Learning (DQN)** for intelligent Bitcoin mining. This research aims to investigate whether a neural agent can learn optimal nonce selection strategies based on real Bitcoin block headers to simulate and eventually perform real Proof-of-Work (PoW) mining.

---

## 📌 Project Overview

Traditional Bitcoin mining relies on brute-force hardware to repeatedly hash block headers until the result meets the required difficulty (i.e., starts with a number of leading zero bits). This project proposes a new direction: **Can an AI agent learn how to find valid nonces intelligently, by recognizing patterns in block headers and optimizing search paths using reinforcement learning?**

---

## 🔬 Research Phases

### ✅ Phase 1: Simulation Environment
- Implemented a custom Gym-like environment that mimics Bitcoin's mining mechanism.
- Used fake/random block headers and a low-difficulty target to verify reward shaping and episode mechanics.

### ✅ Phase 2: DQN Training with Simulated Headers
- Trained a simple DQN model using synthetic headers.
- Introduced entropy into nonce selection and basic reward tracking.

### ✅ Phase 3: Reinforcement Learning with Real Headers
- Introduced real 80-byte Bitcoin block headers.
- Switched reward logic to hash matching using SHA-256d.
- Introduced adaptive difficulty and reward shaping.

### ✅ Phase 4: Trained Model on 10,000 Real Headers
- Finalized model: `trained_dqn_model.pth` (available in this repo).
- Trained over thousands of episodes with entropy, adaptive logging, and exploration/exploitation balance.
- Result: The model learned consistent nonce estimation strategies even under dynamic difficulty targets.

---

## 🚧 Phase 5: Final AI Miner Training (WIP)
- Training on kaggle (p100 GPU).
- **Goal:** Train the AI on **50,000+ real block headers** with exact Bitcoin difficulty targets.
- Targeted training using a pre-generated dataset of 49995 Bitcoin headers.
- 150,000+ episode training with real SHA-256 mining rules.
- Will attempt to mine a valid hash with required leading zeroes.

> **Live progress:** Phase 5 is actively being developed and documented in our [research paper](https://github.com/shadowstorm2266/ai-btc-miner-source_code).

---
