import torch
import numpy as np
from real_miner_env_v3_5 import RealMinerEnvV3_5
from train_dqn_real_v3_5 import DQN

# Load headers
with open("bitcoin_headers.txt", "r") as f:
    block_headers = [line.strip() for line in f.readlines() if line.strip()]

env = RealMinerEnvV3_5(block_headers)

# Load model
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load("trained_dqn_model.pth"))
model.eval()

# Simulate mining
success_count = 0

for _ in range(200):  # Try 200 mining attempts
    state = env.reset()
    with torch.no_grad():
        action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
    next_state, reward, done, _ = env.step(action)

    print(f"Action: {action}, Reward: {reward}")
    if reward > 1:
        success_count += 1

print(f"\n✅ Out of 200 attempts, successful hashes found: {success_count}")
