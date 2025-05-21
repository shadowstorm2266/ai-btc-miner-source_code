import torch
import numpy as np
from real_miner_env_v3 import RealMinerEnv
from train_dqn_real_v3 import DQN

# Load real Bitcoin block headers
with open("bitcoin_headers.txt", "r") as f:
    block_headers = [line.strip() for line in f.readlines() if line.strip()]

env = RealMinerEnv(block_headers)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load("trained_dqn_model.pth"))
model.eval()

successes = 0
for i in range(100):
    state = env.reset()
    action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

    if reward > 0:
        successes += 1

print(f"✅ Out of 100 attempts, successful hashes found: {successes}")
