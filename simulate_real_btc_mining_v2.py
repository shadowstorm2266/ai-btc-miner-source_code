import torch
import numpy as np
from real_miner_env_v2 import RealMinerEnvV2
from train_dqn_real_v2 import DQN

# Load headers
with open("bitcoin_headers.txt", "r") as f:
    block_headers = [line.strip() for line in f if line.strip()]

env = RealMinerEnvV2(block_headers)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load("trained_dqn_model.pth"))
model.eval()

successful_hashes = 0

for _ in range(100):
    state = env.reset()
    with torch.no_grad():
        action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
    next_state, reward, done, _ = env.step(action)

    print(f"Action: {action}, Reward: {reward}")
    if reward >= 0.1:
        successful_hashes += 1

print(f"\n✅ Out of 100 attempts, successful hashes found: {successful_hashes}")
