import torch
import numpy as np
from real_miner_env import RealMinerEnv
from train_dqn_real import DQN

with open("bitcoin_headers.txt", "r") as f:
    block_headers = [line.strip() for line in f if line.strip()]

env = RealMinerEnv(block_headers)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load("trained_dqn_model.pth"))
model.eval()

success_count = 0
attempts = 100

for _ in range(attempts):
    state = env.reset()
    with torch.no_grad():
        action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
    _, reward, _, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    if reward > 0:
        success_count += 1

print(f"\n✅ Out of {attempts} attempts, successful hashes found: {success_count}")

