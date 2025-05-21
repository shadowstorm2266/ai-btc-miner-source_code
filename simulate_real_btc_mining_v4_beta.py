import torch
import numpy as np
from real_miner_env_v4_beta import RealMinerEnv
from train_dqn_real_v4_beta import DQN

# Load Bitcoin headers
with open("bitcoin_headers_v4.txt", "r") as f:
    block_headers = [line.strip() for line in f.readlines() if line.strip()]

env = RealMinerEnv(block_headers)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load("trained_dqn_model.pth"))
model.eval()

success_count = 0
attempts = 200

for attempt in range(attempts):
    state = env.reset()
    with torch.no_grad():
        action = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
    _, reward, done, _ = env.step(action)
    if reward > 0:
        success_count += 1
    print(f"Action: {action}, Reward: {reward:.2f}, {'✅ Success' if reward > 0 else '❌ Fail'}")

print(f"\n✅ Out of {attempts} attempts, successful blocks mined: {success_count}")
