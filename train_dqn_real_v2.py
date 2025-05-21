import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from real_miner_env_v2 import RealMinerEnvV2

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def select_action(state, policy_net, epsilon, action_space):
    if random.random() < epsilon:
        return random.randrange(action_space)
    with torch.no_grad():
        return policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()

def train_dqn(env, episodes=1000):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    update_freq = 10

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, policy_net, epsilon, env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 50 == 0:
            print(f"Episode {episode} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    return policy_net

# Load real headers
with open("bitcoin_headers.txt", "r") as f:
    block_headers = [line.strip() for line in f if line.strip()]

env = RealMinerEnvV2(block_headers)
trained_model = train_dqn(env)
torch.save(trained_model.state_dict(), "trained_dqn_model.pth")
print("✅ Model trained and saved as trained_dqn_model.pth")
