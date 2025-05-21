import gym
import numpy as np
import hashlib
import random

class RealMinerEnvV2(gym.Env):
    def __init__(self, block_headers, target_difficulty=2**240):
        super(RealMinerEnvV2, self).__init__()
        self.block_headers = block_headers
        self.target_difficulty = target_difficulty
        self.current_index = 0
        self.nonce_range = 10000

        self.action_space = gym.spaces.Discrete(self.nonce_range)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(32,), dtype=np.uint8)
        self.previous_action = None

    def reset(self):
        self.current_index = (self.current_index + 1) % len(self.block_headers)
        self.current_block = bytes.fromhex(self.block_headers[self.current_index])
        self.last_hash = np.zeros(32, dtype=np.uint8)
        self.previous_action = None
        return self.last_hash

    def step(self, action):
        entropy_penalty = -0.01 if action == self.previous_action else 0.0
        self.previous_action = action

        nonce = action
        block_with_nonce = self.current_block + nonce.to_bytes(4, 'big')
        hash_result = hashlib.sha256(block_with_nonce).digest()
        hash_int = int.from_bytes(hash_result, 'big')

        if hash_int < self.target_difficulty:
            reward = 1.0
        elif hash_int < self.target_difficulty * 16:
            reward = 0.1
        else:
            reward = -0.05

        reward += entropy_penalty
        self.last_hash = np.frombuffer(hash_result, dtype=np.uint8)
        return self.last_hash, reward, True, {}
