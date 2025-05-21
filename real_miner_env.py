import gym
import numpy as np
import hashlib
import random

class RealMinerEnv(gym.Env):
    def __init__(self, block_headers, target_difficulty=2**240):
        super(RealMinerEnv, self).__init__()
        self.original_headers = block_headers
        self.target_difficulty = target_difficulty
        self.nonce_range = 10000

        self.action_space = gym.spaces.Discrete(self.nonce_range)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(32,), dtype=np.uint8)

    def reset(self):
        random.shuffle(self.original_headers)
        self.current_index = 0
        self.current_block = bytes.fromhex(self.original_headers[self.current_index])
        self.last_hash = np.zeros(32, dtype=np.uint8)
        return self.last_hash

    def step(self, action):
        nonce = action + random.randint(0, 999)  # inject more entropy
        block_with_nonce = self.current_block + nonce.to_bytes(4, 'big')
        hash_result = hashlib.sha256(block_with_nonce).digest()
        hash_int = int.from_bytes(hash_result, 'big')

        reward = 10 if hash_int < self.target_difficulty else -0.05
        done = True
        self.last_hash = np.frombuffer(hash_result, dtype=np.uint8)

        self.current_index = (self.current_index + 1) % len(self.original_headers)
        self.current_block = bytes.fromhex(self.original_headers[self.current_index])
        return self.last_hash, reward, done, {}



