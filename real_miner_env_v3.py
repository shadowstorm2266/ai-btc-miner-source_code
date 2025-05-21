import gym
import numpy as np
import hashlib

class RealMinerEnv(gym.Env):
    def __init__(self, block_headers, target_difficulty=2**240):
        super(RealMinerEnv, self).__init__()
        self.block_headers = block_headers
        self.target_difficulty = target_difficulty
        self.current_index = 0
        self.nonce_range = 10000

        self.action_space = gym.spaces.Discrete(self.nonce_range)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(32,), dtype=np.uint8)

    def reset(self):
        self.current_index = (self.current_index + 1) % len(self.block_headers)
        self.current_block = bytes.fromhex(self.block_headers[self.current_index])
        self.last_hash = np.zeros(32, dtype=np.uint8)
        return self.last_hash

    def step(self, action):
        nonce = action
        block_with_nonce = self.current_block + nonce.to_bytes(4, 'big')
        hash_result = hashlib.sha256(block_with_nonce).digest()
        hash_int = int.from_bytes(hash_result, 'big')

        reward = 1 if hash_int < self.target_difficulty else -0.05  # ⚡ Reduced penalty!
        done = True
        self.last_hash = np.frombuffer(hash_result, dtype=np.uint8)
        return self.last_hash, reward, done, {}
