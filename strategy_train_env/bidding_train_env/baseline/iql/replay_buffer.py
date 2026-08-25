import random
from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Reinforcement learning replay buffer for training data
    """

    def __init__(self, arrays=None):
        self.memory = []
        self.arrays = None
        if arrays is not None:
            self.arrays = tuple(np.ascontiguousarray(array, dtype=np.float32) for array in arrays)
            lengths = {len(array) for array in self.arrays}
            if len(lengths) != 1:
                raise ValueError("All replay-buffer arrays must have the same length")

    def push(self, state, action, reward, next_state, done):
        """saving an experience tuple"""
        if self.arrays is not None:
            raise RuntimeError("Cannot push into an array-backed ReplayBuffer")
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """randomly sampling a batch of experiences"""
        if self.arrays is not None:
            indices = np.random.randint(0, len(self), size=batch_size)
            states, actions, rewards, next_states, dones = (
                torch.from_numpy(array[indices]) for array in self.arrays
            )
            return states, actions, rewards, next_states, dones
        tem = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*tem)
        states, actions, rewards, next_states, dones = (
            torch.as_tensor(np.stack(values), dtype=torch.float32)
            for values in (states, actions, rewards, next_states, dones)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """return the length of replay buffer"""
        return len(self.arrays[0]) if self.arrays is not None else len(self.memory)


if __name__ == '__main__':
    buffer = ReplayBuffer()
    for i in range(1000):
        buffer.push(np.array([1, 2, 3]), np.array(4), np.array(5), np.array([6, 7, 8]), np.array(0))
    print(buffer.sample(20))
