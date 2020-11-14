"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.storage = dict()
        self.storage['state'] = np.zeros((max_size, state_dim))
        self.storage['action'] = np.zeros((max_size, action_dim))
        self.storage['next_state'] = np.zeros((max_size, state_dim))
        self.storage['reward'] = np.zeros((max_size, 1))
        self.storage['not_done'] = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.storage['state'][self.ptr] = state
        self.storage['action'][self.ptr] = action
        self.storage['next_state'][self.ptr] = next_state
        self.storage['reward'][self.ptr] = reward
        self.storage['not_done'][self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.storage['state'][ind]).to(device),
            torch.FloatTensor(self.storage['action'][ind]).to(device),
            torch.FloatTensor(self.storage['next_state'][ind]).to(device),
            torch.FloatTensor(self.storage['reward'][ind]).to(device),
            torch.FloatTensor(self.storage['not_done'][ind]).to(device)
        )

    def save(self, filename):
        np.save("./buffers/" + filename + ".npy", self.storage)

    def load(self, data):
        assert('next_observations' in data.keys())
        for i in range(data['observations'].shape[0] - 1):
            self.add(data['observations'][i], data['actions'][i], data['next_observations'][i],
                     data['rewards'][i], data['terminals'][i])
        print("Dataset size:" + str(self.size))
