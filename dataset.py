import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    """Pairwise Probability dataset"""

    def __init__(self, filename):
        data = np.loadtxt(filename)
        self.ids = torch.from_numpy(data[:, :2].astype(np.long))
        self.probs = torch.from_numpy(data[:, 2].astype(np.long))
        self.length = self.ids.shape[0]

    def __getitem__(self, index):
        return self.ids[index], self.probs[index]

    def __len__(self):
        return self.length

