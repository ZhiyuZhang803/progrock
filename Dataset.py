import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, features, labels, names, snippets):

        self.features = features
        self.labels = labels
        self.names = names
        self.snippets = snippets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Loading and returning a single sample as a tensor
        sample_feature = torch.tensor(self.features[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[idx], dtype=torch.long)
        sample_name = torch.tensor(self.names[idx], dtype=torch.long)
        sample_snippet = torch.tensor(self.snippets[idx], dtype=torch.long)
        return sample_feature, sample_label, sample_name, sample_snippet
