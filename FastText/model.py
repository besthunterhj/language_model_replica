import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int):
        super(FastText, self).__init__()

        # Inherent Property
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Model Architecture
        self.embedding = nn.Embedding()

if __name__ == '__main__':
    pass