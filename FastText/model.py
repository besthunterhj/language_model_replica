import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, word_num: int, embedding_size: int, hidden_size: int, label_num: int):
        super(FastText, self).__init__()

        # Inherent Property
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Model Architecture
        self.embedding = nn.Embedding(word_num, embedding_size)
        self.hidden = nn.Linear(embedding_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, label_num)

    def forward(self, X: torch.Tensor):

        # X: [batch_size, sequence_len, embedding_size]
        embed_tokens = self.embedding(X)

        # compute the embedding of the sequence according the idea of the paper
        # make sure the size of embed_sequences is [batch_size, embedding_size]
        embed_sequences = torch.mean(embed_tokens, dim=1).squeeze()

        # hidden: [batch_size, hidden_size]
        hidden = self.hidden(embed_sequences)

        # outputs: [batch_size, label_len]
        outputs = self.output_layer(hidden)

        return outputs


if __name__ == '__main__':

    model = FastText(1000, 200, 100, 4)
    test_tensor = torch.tensor(
        [
            [11, 6, 342, 0, 31],
            [2, 54, 123, 64, 12]
        ]
    )

    model.forward(test_tensor)