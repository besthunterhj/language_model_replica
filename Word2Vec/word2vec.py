from typing import Tuple, List

import numpy
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab as vc
from torchtext.vocab import Vocab
from collections import Counter
from nltk.corpus import brown
from torchtext.data import get_tokenizer
import numpy as np
from tqdm import tqdm

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class BrownDataset(Dataset):
    def __init__(self, dataset_type: str):
        tmp = [sentence for sentence in brown.sents() if len(sentence) > 3]
        if dataset_type == "train":
            self.sentences = [sentence for sentence in tmp[:33127]]
            self.tokens = [sentence[2] for sentence in tmp[:33127]]
        elif dataset_type == "dev":
            self.sentences = [sentence for sentence in tmp[33127:49691]]
            self.tokens = [sentence[2] for sentence in tmp[33127:49691]]
        elif dataset_type == "test":
            self.sentences = [sentence for sentence in tmp[49691:]]
            self.tokens = [sentence[2] for sentence in tmp[49691:]]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Tuple[str, List[str]]:
        current_token = self.tokens[index]
        current_sentence = self.sentences[index]
        return current_token, current_sentence


def create_vocab(words: list, min_freq: int, unknown_token: str, unknown_index: int) -> Vocab:
    # count the words
    counter = Counter(words)

    # init the ordered dictionary
    tokens_dict = dict(
        sorted(
            counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    # init the vocabulary by the ordered dictionary
    vocab = vc(
        ordered_dict=tokens_dict,
        min_freq=min_freq,
    )

    # handle the situation that exits the unknown word
    vocab.insert_token(
        token=unknown_token,
        index=unknown_index,
    )

    vocab.set_default_index(index=unknown_index)

    return vocab


class Word2Vec(nn.Module):

    def __init__(self, vocab_len: int, embedding_size: int):
        super(Word2Vec, self).__init__()

        self.embedding_size = embedding_size

        # init the architecture of the model(the units don't need to add the bias)
        self.hidden_layer = nn.Embedding(vocab_len, self.embedding_size)
        self.output_layer = nn.Linear(self.embedding_size, vocab_len)

    def forward(self, X: torch.Tensor):
        """

        :param X: the input of the model, its size must be [batch_size, 1]
        :return:
        """
        # hidden_layer: [batch_size, 1, embedding_size]
        hidden_layer = self.hidden_layer(X)

        # output_layer: [batch_size, 1, vocab_len]
        # outputs: [batch_size, 4, vocab_len]
        output_layer = self.output_layer(hidden_layer)
        tmp = torch.clone(output_layer)

        for i in range(3):
            output_layer = torch.cat((output_layer, tmp), dim=1)

        return output_layer


def pad(token_indexes: List[int], context_len: int, default_padding_val: int = 0, target_index: int = 2) -> List[int]:
    if len(token_indexes) > context_len:
        current_context = token_indexes[:context_len]
        current_context.pop(target_index)
        return current_context

    else:
        padded_token_indexes = token_indexes.copy()
        for i in range(context_len - len(token_indexes)):
            padded_token_indexes.append(default_padding_val)

        padded_token_indexes.pop(target_index)

        return padded_token_indexes


def trans_to_onehot(indexes: List[int], one_hots: numpy.ndarray):
    one_hot_context = []
    for item in indexes:
        item = one_hots[item]
        one_hot_context.append(item)

    return one_hot_context


def vocab_for_token(token: str, vocab: Vocab) -> int:
    return vocab([token])


def collate_func(samples: Tuple[str, List[str]], vocab: Vocab, one_hots: numpy.ndarray, context_len: int = 5) -> dict:
    tokens, sentences = list(zip(*samples))

    # 1. Sentences:
    #   + covert the tokens to indexes
    #   + padding
    #   + convert to a tensor
    contexts = torch.tensor(list(
        map(
            lambda current_sentence: trans_to_onehot(pad(vocab(current_sentence), context_len=context_len), one_hots),
            sentences
        )
    ))

    # 2. Tokens:
    #   + convert the tokens to indexes
    #   + convert to a tensor
    tokens = torch.tensor(list(
        map(
            lambda current_token: vocab_for_token(current_token, vocab),
            tokens
        )
    ))

    return {
        "tokens": tokens,
        "contexts": contexts,
    }


def train(model, criterion, optimizer, train_loader: DataLoader):
    model.train()

    losses = []
    for batch in tqdm(train_loader):
        tokens = batch["tokens"].to(device)
        contexts = batch["contexts"].to(device)

        # the result of prediction in this step
        sigmoid = nn.Sigmoid()
        predictions = model(tokens)

        current_loss = criterion(sigmoid(predictions).float(), contexts.float())
        losses.append(current_loss)

        # back propagation
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        # calculate and print the training loss for this epoch
        train_loss = torch.tensor(losses).mean()
        print(f"Train Loss : {train_loss}")


def validate(model, criterion, dev_loader: DataLoader):
    model.eval()

    all_contexts = []
    all_predictions = []

    losses = []
    with torch.no_grad():
        for batch in dev_loader:
            tokens = batch["tokens"].to(device)
            contexts = batch["contexts"].to(device)

            softmax = nn.Softmax(dim=-1)
            current_prediction = softmax(model(tokens))

            current_loss = criterion(current_prediction, contexts)
            losses.append(current_loss)

            all_contexts.append(contexts)
            all_predictions.append(current_prediction.argmax(dim=-1))

    all_contexts = torch.cat(all_contexts)
    all_predictions = torch.cat(all_predictions)
    valid_loss = torch.tensor(losses).mean()

    # accuracy_score function is not compatible with tensor, so we need to change them to numpy
    valid_acc = accuracy_score(
        y_true=all_contexts.detach().cpu().numpy(),
        y_pred=all_predictions.detach().cpu().numpy()
    )

    print(f"Valid Loss : {valid_loss}")
    print(f"Valid Acc  : {valid_acc}")

    return valid_loss


if __name__ == '__main__':

    words = list(brown.words())

    train_dataset = BrownDataset("train")
    dev_dataset = BrownDataset("dev")
    test_dataset = BrownDataset("test")

    min_freq = 3

    tokenizer = get_tokenizer("basic_english")
    vocab = create_vocab(words=words, min_freq=min_freq, unknown_token="<unk>", unknown_index=0)

    embedding_size = 100
    model = Word2Vec(
        vocab_len=len(vocab),
        embedding_size=embedding_size,
    )
    model.to(device=device)

    criterion = nn.BCELoss()

    lr = 1e-3
    optimizer = Adam(
        params=model.parameters(),
        lr=lr,
    )

    collate_fc = lambda samples: collate_func(
        samples=samples,
        vocab=vocab,
        one_hots=np.eye(len(vocab))
    )

    batch_size = 16
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fc,
    )

    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fc,
    )

    epoch = 10

    for i in range(epoch):
        print(f"Epoch: {i + 1}", "\n")
        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
        )

        # validate_loss = validate(
        #     model=model,
        #     criterion=criterion,
        #     dev_loader=dev_loader,
        # )
