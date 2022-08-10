import copy

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm

from dataset.ag_news_dataset import AGNewsDataset, collate_func
from utils.preprocessing_utils import create_vocab, create_labels_mapping
from utils.training_utils import device


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


def train(model, criterion, optimizer, train_loader: DataLoader):
    # sign the "train" label
    model.train()

    # create a variable named "losses" to store the training losses
    losses = []
    for batch in tqdm(train_loader):
        texts = batch["texts"].to(device)
        labels = batch["labels"].to(device)

        # the result of prediction in this step
        current_prediction = model(texts)

        # criterion function: input[batch_size, num_labels], output[batch_size]
        current_loss = criterion(current_prediction, labels)
        losses.append(current_loss)

        # back propagation
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

    # calculate and print the training loss for this epoch
    train_loss = torch.tensor(losses).mean()
    print(f"Train Loss : {train_loss:.3f}")


def validate(model, criterion, dev_loader: DataLoader) -> float:
    # sign the "validate" label
    model.eval()

    # init "all_prediction" and "all_labels" to store the all results of prediction and labels in dev data
    all_labels = []
    all_predictions = []

    # init the variable "losses" to store all loss in dev data
    losses = []

    # the validation step doesn't need gradient descent
    with torch.no_grad():
        for batch in dev_loader:
            # get the data of samples in each batch
            texts = batch["texts"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(texts)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            all_labels.append(labels)
            # predictions are the shape as [batch_size, 4]
            all_predictions.append(predictions.argmax(dim=-1))

    # all_labels and all_predictions are list of many tensors, so need to concatenate them to a tensor
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    valid_loss = torch.tensor(losses).mean()

    # accuracy_score function is not compatible with tensor, so we need to change them to numpy
    valid_acc = accuracy_score(
        y_true=all_labels.detach().cpu().numpy(),
        y_pred=all_predictions.detach().cpu().numpy()
    )

    print(f"Valid Loss : {valid_loss:.3f}")
    print(f"Valid Acc  : {valid_acc:.3f}")

    return valid_loss


def test(model, test_loader: DataLoader, labels_mapping: dict):

    # init "all_prediction" and "all_labels" to store the all results of prediction and labels in dev data
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            texts = batch["texts"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(texts)

            all_labels.append(labels)
            # predictions are the shape as [batch_size, 4]
            all_predictions.append(predictions.argmax(dim=-1))

    all_labels = torch.cat(all_labels).detach().cpu().numpy()
    all_predictions = torch.cat(all_predictions).detach().cpu().numpy()

    # show the results of testing
    test_acc = accuracy_score(
        y_true=all_labels,
        y_pred=all_predictions,
    )

    print(f"Test Acc   : {test_acc:.3f}")

    print("\nClassification Report : ")
    print(classification_report(all_labels, all_predictions, target_names=labels_mapping.keys()))

    print("\nConfusion Matrix : ")
    print(confusion_matrix(all_labels, all_predictions))


def predict(model, text: str, tokenizer, vocab: Vocab, labels_mapping: dict):

    # get the tokens from the text
    tokens = tokenizer(text)

    # change the textual tokens to their indexes
    indexes = vocab(tokens)

    # Because the model only accept the mini-batch data, we need to change the shape of indexes
    temp_input = torch.tensor([indexes]).to(device)

    with torch.no_grad():
        prediction = model(temp_input)

    # the reason that add ".item()" : get the data from a tensor
    prediction_index = prediction[0].argmax(dim=0).item()

    prediction_label = {
        index: label for label, index in labels_mapping.items()
    }[prediction_index]

    print(f"\ntext: {text}")


def main(data_path: str, min_freq: int, embedding_size: int, hidden_size: int, lr: float, max_len: int, batch_size: int, epochs: int):
    # init the dataset
    train_dataset = AGNewsDataset(data_path + "/train.csv")
    dev_dataset = AGNewsDataset(data_path + "/dev.csv")
    test_dataset = AGNewsDataset(data_path + "/test.csv")

    # init the tokenizer
    tokenizer = get_tokenizer("basic_english")

    # init the vocabulary
    vocab = create_vocab(texts=train_dataset.texts, tokenizer=tokenizer, min_freq=min_freq)

    # init the label mapping
    label_mapping = create_labels_mapping(train_dataset.labels)

    # init the model
    model = FastText(word_num=len(vocab), embedding_size=embedding_size, hidden_size=hidden_size, label_num=len(label_mapping.items()))
    model.to(device)

    # set the loss function
    criterion = nn.CrossEntropyLoss()

    # init the optimizer
    optimizer = Adam(params=model.parameters(), lr=lr)

    # explicitly define the collection function
    collate_fc = lambda samples: collate_func(
        samples=samples,
        tokenizer=tokenizer,
        vocab=vocab,
        max_len=max_len,
        labels_mapping=label_mapping,
    )

    # training loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fc
    )

    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fc
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fc
    )

    best_model = None
    min_validate_loss = float("inf")

    for i in range(epochs):
        print(f"Epoch {i + 1} :")

        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader
        )

        validate_loss = validate(
            model=model,
            criterion=criterion,
            dev_loader=dev_loader,
        )

        print()
        # Update and store the best model
        if validate_loss < min_validate_loss:
            min_validate_loss = validate_loss
            # assign the memory and copy the whole model
            best_model = copy.deepcopy(model)

    best_model.to(device=device)
    test(
        model=best_model,
        test_loader=test_loader,
        labels_mapping=label_mapping,
    )


if __name__ == '__main__':

    main("data", 3, 200, 100, 1e-3, 25, 32, 10)
