from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab as vc
from torchtext.vocab import Vocab
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def create_vocab(word_dict: dict, min_freq: int, unknown_token: str = "<unk>", unknown_index: int = 0) -> Vocab:
    vocab = vc(
        ordered_dict=word_dict,
        min_freq=min_freq
    )

    vocab.insert_token(
        token=unknown_token,
        index=unknown_index
    )

    vocab.set_default_index(index=unknown_index)

    return vocab


class NewsDataset(Dataset):

    def __init__(self, docs: List[List[str]], vocab: Vocab, win_size: int = 5):
        """
        Create the co-occurrence matrix and init the training data(target_words and context_words) and labels(co_scores)
        :param docs: a list contains the results of tokenized news, and the tokenized news are lists of tokens
        :param vocab: the vocabulary of the corpus
        :param win_size: the size of the slide windows of co-occurrence matrix
        """

        super(NewsDataset, self).__init__()

        # create the co-occurrence matrix
        co_matrix = Counter()

        # for each piece of the tokenized news, create the matrix
        for current_doc in docs:
            current_indexes = vocab(current_doc)
            for target_index, target_word in enumerate(current_indexes):
                for context_index, context_word in enumerate(current_indexes[target_index + 1: target_index + win_size], start=target_index + 1):
                    # the value of the co-occurrence matrix depends on the distance between the target and context
                    co_matrix[(target_word, context_word)] += 1 / (context_index - target_index)

        # store the information of co-occurrence matrix
        target_words, context_words, co_scores = zip(*((target, context, score) for (target, context), score in co_matrix.items()))
        self.target_words = torch.LongTensor(target_words)
        self.context_words = torch.LongTensor(context_words)
        self.co_scores = torch.FloatTensor(co_scores)

    def __len__(self):
        return len(self.co_scores)

    def __getitem__(self, index):
        return self.target_words[index], self.context_words[index], self.co_scores[index]


class GloVe(nn.Module):
    def __init__(self, embedding_size: int, word_nums: int, y_max: int = 100, alpha: float = 0.75):
        super().__init__()
        self.embedding_size = embedding_size
        self.word_nums = word_nums
        self.y_max = y_max
        self.alpha = alpha
        # the parameters are initialized by other function
        self.context_vecs = nn.Embedding(num_embeddings=self.word_nums, embedding_dim=self.embedding_size)
        self.target_vecs = nn.Embedding(num_embeddings=self.word_nums, embedding_dim=self.embedding_size)
        self.context_bias = nn.Embedding(num_embeddings=self.word_nums, embedding_dim=1)
        self.target_bias = nn.Embedding(num_embeddings=self.word_nums, embedding_dim=1)

    def forward(self, targets, contexts, scores):
        """
        compute the loss according to the equation of the paper
        :param targets: [batch_size, 1]
        :param contexts: [batch_size, 1]
        :param scores: [batch_size, 1]
        :return: the loss of this batch
        """
        f_function = scores.div(self.y_max).pow(self.alpha).clamp_max(1.0)
        # targets_vecs, target_bias: [batch_size, embedding_size], [batch_size]
        targets_vecs = self.target_vecs(targets)
        targets_bias = self.target_bias(targets)

        # contexts_vecs: [batch_size, embedding_size], [batch_size]
        contexts_vecs = self.context_vecs(contexts)
        contexts_bias = self.context_bias(contexts)

        log_items = torch.log(1 + scores)
        square_items = (torch.sum(targets_vecs * contexts_vecs + targets_bias + contexts_bias, dim=1) - log_items) ** 2

        loss = torch.mean(f_function * square_items)

        return loss


def train(model, optimizer, data_loader: DataLoader):

    model.train()

    losses = []
    for batch_targets, batch_contexts, batch_scores in tqdm(data_loader):
        current_loss = model(batch_targets, batch_contexts, batch_scores)
        losses.append(current_loss)

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

    train_loss = torch.tensor(losses).mean()
    print(f"Train Loss : {train_loss:.3f}")


def show_vector_space(showed_word_num: int, trained_model: GloVe, vocab: Vocab):
    final_embedding = trained_model.target_vecs.weight.detach().numpy() + trained_model.context_vecs.weight.detach().numpy()
    embed_pca = PCA(n_components=4).fit_transform(final_embedding[:showed_word_num, :])
    embed_tsne = TSNE(metric='euclidean', verbose=1, n_jobs=4).fit_transform(embed_pca)
    fig, ax = plt.subplots(figsize=(20, 14))
    for index in range(showed_word_num):
        x, y = embed_tsne[index, :]
        ax.scatter(x, y, color='steelblue')
        ax.annotate(vocab.get_itos()[index], (x, y), alpha=0.7)

    plt.show()


# main function
def main(n_docs: int, epochs: int, batch_size: int, embedding_size: int, lr: float):
    # init the corpus dataset
    news = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))["data"][:n_docs]

    # init the tokenizer
    tokenizer = get_tokenizer("basic_english")

    # get the tokens from news
    tokens = [token
              for doc in news[:n_docs]
              for token in tokenizer(doc.lower().strip())]

    # count the tokens
    counter = Counter(tokens)

    # init the word_dict for vocabulary
    word_dict = dict(
        sorted(
            counter.items(),
            key= lambda x: x[1],
            reverse=True
        )
    )

    # init the vocabulary
    vocab = create_vocab(word_dict=word_dict, min_freq=3)

    # tokenize the news
    tokenized_docs = [tokenizer(doc.lower().strip()) for doc in news]

    # init the training data
    news_dataset = NewsDataset(docs=tokenized_docs, vocab=vocab)

    # init the data_loader for training
    news_dataloader = DataLoader(dataset=news_dataset, batch_size=batch_size, shuffle=True)

    # init the GloVe model and the optimizer for training
    model = GloVe(embedding_size, len(vocab))
    optimizer = optim.Adagrad(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print("The training step is ready! ")
        train(
            model=model,
            optimizer=optimizer,
            data_loader=news_dataloader
        )

    # show the 2-D space of the word vectors by plt
    show_vector_space(showed_word_num=300, trained_model=model, vocab=vocab)


if __name__ == '__main__':
    main(1000, 10, 128, 300, 0.05)

