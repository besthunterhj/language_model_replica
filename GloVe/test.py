from collections.abc import Callable
from collections import Counter
from typing import List

import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset
from torchtext.vocab import vocab as vc
from torchtext.vocab import Vocab
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm


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


# main function
def main(n_docs: int, epochs: int, batch_size: int):
    # init the corpus dataset
    news = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))["data"][:n_docs]

    # init the tokenizer
    tokenizer = get_tokenizer("basic_english")

    # get the tokens from news
    all_tokens = []
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


class GloVe(nn.Module):
    def __init__(self, embedding_size: int, word_num: int, y_max: int = 100, alpha: float = 0.75):
        super().__init__()
        self.embedding_size = embedding_size
        self.y_max = y_max
        self.alpha = alpha
        # the parameters are initialized by other function
        self.context_vecs = nn.Embedding(num_embeddings=word_num, embedding_dim=embedding_size)
        self.target_vecs = nn.Embedding(num_embeddings=word_num, embedding_dim=embedding_size)
        self.context_bias = nn.Parameter(torch.randn(word_num))
        self.target_bias = nn.Parameter(torch.randn(word_num))
        self.w2i, self.i2w = {}, []

    # def build_model_from_dataset(self, dataset: NewsDataset, word_nums: int):
    #     self.context_vecs = nn.Embedding(word_nums, )


if __name__ == '__main__':
    main(10, 1, 1)

