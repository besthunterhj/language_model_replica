from collections import Callable, Counter
from typing import List

import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset
from torchtext.vocab import vocab as vc
from torchtext.vocab import Vocab
from sklearn.datasets import fetch_20newsgroups

def create_vocab() -> Vocab:
    pass

class NewsDataset(Dataset):

    def __init__(self, docs: List[str], tokenizer: Callable):
        super(NewsDataset, self).__init__()

        # get the tokens from input news docs
        tokens = [tokenizer(doc.lower().strip()) for doc in docs]

        # init the words_counter for the tokens
        counter = Counter(tokens)
        words_dict = dict(
            sorted(
                counter.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        vocab = vc(
            ordered_dict=words_dict,
            min_freq=3,
        )


# main function
def main(n_docs: int, epochs: int, batch_size: int):
    # init the corpus dataset
    news = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))["data"]

    tokenizer = get_tokenizer("basic_english")

    news_dataset = NewsDataset(docs=news[:n_docs], tokenizer=tokenizer)

if __name__ == '__main__':
    main(1,1,1)