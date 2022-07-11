# This file try to achieve the goal of encoing the Neural Network Language Model
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from typing import List, Tuple
from nltk import FreqDist
from nltk.corpus import brown

# init the type of data
dtype = torch.FloatTensor
m = 30
hidden_unit_num = 100


def init_corpus(raw_corpus: List[str]) -> Tuple[int, dict, dict]:
    """
    Initialize the corpus
    :param raw_corpus: the corpus of the Neural Network Language Model
    :return: V, the dictionaries of vocabulary
    """

    # init the vocabulary
    vocabulary = " ".join(raw_corpus).split(" ")

    # remove the words which repeats
    vocabulary = list(set(vocabulary))

    # init the mapping lists word_dict:'indexes->words' and index_dict:'words->indexes'
    word_dict = {text: index for index, text in enumerate(vocabulary)}
    index_dict = {index: text for index, text in enumerate(vocabulary)}

    # get the number of vocabulary
    V = len(vocabulary)

    return V, word_dict, index_dict


def make_batch(sentences: List[str], word_dict: dict) -> Tuple[list, list]:
    """
    The input of Pytorch must be mini-batch, so this function is making batch of input
    :param sentences: the corpus
    :param word_dict: the vocabulary
    :return: the batch of input and the batch of target
    """

    input_batch = []
    target_batch = []

    for sentence in sentences:
        # Init the data of training(previous n_words and their corresponding labels)
        words_for_sentence = sentence.split(" ")
        input = [word_dict[n] for n in words_for_sentence[:-1]]
        label = word_dict[words_for_sentence[-1]]

        input_batch.append(input)
        target_batch.append(label)

    return input_batch, target_batch


class NNLM(nn.Module):
    def __init__(self, V, n_steps):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(V, m)
        self.H = nn.Parameter(torch.randn(n_steps * m, hidden_unit_num).type(dtype))
        self.W = nn.Parameter(torch.randn(n_steps * m, V).type(dtype))
        self.d = nn.Parameter(torch.randn(hidden_unit_num).type(dtype))
        self.U = nn.Parameter(torch.randn(hidden_unit_num, V).type(dtype))
        self.b = nn.Parameter(torch.randn(V).type(dtype))

    def forward(self, X, n_steps):
        """
        :param X: [batch_size, n_steps]
        """

        # nn.Embedding accepts a list([batch_size, vector of each element(each element of the input data)])
        # and then change it to the embedding, so the variable of its return is (batch_size, vector of each element, embedding size)
        X = self.C(X)
        X = X.view(-1, n_steps * m)
        hidden_out = torch.tanh(self.d + torch.mm(X, self.H))
        output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U)
        return output


if __name__ == "__main__":

    """
    Experiment of the Brown Corpus
    """

    texts = brown.words()
    freq_dict = FreqDist(texts)

    word_list = []
    for item in list(freq_dict.items()):
        if item[-1] > 3:
            word_list.append(item[0].lower())

    vocabulary = list(set(word_list))
    V = len(vocabulary) + 1
    n_steps = 5

    # make the word_dict
    word_dict = {text: index for index, text in enumerate(vocabulary)}
    word_dict["[rare]"] = len(vocabulary)

    # sentences from the Brown corpus
    sentences = brown.sents()

    input_batch = []
    target_batch = []

    for sentence in sentences:
        if len(sentence) >= 6:
            input_data = []

            for n in sentence[:5]:
                if n not in list(word_dict.keys()):
                    input_data.append(word_dict["[rare]"])

                else:
                    input_data.append(word_dict[n.lower()])

            if sentence[5] not in list(word_dict.keys()):
                label = word_dict["[rare]"]

            else:
                label = word_dict[sentence[5].lower()]

            input_batch.append(input_data)
            target_batch.append(label)

    # sentences = ['I like studying', 'I love coding', 'I hate wars']
    # n_steps = len(sentences[0].split(" ")) - 1
    #
    # V, word_dict, index_dict = init_corpus(sentences)
    #
    # input_batch, target_batch = make_batch(sentences, word_dict)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    model = NNLM(V, n_steps)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("model is ready")

    # Training
    for epoch in range(10):
        for batch_x, batch_y in loader:
            # First, the gradient must be zero
            optimizer.zero_grad()

            # Then, start to train(forward function)
            output = model(batch_x, n_steps)
            # Compute the loss
            loss = criterion(output, batch_y)

            # if (epoch + 1) % 10 == 0:
            print("Epoch:", (epoch + 1), "cost = ", "{:.6f}".format(loss), " pp = ", int(torch.exp(loss)))

            loss.backward()
            optimizer.step()

    # print(model.C)
    #
    # # Predict
    # predict = model(input_batch, n_steps).data.max(1, keepdim=True)[1]
    #
    # # Test
    # print([sen.split()[:n_steps] for sen in sentences], '->', [index_dict[n.item()] for n in predict.squeeze()])