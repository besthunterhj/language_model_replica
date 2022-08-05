import copy

import numpy as np
import torch
import torch.nn as nn

#
# def random_batch():
#     random_inputs = []
#     random_labels = []
#     random_index = np.random.choice(range(len(skip_grams)), 2, replace=False)
#
#     for i in random_index:
#         random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # target
#         random_labels.append(skip_grams[i][1])  # context word
#
#     return random_inputs, random_labels


# if __name__ == '__main__':
    #
    # test = np.random.choice(a=5, size=1, replace=False)
    # print(test)
    #
    # sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
    #              "dog cat animal", "cat monkey animal", "monkey dog animal"]
    #
    # word_sequence = " ".join(sentences).split()
    # word_list = " ".join(sentences).split()
    # word_list = list(set(word_list))
    # word_dict = {w: i for i, w in enumerate(word_list)}
    # voc_size = len(word_list)
    #
    # # Make skip gram of one size window
    # skip_grams = []
    # for i in range(1, len(word_sequence) - 1):
    #     target = word_dict[word_sequence[i]]
    #     context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    #     for w in context:
    #         skip_grams.append([target, w])
    #
    # for epoch in range(5000):
    #     input_batch, target_batch = random_batch()
    #     input_batch = torch.Tensor(input_batch)
    #     target_batch = torch.LongTensor(target_batch)
    #
    #     print(f"input batch: {input_batch}")
    #
    #     print()
    #
    #     print(f"target batch: {target_batch}")
    #
    #     input()

from nltk.corpus import brown


# count = 0
# for item in brown.sents():
#     if len(item) > 3:
#         count += 1
#
# print(count)


test = np.eye(5)

print(test)
# print(type(test))
#
# print()
#
# print(test[2])
# print(type(test[2]))

# a = torch.tensor([[[-1.1, 2.3, -3.1], [5., -5.2, 6.1], [9.2, -12.4, 5.]]])
# sigmoid = nn.Sigmoid()
#
# print(a)
# print()
# print(sigmoid(a))