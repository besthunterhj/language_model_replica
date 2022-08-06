# import re
# from collections.abc import Callable
# from collections import Counter
# from typing import List
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from sklearn.datasets import fetch_20newsgroups
# from tqdm import tqdm
#
#
# class GloveDataset(Dataset):
#     """ 将一份英文语料切词，然后组织成可供 GloVe训练用的 共现矩阵 """
#     REGEX_WORD = re.compile(r"\b[a-zA-Z]{2,}\b")
#
#     def __init__(self, docs, min_word_occurences=3, oov_token='<oov>', window_size=3):
#         # 数值化
#         docs_tok = [self.REGEX_WORD.findall(doc.lower()) for doc in docs]  # docs tokenized
#         word_counter = {w: c for w, c in Counter(w for d in docs_tok for w in d).items()
#                         if c > min_word_occurences}  # 比 seq快一倍
#         w2i = {oov_token: 0}
#         docs_tok_id = [[w2i.setdefault(w, len(w2i)) if w in word_counter else 0
#                         for w in doc] for doc in docs_tok]  # docs tokenized, in id
#        # self.w2i, self.i2w = w2i, seq(w2i.items()).order_by(lambda w_i: w_i[1]).smap(lambda w, i: w).to_list()
#         self.n_words = len(w2i)  # 注意不是 len(word_counter), 否则缺个OOV, 越界
#
#         # 统计共现矩阵
#         comatrix = Counter()
#         for words_id in tqdm(docs_tok_id, desc='docs2comtx'):
#             for i, w1 in enumerate(words_id):  # 注意窗口限制
#                 for j, w2 in enumerate(words_id[i + 1: i + window_size], start=i + 1):
#                     comatrix[(w1, w2)] += 1 / (j - i)
#
#         # 从共现矩阵中提取训练样本: (中心词A的下标, 邻居词B的下标) -> A和B的"共现值"
#         a_words, b_words, co_score = zip(*((left, right, x) for (left, right), x in comatrix.items()))
#         self.L_words = torch.LongTensor(a_words)
#         self.R_words = torch.LongTensor(b_words)
#         self.Y = torch.FloatTensor(co_score)
#
#     def __len__(self):
#         return len(self.Y)
#
#     def __getitem__(self, item):
#         return self.L_words[item], self.R_words[item], self.Y[item]
#
#
# if __name__ == '__main__':
#     newsgroup = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
#
#     glove_data = GloveDataset(newsgroup.data[:10])

import torch
test = torch.randn(8)
print(test)