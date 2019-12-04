"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):

        super(BiDAF, self).__init__()

        self.hidden_size = hidden_size

        self.word_emb = layers.WordEmbedding(word_vectors, hidden_size)
        self.char_emb = layers.CharEmbedding(char_vectors, hidden_size)

        # assert hidden_size * 2 == (char_channel_size + word_dim)

        # highway network
        self.hwy = layers.HighwayEncoder(2, hidden_size*2)

        # highway network
        # for i in range(2):
        #     setattr(self, f'highway_linear{i}', nn.Sequential(
        #         nn.Linear(hidden_size * 2, hidden_size * 2), nn.ReLU()))

        #     setattr(self, f'hightway_gate{i}', nn.Sequential(
        #         nn.Linear(hidden_size * 2, hidden_size * 2), nn.Sigmoid()))

        # self.emb = layers.Embedding(word_vectors=word_vectors,
        #                             hidden_size=hidden_size,
        #                             drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size*2,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob,
                                         c_len,
                                         ndf=100)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs       # Context
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs       # Query
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # (batch_size, c_len, hidden_size)
        # (batch_size, q_len, hidden_size)
        c_word = self.word_emb(cw_idxs)
        q_word = self.word_emb(qw_idxs)

        c_char = self.char_emb(cc_idxs)
        q_char = self.char_emb(qc_idxs)

        # print("c word size: ", c_word.size())
        # print("q word size: ", q_word.size())
        # print("c character size: ", c_char.size())
        # print("q character size: ", q_char.size())

        c_cat = torch.cat([c_word, c_char], dim=-1)
        q_cat = torch.cat([q_word, q_char], dim=-1)

        # print("c cat size:", c_cat.size())
        # print("q cat size:", q_cat.size())
        # print("hidden size:", self.hidden_size)

        c = self.hwy(c_cat)
        q = self.hwy(q_cat)

        # (batch_size, c_len, 2 * hidden_size)
        c_enc = self.enc(c, c_len)
        # (batch_size, q_len, 2 * hidden_size)
        q_enc = self.enc(q, q_len)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask, )    # (batch_size, c_len, 8 * hidden_size)

        # (batch_size, c_len, 2 * hidden_size)
        mod = self.mod(att, c_len)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out, f
