#!/usr/bin/env python
from typing import Tuple
import torch
import logging
from .input_embed_base import InputEmbedderBase
from .embeddings import Embeddings
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.nn.util import get_mask_from_sequence_lengths
logger = logging.getLogger(__name__)


class LstmTokenEmbedder(InputEmbedderBase):
    def __init__(self, input_field_name: str,
                 output_dim: int,
                 embeddings: Embeddings,
                 dropout: float,
                 use_cuda: bool):
        super(LstmTokenEmbedder, self).__init__(input_field_name)
        self.embeddings = embeddings
        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.encoder_ = torch.nn.LSTM(embeddings.get_embed_dim(), embeddings.get_embed_dim(),
                                      num_layers=1, bidirectional=False,
                                      batch_first=True, dropout=dropout)
        self.attention = MultiHeadSelfAttention(num_heads=1,
                                                input_dim=embeddings.get_embed_dim(),
                                                attention_dim=embeddings.get_embed_dim(),
                                                values_dim=embeddings.get_embed_dim(),
                                                attention_dropout_prob=dropout)

    def forward(self, input_: Tuple[torch.Tensor, torch.Tensor]):
        chars, lengths = input_
        batch_size, seq_len, max_chars = chars.size()

        chars = chars.view(batch_size * seq_len, -1)
        lengths = lengths.view(batch_size * seq_len)
        mask = get_mask_from_sequence_lengths(lengths, max_chars)
        chars = torch.autograd.Variable(chars, requires_grad=False)

        embeded_chars = self.embeddings(chars)
        output, _ = self.encoder_(embeded_chars)

        output = self.attention(output, mask).sum(dim=-2)

        return output.view(batch_size, seq_len, -1)

    def get_embed_dim(self):
        return self.output_dim
