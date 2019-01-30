#!/usr/bin/env python
from overrides import overrides
import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class DummyContextEncoder(Seq2SeqEncoder):
    def __init__(self):
        super(DummyContextEncoder, self).__init__(False)
        self._dim = None

    @overrides
    def forward(self, inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
        if self._dim is None:
            self._dim = inputs.size()[-1]
        return inputs

    @overrides
    def get_input_dim(self) -> int:
        return self._dim

    @overrides
    def get_output_dim(self) -> int:
        return self._dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False
