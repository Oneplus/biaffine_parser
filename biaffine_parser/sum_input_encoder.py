#!/usr/bin/env python
from typing import Dict, List
import torch
from .input_encoder_base import InputEncoderBase


class SummationInputEncoder(InputEncoderBase):
    def __init__(self, input_info_: Dict[str, int],
                 output_dim_: int,
                 use_cuda: bool):
        super(SummationInputEncoder, self).__init__(use_cuda)
        self.input_info_ = input_info_
        self.output_dim_ = output_dim_

        self.projections = {}
        self.ordered_names = []
        for i, (name, dim) in enumerate(input_info_.items()):
            if i == 0:
                self.projections[name] = torch.nn.Linear(dim, output_dim_)
            else:
                self.projections[name] = torch.nn.Linear(dim, output_dim_, bias=False)
            self.ordered_names.append(name)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        arbitrary_input_ = next(iter(inputs.values()))
        batch_size, seq_len, _ = arbitrary_input_.size()
        output = torch.zeros([batch_size, seq_len, self.output_dim_])
        if self.use_cuda:
            output = output.cuda()

        for name, projection in self.projections.items():
            output.add_(projection(inputs[name]))

        return output

    def get_output_dim(self) -> int:
        return self.output_dim_

    def get_ordered_names(self) -> List[str]:
        return self.ordered_names
