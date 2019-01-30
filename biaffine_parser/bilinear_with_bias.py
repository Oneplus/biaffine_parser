#!/usr/bin/env python
import torch


class BilinearWithBias(torch.nn.Module):

    def __init__(self, in1_features, in2_features, out_features):
        super(BilinearWithBias, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.bilinear_ = torch.nn.Bilinear(in1_features, in2_features, out_features, bias=False)
        self.linear1_ = torch.nn.Linear(in1_features, out_features, bias=False)
        self.linear2_ = torch.nn.Linear(in2_features, out_features)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.bilinear_.weight.data)
        torch.nn.init.xavier_uniform_(self.linear1_.weight.data)
        torch.nn.init.xavier_uniform_(self.linear2_.weight.data)
        self.linear2_.bias.data.fill_(0)

    def forward(self, input1, input2):
        return self.bilinear_(input1, input2) + self.linear1_(input1) + self.linear2_(input2)

    def extra_repr(self):
        return 'in1_features={}, in2_features={}, out_features={}'.format(
            self.in1_features, self.in2_features, self.out_features
        )
