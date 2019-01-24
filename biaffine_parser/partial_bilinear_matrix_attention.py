#!/usr/bin/evn python
from overrides import overrides
import torch
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import Activation


class PartialBilinearMatrixAttention(MatrixAttention):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
    and ``Y`` is computed as ``X W Y^T + b``.

    Parameters
    ----------
    matrix_1_dim : ``int``
        The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : ``int``
        The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``X W Y^T + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 activation: Activation = None) -> None:
        super(PartialBilinearMatrixAttention, self).__init__()
        matrix_2_dim += 1

        self._weight_matrix = torch.nn.Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))

        self._bias = torch.nn.Parameter(torch.Tensor(1))
        self._activation = activation or Activation.by_name('linear')()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0.)

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))
        matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        return self._activation(final.squeeze(1) + self._bias)
