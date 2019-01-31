from typing import Optional, Tuple
import torch
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_dropout_mask
from allennlp.nn.initializers import block_orthogonal
from allennlp.nn.activations import Activation
from allennlp.modules.input_variational_dropout import InputVariationalDropout


class LstmCellBase(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 activation: Activation,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0):
        super(LstmCellBase, self).__init__()
        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.go_forward = go_forward
        self.activation = activation
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.input_dropout_ = InputVariationalDropout(p=recurrent_dropout_probability)


class DozatLstmCell(LstmCellBase):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 activation: Activation,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0) -> None:
        super(DozatLstmCell, self).__init__(input_size, hidden_size, activation, go_forward,
                                            recurrent_dropout_probability)

        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        self.input_linearity = torch.nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError('inputs must be PackedSequence but got %s' % (type(inputs)))

        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]

        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.hidden_size)
        if initial_state is None:
            full_batch_previous_state = sequence_tensor.new_zeros(batch_size, self.hidden_size)
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_state)
            sequence_tensor = self.input_dropout_(sequence_tensor)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (len(batch_lengths) - 1) and \
                                batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # Actually get the slices of the batch which we need for the computation at this timestep.
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            timestep_input = sequence_tensor[0: current_length_index + 1, index]

            # Do the projections for all the gates all at once.
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                       projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                        projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                     projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                        projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            timestep_output = output_gate * self.activation(memory)

            # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
            if dropout_mask is not None and self.training:
                timestep_output = timestep_output * dropout_mask[0: current_length_index + 1]

            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)

        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state


class MaLstmCell(LstmCellBase):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 activation: Activation,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0) -> None:
        super(MaLstmCell, self).__init__(input_size, hidden_size, activation, go_forward,
                                         recurrent_dropout_probability)
        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        self.input_linearities = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, hidden_size, bias=False) for _ in range(4)])
        self.state_linearities = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size, bias=True) for _ in range(4)])
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for i in range(4):
            torch.nn.init.uniform_(self.input_linearities[i].weight, -stdv, stdv)
            torch.nn.init.uniform_(self.state_linearities[i].weight, -stdv, stdv)
            self.state_linearities[i].bias.data.fill_(0.0 if i != 1 else 1.0)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if not isinstance(inputs, PackedSequence):
            raise ConfigurationError('inputs must be PackedSequence but got %s' % (type(inputs)))

        sequence_tensor, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        batch_size = sequence_tensor.size()[0]
        total_timesteps = sequence_tensor.size()[1]

        output_accumulator = sequence_tensor.new_zeros(batch_size, total_timesteps, self.hidden_size)
        if initial_state is None:
            full_batch_previous_state = sequence_tensor.new_zeros(batch_size, self.hidden_size)
            full_batch_previous_memory = sequence_tensor.new_zeros(batch_size, self.hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_masks = [get_dropout_mask(self.recurrent_dropout_probability, full_batch_previous_state)
                             for _ in range(4)]
            sequence_tensor = self.input_dropout_(sequence_tensor)
        else:
            dropout_masks = None

        for timestep in range(total_timesteps):
            # The index depends on which end we start.
            index = timestep if self.go_forward else total_timesteps - timestep - 1

            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            # If we're going backwards, we are _picking up_ more indices.
            else:
                # First conditional: Are we already at the maximum number of elements in the batch?
                # Second conditional: Does the next shortest sequence beyond the current batch
                # index require computation use this timestep?
                while current_length_index < (len(batch_lengths) - 1) and \
                                batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            # Actually get the slices of the batch which we need for the computation at this timestep.
            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            timestep_input = sequence_tensor[0: current_length_index + 1, index]

            # Main LSTM equations using relevant chunks of the big linear
            # projections of the hidden state and inputs.
            if self.recurrent_dropout_probability > 0.0 and self.training:
                input_gate = torch.sigmoid(self.input_linearities[0](timestep_input) +
                                           self.state_linearities[0](previous_state * dropout_masks[0][0: current_length_index + 1]))
                forget_gate = torch.sigmoid(self.input_linearities[1](timestep_input) +
                                            self.state_linearities[1](previous_state * dropout_masks[1][0: current_length_index + 1]))
                memory_init = torch.tanh(self.input_linearities[2](timestep_input) +
                                         self.state_linearities[2](previous_state * dropout_masks[2][0: current_length_index + 1]))
                output_gate = torch.sigmoid(self.input_linearities[3](timestep_input) +
                                            self.state_linearities[3](previous_state * dropout_masks[3][0: current_length_index + 1]))
            else:
                input_gate = torch.sigmoid(self.input_linearities[0](timestep_input) +
                                           self.state_linearities[0](previous_state))
                forget_gate = torch.sigmoid(self.input_linearities[1](timestep_input) +
                                            self.state_linearities[1](previous_state))
                memory_init = torch.tanh(self.input_linearities[2](timestep_input) +
                                         self.state_linearities[2](previous_state))
                output_gate = torch.sigmoid(self.input_linearities[3](timestep_input) +
                                            self.state_linearities[3](previous_state))

            memory = input_gate * memory_init + forget_gate * previous_memory
            timestep_output = output_gate * self.activation(memory)

            full_batch_previous_memory = full_batch_previous_memory.data.clone()
            full_batch_previous_state = full_batch_previous_state.data.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        output_accumulator = pack_padded_sequence(output_accumulator, batch_lengths, batch_first=True)

        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state


class InputDropoutedStackedBidirectionalLstm(torch.nn.Module):
    """
    Dozat's implementation of stacked bidirectional lstm. This was based on allennlp's
    stacked bidirectional lstm implementation.
    """
    def __init__(self,
                 cell_func: LstmCellBase,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 activation: Activation,
                 recurrent_dropout_probability: float = 0.0) -> None:
        super(InputDropoutedStackedBidirectionalLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):

            forward_layer = cell_func(lstm_input_size, hidden_size,
                                      activation=activation, go_forward=True,
                                      recurrent_dropout_probability=recurrent_dropout_probability)
            backward_layer = cell_func(lstm_input_size, hidden_size,
                                       activation=activation, go_forward=False,
                                       recurrent_dropout_probability=recurrent_dropout_probability)

            lstm_input_size = hidden_size * 2
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            layers.append([forward_layer, backward_layer])
        self.lstm_layers = layers

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        output_sequence = inputs
        final_states = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(i))
            backward_layer = getattr(self, 'backward_layer_{}'.format(i))

            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            backward_output, final_backward_state = backward_layer(output_sequence, state)

            forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)

            output_sequence = torch.cat([forward_output, backward_output], -1)
            output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)
            final_states.append((torch.cat(both_direction_states, -1) for both_direction_states
                                 in zip(final_forward_state, final_backward_state)))

        final_state_tuple = (torch.cat(state_list, 0) for state_list in zip(*final_states))
        return output_sequence, final_state_tuple
