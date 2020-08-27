""" Contains code for encoding an input sequence. """

import torch
from ratsql.models.modules.torch_utils import create_multilayer_lstm_params, forward_one_multilayer


class Encoder(torch.nn.Module):
    """ Encodes an input sequence. """
    def __init__(self, num_layers, input_size, state_size):
        super().__init__()

        self.num_layers = num_layers
        self.forward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2)
        self.backward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2)

    def forward(self, sequence, embedder, dropout_amount=0.):
        """ Encodes a sequence forward and backward.
        Inputs:
            forward_seq (list of str): The string forwards.
            backward_seq (list of str): The string backwards.
            f_rnns (list of dy.RNNBuilder): The forward RNNs.
            b_rnns (list of dy.RNNBuilder): The backward RNNS.
            emb_fn (dict str->dy.Expression): Embedding function for tokens in the
                sequence.
            size (int): The size of the RNNs.
            dropout_amount (float, optional): The amount of dropout to apply.

        Returns:
            (list of dy.Expression, list of dy.Expression), list of dy.Expression,
            where the first pair is the (final cell memories, final cell states) of
            all layers, and the second list is a list of the final layer's cell
            state for all tokens in the sequence.
        """
        sequence = list(torch.split(sequence, split_size_or_sections=1, dim=0))
        # forward_list = []
        # backward_list = []
        # for i, tok in enumerate(sequence):
        #     forward_list.append(tok)
        #     backward_list.append(sequence[sequence.shape[0] - i -1])

        forward_state, forward_outputs = encode_sequence(
            sequence,
            self.forward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        backward_state, backward_outputs = encode_sequence(
            sequence[::-1],
            self.backward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        cell_memories = []
        hidden_states = []
        for i in range(self.num_layers):
            cell_memories.append(torch.cat([forward_state[0][i], backward_state[0][i]], dim=0))
            hidden_states.append(torch.cat([forward_state[1][i], backward_state[1][i]], dim=0))

        assert len(forward_outputs) == len(backward_outputs)

        backward_outputs = backward_outputs[::-1]

        final_outputs = []
        for i in range(len(sequence)):
            final_outputs.append(torch.cat([forward_outputs[i], backward_outputs[i]], dim=0))

        return (cell_memories, hidden_states), final_outputs


def encode_sequence(sequence, rnns, embedder, dropout_amount=0.):
    """ Encodes a sequence given RNN cells and an embedding function.

    Inputs:
        seq (list of str): The sequence to encode.
        rnns (list of dy._RNNBuilder): The RNNs to use.
        emb_fn (dict str->dy.Expression): Function that embeds strings to
            word vectors.
        size (int): The size of the RNN.
        dropout_amount (float, optional): The amount of dropout to apply.

    Returns:
        (list of dy.Expression, list of dy.Expression), list of dy.Expression,
        where the first pair is the (final cell memories, final cell states) of
        all layers, and the second list is a list of the final layer's cell
        state for all tokens in the sequence.
    """

    batch_size = 1
    layer_states = []
    for rnn in rnns:
        hidden_size = rnn.weight_hh.size()[1]
        # h_0 of shape (batch, hidden_size)
        # c_0 of shape (batch, hidden_size)
        if rnn.weight_hh.is_cuda:
            h_0 = torch.cuda.FloatTensor(batch_size, hidden_size).fill_(0)
            c_0 = torch.cuda.FloatTensor(batch_size, hidden_size).fill_(0)
        else:
            h_0 = torch.zeros(batch_size, hidden_size)
            c_0 = torch.zeros(batch_size, hidden_size)

        layer_states.append((h_0, c_0))

    outputs = []
    for token in sequence:
        token = token.squeeze()
        rnn_input = embedder(token)
        (cell_states, hidden_states), output, layer_states = forward_one_multilayer(rnns, rnn_input, layer_states,
                                                                                    dropout_amount)

        outputs.append(output)

    return (cell_states, hidden_states), outputs

