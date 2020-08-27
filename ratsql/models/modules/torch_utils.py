import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

def add_params(size):
    """ Adds parameters to the model.

    Inputs:
        model (dy.ParameterCollection): The parameter collection for the model.
        size (tuple of int): The size to create.
        name (str, optional): The name of the parameters.
    """
    size_int = tuple([int(ss) for ss in size])
    return torch.nn.Parameter(torch.empty(size_int).uniform_(-0.1, 0.1))


def create_multilayer_lstm_params(num_layers, in_size, state_size):
    """ Adds a multilayer LSTM to the model parameters.

    Inputs:
        num_layers (int): Number of layers to create.
        in_size (int): The input size to the first layer.
        state_size (int): The size of the states.
        model (dy.ParameterCollection): The parameter collection for the model.
        name (str, optional): The name of the multilayer LSTM.
    """
    lstm_layers = []
    for i in range(num_layers):
        lstm_layer = torch.nn.LSTMCell(input_size=int(in_size), hidden_size=int(state_size), bias=True)
        lstm_layers.append(lstm_layer)
        in_size = state_size
    return torch.nn.ModuleList(lstm_layers)

def linear_layer(exp, weights, biases=None):
    # exp: input as size_1 or 1 x size_1
    # weight: size_1 x size_2
    # bias: size_2
    if exp.dim() == 1:
        exp = torch.unsqueeze(exp, 0)
    assert exp.size()[1] == weights.size()[0]
    if biases is not None:
        assert weights.size()[1] == biases.size()[0]
        result = torch.mm(exp, weights) + biases
    else:
        result = torch.mm(exp, weights)
    return result


def forward_one_multilayer(rnns, lstm_input, layer_states, dropout_amount=0.):
    """ Goes forward for one multilayer RNN cell step.

    Inputs:
        lstm_input (dy.Expression): Some input to the step.
        layer_states (list of dy.RNNState): The states of each layer in the cell.
        dropout_amount (float, optional): The amount of dropout to apply, in
            between the layers.

    Returns:
        (list of dy.Expression, list of dy.Expression), dy.Expression, (list of dy.RNNSTate),
        representing (each layer's cell memory, each layer's cell hidden state),
        the final hidden state, and (each layer's updated RNNState).
    """
    num_layers = len(layer_states)
    new_states = []
    cell_states = []
    hidden_states = []
    state = lstm_input
    for i in range(num_layers):
        # view as (1, input_size)
        layer_h, layer_c = rnns[i](torch.unsqueeze(state,0), layer_states[i])
        new_states.append((layer_h, layer_c))

        layer_h = layer_h.squeeze()
        layer_c = layer_c.squeeze()

        state = layer_h
        if i < num_layers - 1:
            # In both Dynet and Pytorch
            # p stands for probability of an element to be zeroed. i.e. p=1 means switch off all activations.
            state = F.dropout(state, p=dropout_amount)

        cell_states.append(layer_c)
        hidden_states.append(layer_h)

    return (cell_states, hidden_states), state, new_states