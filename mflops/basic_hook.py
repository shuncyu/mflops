'''
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import numpy as np
import torch.nn as nn

# ---- Internal functions
def empty_compute_hook(module, input, output):
    module.__flops__ += 0
    module.__mac__ += 0


def conv_compute_hook(module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    
    # Caculate convs FLOPs and MAC
    batch_size = input.shape[0]
    output_dims = output.shape[2:]
    
    kernel_dims = module.kernel_size
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups

    conv_per_position_flops = np.prod(kernel_dims) * in_channels * out_channels / groups
    active_elements_count = batch_size * np.prod(output_dims)
    
    # conv_flops = h_out * w_out * k^2 * c_in * c_out / g
    overall_flops = conv_per_position_flops * active_elements_count
    
    overall_mac = np.prod(input.shape[1:]) + np.prod(output.shape[1:]) + \
        conv_per_position_flops
        
    bias_flops = 0
    bias_mac = 0
    if module.bias is not None:
        # bias_flops = c_out * h_out * w_out
        bias_flops = out_channels * active_elements_count
        bias_mac = np.prod(output.shape[1:])
    overall_flops += bias_flops
    overall_mac += bias_mac
    
    module.__flops__ += overall_flops
    module.__mac__ += overall_mac
    

def bn_compute_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    batch_mac = 2 * batch_flops
    
    if module.affine:
        batch_flops *= 2
        batch_mac += 2 * input.shape[1]

    module.__flops__ += batch_flops
    module.__mac__ += batch_mac


def relu_compute_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += active_elements_count
    module.__mac__ += input[0].numel() + output.numel()
    

def pool_compute_hook(module, input, output):
    input = input[0]

    module.__flops__ += np.prod(output.shape[1:])
    
    pool_mac = np.prod(input.shape) + np.prod(output.shape[1:])
    module.__mac__ += pool_mac
    
    
def upsample_compute_hook(module, input, output):
    module.__flops__ += np.prod(output.shape[1:])
    module.__mac__ += np.prod(input[0].shape) + np.prod(output.shape[1:])
    

def linear_compute_hook(module, input, output):
    input = input[0]
    
    output_last_dim = output.shape[-1]
    linear_flops = np.prod(input.shape) * output_last_dim
    module.__flops__ += linear_flops
    module.__mac__ += np.prod(input.shape) + output_last_dim + linear_flops


# ---- rnn flops
def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    IF sigmoid and tanh are made hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_compute_hook,
    nn.Conv2d: conv_compute_hook,
    nn.Conv3d: conv_compute_hook,
    # activations
    nn.ReLU: relu_compute_hook,
    nn.PReLU: relu_compute_hook,
    nn.ELU: relu_compute_hook,
    nn.LeakyReLU: relu_compute_hook,
    nn.ReLU6: relu_compute_hook,
    # poolings
    nn.MaxPool1d: pool_compute_hook,
    nn.AvgPool1d: pool_compute_hook,
    nn.AvgPool2d: pool_compute_hook,
    nn.MaxPool2d: pool_compute_hook,
    nn.MaxPool3d: pool_compute_hook,
    nn.AvgPool3d: pool_compute_hook,
    nn.AdaptiveMaxPool1d: pool_compute_hook,
    nn.AdaptiveAvgPool1d: pool_compute_hook,
    nn.AdaptiveMaxPool2d: pool_compute_hook,
    nn.AdaptiveAvgPool2d: pool_compute_hook,
    nn.AdaptiveMaxPool3d: pool_compute_hook,
    nn.AdaptiveAvgPool3d: pool_compute_hook,
    # BNs
    nn.BatchNorm1d: bn_compute_hook,
    nn.BatchNorm2d: bn_compute_hook,
    nn.BatchNorm3d: bn_compute_hook,
    # FC
    nn.Linear: linear_compute_hook,
    # Upscale
    nn.Upsample: upsample_compute_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_compute_hook,
    nn.ConvTranspose2d: conv_compute_hook,
    nn.ConvTranspose3d: conv_compute_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook
}
