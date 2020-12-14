'''
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from functools import partial

import torch
import torch.nn as nn
import prettytable as pt

from .basic_hook import MODULES_MAPPING

def get_model_compute_info(model, input_res,
                           print_per_layer_stat=False,
                           input_constructor=None, ost=sys.stdout,
                           verbose=False, ignore_modules=[],
                           custom_modules_hooks={}):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    compute_model = add_computing_methods(model)
    compute_model.eval()
    compute_model.start_compute(ost=ost, verbose=verbose, ignore_list=ignore_modules)
    if input_constructor:
        input = input_constructor(input_res)
        _ = compute_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(compute_model.parameters()).dtype,
                                             device=next(compute_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = compute_model(batch)

    flops_count, mac_count, params_count = compute_model.compute_average_compute_cost()
    
    if print_per_layer_stat:
        print_model_with_compute(compute_model, flops_count, mac_count, params_count, ost=ost)
    compute_model.stop_compute()
    CUSTOM_MODULES_MAPPING = {}
    
    tb = pt.PrettyTable()
    tb.field_names = ['Metrics', 'Value']
    tb.add_row(['%s' %'Floating Point Operations (FLOPs)', '%8s' %to_string(flops_count)])
    tb.add_row(['%s' %'Memory Access Cost (MAC)', '%8s' %to_string(mac_count)])
    tb.add_row(['%s' %'Number of Parameters', '%8s' %to_string(params_count)])
    print(tb)
    
    return flops_count, mac_count, params_count


def to_string(params_num, units=None, precision=3):
    if units is None:
        if params_num // 10**9 > 0:
            return str(round(params_num / 10**9, 3)) + ' G'
        elif params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 3)) + ' M'
        elif params_num // 10**3 > 0:
            return str(round(params_num / 10**3, 3)) + ' K'
        else:
            return str(params_num)
    else:
        if units == 'G':
            return str(round(params_num / 10**9, precision)) + ' ' + units
        if units == 'M':
            return str(round(params_num / 10**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10**3, precision)) + ' ' + units
        else:
            return str(params_num)


def print_model_with_compute(model, total_flops, total_mac, total_params, units='M',
                             precision=3, ost=sys.stdout):

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum
        
    def accumulate_mac(self):
        if is_supported_instance(self):
            return self.__mac__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_mac()
            return sum

    def compute_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops()
        accumulated_mac_cost = self.accumulate_mac()
        return ', '.join([to_string(accumulated_params_num,
                                    units=units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          to_string(accumulated_flops_cost,
                                    units=units, precision=precision),
                          '{:.3%} FLOPs'.format(accumulated_flops_cost / total_flops),
                          to_string(accumulated_mac_cost,
                                    units=units, precision=precision),
                          '{:.3%} MAC'.format(accumulated_mac_cost / total_mac),
                          '{:.3} MAC/FLOPs'.format(accumulated_mac_cost / (accumulated_flops_cost + 1e-5) \
                                                    * total_flops / (total_mac + 1e-5)),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_mac = accumulate_mac.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        compute_extra_repr = compute_repr.__get__(m)
        if m.extra_repr != compute_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = compute_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
        if hasattr(m, 'accumulate_mac'):
            del m.accumulate_mac

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_computing_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_compute = start_compute.__get__(net_main_module)
    net_main_module.stop_compute = stop_compute.__get__(net_main_module)
    net_main_module.reset_compute = reset_compute.__get__(net_main_module)
    net_main_module.compute_average_compute_cost = compute_average_compute_cost.__get__(
        net_main_module)

    net_main_module.reset_compute()

    return net_main_module


def compute_average_compute_cost(self):
    """
    A method that will be available after add_computing_methods() is called
    on a desired net object.

    Returns current mean flops/mac consumption per image.

    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    mac_sum = 0
    params_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
            mac_sum += module.__mac__
    params_sum = get_model_parameters_number(self)
    return flops_sum / batches_count, mac_sum / batches_count, params_sum


def start_compute(self, **kwargs):
    """
    A method that will be available after add_computing_methods() is called
    on a desired net object.

    Activates the computation of mean flops/mac consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_compute_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            module.__mac_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_compute_hook_function, **kwargs))


def stop_compute(self):
    """
    A method that will be available after add_computing_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_compute_hook_function)


def reset_compute(self):
    """
    A method that will be available after add_computing_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_compute_variable_or_reset)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size
    
def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_compute_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__mac__') or \
            hasattr(module, '__params__'):
            print('Warning: variables __flops__ or __mac__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
        module.__flops__ = 0
        module.__mac__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_compute_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
        if hasattr(module, '__mac_handle__'):
            module.__mac_handle__.remove()
            del module.__mac_handle__
