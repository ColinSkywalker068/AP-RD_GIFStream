from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, device

def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

def fake_quantize(input: Tensor, lower_bd: float, upper_bd: float, bitwidth: int = 8, q_type: str = "round"):
    q_step = (upper_bd - lower_bd) / (2**bitwidth - 1)

    clamp_input = torch.clamp(input, lower_bd, upper_bd)

    shift_scale_value = (clamp_input - lower_bd) / q_step

    if q_type == "round":
        q_level = torch.round(shift_scale_value) # quantized level: range: [0, 2**bitwidth - 1]
    elif q_type == "noise":
        raise NotImplementedError
    else:
        raise NotImplementedError

    fq_value = q_level * q_step + lower_bd  # fake quantized value: range: [lower_bd, upper_bd]

    output_value = (fq_value - input).detach() + input

    out_dict = {
        "output_value": output_value, 
        "q_step": q_step # return q_step as Q for entropy model
    }

    return out_dict

def fake_quantize_ste(input: Tensor, lower_bd: float, upper_bd: float, bitwidth: int = 8, q_type: str = "noise"):
    q_step = (upper_bd - lower_bd) / (2**bitwidth - 1)

    if q_type == "round":
        output_value = STE.apply(input, bitwidth, lower_bd, upper_bd)
    elif q_type == "noise":
        input = torch.clamp(input, lower_bd, upper_bd)
        noise = torch.empty_like(input).uniform_(-0.5, 0.5)
        output_value = input + noise * q_step # whether to exclude the data pts that overflows/underflows?

    out_dict = {
        "output_value": output_value, 
        "q_step": q_step # return q_step as Q for entropy model
    }

    return out_dict

def fake_quantize_scaling(input: Tensor, scaling: float, q_type: str = "noise"):

    if q_type == "round":
        output_value = (torch.round(input / scaling) * scaling - input).detach() + input
    elif q_type == "noise":
        noise = torch.empty_like(input).uniform_(-0.5, 0.5)
        output_value = input + noise * scaling # whether to exclude the data pts that overflows/underflows?

    out_dict = {
        "output_value": output_value, 
        "q_step": 1 # return q_step as Q for entropy model
    }

    return out_dict

def fake_quantize_factors(input: Tensor, scaling: float = 1/16, q_type: str = "round", q_aware=True, qs=None):
    def fake_quantize(rounded, origin):
        return (rounded - origin).detach() + origin
    
    output_value = torch.sigmoid(input)
    if q_aware:
        if q_type == "round":
            if qs is not None:
                output_value = fake_quantize(torch.round(output_value / scaling * qs) * scaling / qs, output_value)
            else:
                output_value = fake_quantize(torch.round(output_value / scaling) * scaling, output_value)
        elif q_type == "noise":
            noise = torch.empty_like(output_value).uniform_(-0.5, 0.5)
            if qs is not None:
                output_value = output_value + noise * scaling / qs
            else:
                output_value = output_value + noise * scaling
    output_value = torch.cat([
        fake_quantize(torch.where(output_value[:,:-1] < 0.1, 0., output_value[:,:-1]), output_value[:,:-1]),
        fake_quantize(torch.where(output_value[:,-1] < 0.1, 0., 1.0), output_value[:,-1]).unsqueeze(-1)
    ],dim=-1)

    return output_value

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bitdepth = 8, min = -1, max = 1):
        maxs = max # 2.5
        mins = min # -10

        input = input.clamp_(mins, maxs)

        norm_input = (input - mins) / (maxs - mins)
        q_step = 1 / (2**bitdepth - 1)
        q_norm_input = (norm_input / q_step).round() * q_step

        output = q_norm_input * (maxs - mins) + mins

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

    
class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask