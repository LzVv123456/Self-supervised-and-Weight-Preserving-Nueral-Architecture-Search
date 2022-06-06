import math
import torch
import random
import numpy as np
import torch.nn as nn
from utils.pytorch_utils import detach_variable, delta_ij
from operations import OPS
from torch.nn.parameter import Parameter


class MixedEdge(nn.Module):
    def __init__(self, channel, stride, mode, candidates, affine):
        super(MixedEdge, self).__init__()
        self.mode = mode
        self._ops = nn.ModuleList()
        self.shortcut = None
        self.active_index = [0]
        self.inactive_index = None

        for primitive in candidates:
            op = OPS[primitive](channel, stride, affine)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(channel, affine=False))
            self._ops.append(op)
        
        self.n_choices = len(self._ops)
        self.binary_gates = Parameter(torch.zeros(self.n_choices))  # binary gates
        self.alpha = Parameter(torch.zeros(self.n_choices))  # arch params

        self.freezed = False
    
    def binarize(self, skip_drop_prob, pool_drop_prob, warmup):
        if not self.freezed:
            # reset binary gates
            self.binary_gates.data.zero_()
            probs = torch.softmax(self.alpha, dim=0)
            if self.mode == 'two':
                # sample two ops according to probs
                sample_op = torch.multinomial(probs.data, 2, replacement=False).tolist()

                if not warmup:
                    # here random drop out non-parameter operation given a dropout rate
                    probs[:3] = 0 
                    if not set(sample_op).isdisjoint({1, 2}) and random.random() < pool_drop_prob:
                        sample_op = torch.multinomial(probs.data, 2, replacement=False).tolist()
                    if 0 in sample_op and random.random() < skip_drop_prob:
                        sample_op = torch.multinomial(probs.data, 2, replacement=False).tolist()
                        
                probs_slice = torch.softmax(torch.stack([self.alpha[idx] for idx in sample_op]), dim=0)
                # choose one to be active and the other to be inactive according to probs_slice
                chosen = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
                active_op = sample_op[chosen]
                inactive_op = sample_op[1 - chosen]
                # get active_op and inactive_op
                self.active_index = [active_op]
                self.inactive_index = [inactive_op]
                # set binary gates
                self.binary_gates.data[active_op] = 1.0
            
            elif self.mode == 'full_v2':
                sample_op = torch.multinomial(probs.data, 1)[0].item()

                if not warmup:
                    probs[:3] = 0
                    if sample_op in [1, 2] and random.random() < pool_drop_prob:
                        sample_op = torch.multinomial(probs.data, 1)[0].item()
                    if sample_op in [0] and random.random() < skip_drop_prob:
                        sample_op = torch.multinomial(probs.data, 1)[0].item()

                self.active_index = [sample_op]
                self.inactive_index = [_i for _i in range(0, sample_op)] + [_i for _i in range(sample_op + 1, self.n_choices)]
                # set binary gate
                self.binary_gates.data[sample_op] = 1.0

            # avoid over-regularization
            for _i in range(len(probs)):
                for name, param in self._ops[_i].named_parameters():
                    param.grad = None
    
    def forward(self, x):
        if self.mode == 'two':
            output = 0
            # Only 2 of N op weights input, and only activate one op
            for _i in self.active_index:
                oi = self._ops[_i](x)
                output = output + self.binary_gates[_i] * oi
            for _i in self.inactive_index:
                oi = self._ops[_i](x)
                output = output + self.binary_gates[_i] * oi.detach()
        elif self.mode == 'full_v2':
            def run_function(candidate_ops, active_id):
                def forward(_x):
                    return candidate_ops[active_id](_x)
                return forward

            def backward_function(candidate_ops, active_id, binary_gates):
                def backward(_x, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        for k in range(len(candidate_ops)):
                            if k != active_id:
                                out_k = candidate_ops[k](_x.data)
                            else:
                                out_k = _output.data
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grads[k] = grad_k
                    return binary_grads
                return backward
            
            output = ArchGradientFunction.apply(
                x, self.binary_gates, run_function(self._ops, self.active_index[0]),
                backward_function(self._ops, self.active_index[0], self.binary_gates)
            )

        return output
    
    def set_arch_param_grad(self):
        if not self.freezed:
            if self.alpha.grad is None:
                self.alpha.grad = torch.zeros_like(self.alpha.data)
            binary_grads = self.binary_gates.grad.data

            if self.mode == 'two':
                involved_idx = self.active_index + self.inactive_index
                probs_slice = torch.softmax(torch.stack([self.alpha[idx] for idx in involved_idx]), dim=0).data
                for i in range(2):
                    for j in range(2):
                        origin_i = involved_idx[i]
                        origin_j = involved_idx[j]
                        self.alpha.grad.data[origin_i] += binary_grads[origin_j] * probs_slice[j] * (delta_ij(i,j) - probs_slice[i])
                for _i, idx in enumerate(self.active_index):
                    self.active_index[_i] = (idx, self.alpha.data[idx].item())
                for _i, idx in enumerate(self.inactive_index):
                    self.inactive_index[_i] = (idx, self.alpha.data[idx].item())
            
            elif self.mode == 'full_v2':
                probs = torch.softmax(self.alpha, dim=0)
                for i in range(self.n_choices):
                    for j in range(self.n_choices):
                        self.alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])

    
    def rescale_updated_arch_param(self):
        if not self.freezed:
            if not isinstance(self.active_index[0], tuple):
                return
            involved_idx = [idx for idx, _ in self.active_index + self.inactive_index]
            old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
            new_alphas = [self.alpha.data[idx] for idx in involved_idx]

            offset = math.log(sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas]))
            
            for idx in involved_idx:
                self.alpha.data[idx] -= offset
    
    def set_chosen_op_active(self):
        if not self.freezed:
            index = int(np.argmax(self.alpha.detach().cpu().numpy()))
            self.active_index = [index]
            self.inactive_index = [_ for _ in range(0, index)] + [_ for _ in range(index+1, self.n_choices)]
    
    def freeze(self):
        if not self.freezed:
            self.set_chosen_op_active()  # set active and inactive index

            self.alpha.grad = None
            self.alpha.requires_grad_(False)

            self.binary_gates.data.zero_()
            self.binary_gates.data[self.active_index[0]] = 1.0
            self.binary_gates.grad.zero_()
            self.binary_gates.requires_grad_(False)

            self.freezed = True


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x)
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None
