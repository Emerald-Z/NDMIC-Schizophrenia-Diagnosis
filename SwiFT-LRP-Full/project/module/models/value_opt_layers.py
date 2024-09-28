import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'GELU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'einsum', 'Softmax', 'IndexSelect',
           'LayerNorm', 'AddEye']


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha, epsilon, gamma):
        return R

class RelPropSimple(RelProp):
    def relprop(self, R, alpha, epsilon, gamma):
        Z = self.forward(self.X)
        if self.label == "epsilon":
            Z += epsilon
        S = safe_divide(R, Z)

        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

# TODO: does this even work? would it add too much complexity for PSO?
"""class RelPropSimple(RelProp):
    def relprop(self, R, alpha=None, epsilon=None, gamma=None):
        Z = self.forward(self.X)
        
        if self.label == "epsilon":
            Z = Z + epsilon
            S = safe_divide(R, Z)
            C = self.gradprop(Z, self.X, S)
            if torch.is_tensor(self.X):
                outputs = self.X * C
            else:
                outputs = [self.X[i] * C[i] for i in range(len(self.X))]
            return outputs

        elif self.label == "alpha":
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.linear(x1, w1)
                Z2 = F.linear(x2, w2)
                S1 = safe_divide(R, Z1 + Z2)
                S2 = safe_divide(R, Z1 + Z2)
                C1 = x1 * torch.autograd.grad(Z1, x1, S1, retain_graph=True)[0]
                C2 = x2 * torch.autograd.grad(Z2, x2, S2, retain_graph=True)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)
            R = alpha * activator_relevances - beta * inhibitor_relevances

        elif self.label == "gamma":
            pw = torch.clamp(self.weight, min=0)
            
            def f(w, x):
                Z = F.linear(x, w)
                S = safe_divide(R, Z)
                C = x * torch.autograd.grad(Z, x, S, retain_graph=True)[0]
                return C

            modified_weight = self.weight + gamma * pw
            R = f(modified_weight, self.X)

        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X):
            outputs = self.X * C
        else:
            outputs = [self.X[i] * C[i] for i in range(len(self.X))]

        return outputs"""
    
# Not done
# class RelPropGamma(RelProp):
#     def relprop(self, R, epsilon):
#         Z = epsilon + self.forward(self.X)
#         S = safe_divide(R, Z)
#         C = self.gradprop(Z, self.X, S) 

#         if torch.is_tensor(self.X) == False:
#             outputs = []
#             outputs.append(self.X[0] * C[0])
#             outputs.append(self.X[1] * C[1])
#         else:
#             outputs = self.X * (C[0])
#         return outputs
    
class AddEye(RelPropSimple):
    def __init__(self, labels="alpha", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label

    # input of shape B, C, seq_len, seq_len
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)

class ReLU(nn.ReLU, RelProp):
    pass

class GELU(nn.GELU, RelProp):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels
        
    pass

class Softmax(nn.Softmax, RelProp):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    pass

class LayerNorm(nn.LayerNorm, RelProp):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    pass

class Dropout(nn.Dropout, RelProp):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass

class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass

# New
class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, RelPropSimple):
    pass

class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, RelPropSimple):
    pass

class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass

class Parameter(nn.Parameter, RelProp):
    pass


class Add(RelPropSimple):
    def __init__(self, labels="alpha", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha, epsilon, gamma):
        Z = self.forward(self.X)
        if self.label == "epsilon":
            Z += epsilon
            
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs

class einsum(RelPropSimple):
    def __init__(self, equation, labels="alpha", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

# not sure this is used
class IndexSelect(RelProp):
    def __init__(self, labels="alpha", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha, epsilon, gamma):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs



class Clone(RelProp):
    def __init__(self, labels="alpha", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha, epsilon, gamma):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R

class Cat(RelProp):
    def __init__(self, labels="alpha", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha, epsilon, gamma):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs


class Sequential(nn.Sequential):
    def relprop(self, R, alpha, epsilon, gamma):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R
# not used
class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def __init__(self, labels="alpha", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = labels

    # this is epsilon
    def relprop(self, R, alpha, epsilon, gamma):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        
        if self.label == "gamma":
            pw = torch.clamp(weight, min=0)
            weight += alpha * pw
            Z = X * weight
            # TODO: use self.X and self.weight to manually calculate "forward prop"

        elif self.label == "alpha":
            pass
        else: 
            Z = X * weight
            if self.label == "epsilon":
                Z += alpha #1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


class Linear(nn.Linear, RelProp):
    def __init__(self, labels, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.label = labels

    def relprop(self, R, alpha, epsilon, gamma):
        if self.label == "zero":
            alpha = 1
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        if self.label == "alpha" or self.label == "zero":
            def f(w1, w2, x1, x2):
                Z1 = F.linear(x1, w1)
                Z2 = F.linear(x2, w2)
                S1 = safe_divide(R, Z1 + Z2)
                S2 = safe_divide(R, Z1 + Z2)
                C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
                C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)
            R = alpha * activator_relevances - beta * inhibitor_relevances
        elif self.label == "gamma": # might have to redo it
            def f(w1, x1):
                Z = F.linear(x1, w1)
                S = safe_divide(R, Z)
                C = x1 * torch.autograd.grad(Z, x1, S)[0]
                return C
            R = f(self.weight + pw * gamma, self.X)

        elif self.label == "epsilon": # TODO: check
            # TODO: new! has anyone done sign-epsilon?
            Z_0 = F.linear(self.X, self.weight)
            Z = Z_0 + epsilon * torch.sign(Z_0)
            S = safe_divide(R, Z)
            R = self.X * torch.autograd.grad(Z, self.X, S)[0]
        return R

# not used
class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha, epsilon, gamma):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R
