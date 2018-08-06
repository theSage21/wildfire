import math
import torch
import torch.nn as nn


class RWA(nn.Module):
    """
    Recurrent Weighted Average
    https://arxiv.org/pdf/1703.01253.pdf
    """
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()
        self.activation = nn.Tanh() if activation is None else activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._u = nn.Linear(input_dim, output_dim)
        self._g = nn.Linear(input_dim+output_dim, output_dim)
        self._a = nn.Linear(input_dim+output_dim, output_dim, bias=False)
        self.s0 = nn.parameter.Parameter(torch.Tensor(output_dim,))
        # ----------- initialize weights
        self.s0 = nn.init.normal_(self.s0, 0, 1)
        self._u.bias.data.fill_(0)
        self._g.bias.data.fill_(0)

        low = - math.sqrt(6 / (input_dim + output_dim))
        high = - low
        self._u.weight.data = nn.init.uniform_(self._u.weight.data, low, high)
        self._g.weight.data = nn.init.uniform_(self._g.weight.data, low, high)
        self._a.weight.data = nn.init.uniform_(self._a.weight.data, low, high)

    def forward(self, x):
        s0 = torch.stack([self.s0]*x.size()[0], dim=0)
        # ------------keep track of these
        last_h = nn.Tanh()(s0)
        numerator = torch.zeros((x.size()[0], self.output_dim))
        denominator = torch.zeros((x.size()[0], self.output_dim))
        last_a_max = torch.ones((x.size()[0], self.output_dim)) * 1e-38
        # ------------initialization done
        U = self._u(x)
        outputs = []
        for idx in range(x.size()[1]):
            xi = x[:, idx, :]
            ui = U[:, idx, :]
            xh = torch.cat([xi, last_h], dim=1)
            # ----- calculate Z and A
            z = ui * nn.Tanh()(self._g(xh))
            a = self._a(xh)
            a_max, _ = torch.max(torch.stack([a, last_a_max], dim=1), dim=1)
            # ----- calculate  num and den
            e_to_a = torch.exp(a_max - last_a_max)
            numerator = numerator + z * e_to_a
            denominator = denominator + e_to_a
            last_h = self.activation(numerator / denominator)
            last_a_max = a_max
            outputs.append(last_h)
        return torch.stack(outputs, dim=1)
