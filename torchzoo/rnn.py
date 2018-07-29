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


class CorefGRU(nn.Module):
    """
    Coreference GRU
    https://arxiv.org/pdf/1804.05922.pdf
    """
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.w_z = nn.Linear(inp_dim, out_dim)
        self.u_z = nn.Linear(out_dim, out_dim, bias=False)
        self.w_r = nn.Linear(inp_dim, out_dim)
        self.u_r = nn.Linear(inp_dim, out_dim, bias=False)
        self.w_h = nn.Linear(inp_dim, out_dim)
        self.u_h = nn.Linear(out_dim, out_dim, bias=False)
        self.out_dim = out_dim
        self.inp_dim = inp_dim

    def forward(self, inp, last_coref_idx):
        h_last = torch.zeros((inp.size()[0], self.out_dim))
        hidden_states = []
        for i in range(inp.size()[1]):
            xi, idx = inp[i], last_coref_idx[i]
            # -------
            z = self.w_z(xi) + self.u_z(h_last)
            z = nn.Sigmoid()(z)
            # -------
            r = self.w_r(xi) + self.u_r(h_last)
            r = nn.Sigmoid()(r)
            # --------
            z_partial = self.w_z(x) + self.u_z(r * h_last)
            h = (1 - z) * h_last + z * nn.Tanh()(z_partial)
            # --------
            h_last = h
            hidden_states.append(h_last)
        return inp
