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
        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.split_dim = out_dim // 2
        self.w_r = nn.Linear(inp_dim, out_dim)
        self.u_r = nn.Linear(out_dim, out_dim, bias=False)
        self.w_z = nn.Linear(inp_dim, out_dim)
        self.u_z = nn.Linear(out_dim, out_dim, bias=False)
        self.w_h = nn.Linear(inp_dim, out_dim)
        self.u_h = nn.Linear(out_dim, out_dim, bias=False)
        self.k1 = nn.Linear(inp_dim, 1, bias=False)
        self.k2 = nn.Linear(inp_dim, 1, bias=False)

    def forward(self, inp, last_coref_idx):
        """
        Expects inputs and a corresponding tensor of
        last coref index.

            inp shape B, L, D
            cor shape B, L
        """
        B = inp.size()[0]
        h_last = torch.zeros((B, self.out_dim))
        state_history = []
        for i in range(inp.size()[1]):
            xi, ci = inp[:, i], last_coref_idx[:, i]
            # ---------- generate mt
            coref = [state_history[idx-1][bi] if idx > 0 else torch.zeros((B, self.out_dim))[bi]
                     for bi, idx in enumerate(ci)]
            coref = torch.stack(coref, dim=0)
            #       ------- calculate a
            e1e2 = torch.cat([self.k1(xi), self.k2(xi)], dim=1)
            a = nn.Softmax(dim=1)(e1e2)
            a = torch.unsqueeze(a[:, 0], dim=1)
            p1, p2 = h_last[:, :self.split_dim], coref[:, self.split_dim:]
            mt = torch.cat([a * p1, (1 - a) * p2], dim=1)
            # ---------- generate r
            r = nn.Sigmoid()(self.w_r(xi) + self.u_r(mt))
            # ---------- generate z
            z = nn.Sigmoid()(self.w_z(xi) + self.u_z(mt))
            # ---------- generate h~
            h_ = nn.Tanh()(self.w_h(xi) + r * self.u_h(mt))
            h = (1 - z) * mt + z * h_
            state_history.append(h)
        return torch.stack(state_history, dim=1)
