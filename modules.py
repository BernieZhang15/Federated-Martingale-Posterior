import warnings
warnings.filterwarnings("ignore",message=".*Can't initialize NVML.*", category=UserWarning)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, dim_out, dim_hidden=128, num_heads=8):
        super(MAB, self).__init__()

        assert dim_hidden % num_heads == 0, "dim_hidden must be divisible by num_heads"

        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.head_dim = self.dim_hidden // self.num_heads

        self.fc_q = nn.Linear(dim_out, dim_hidden)
        self.fc_k = nn.Linear(dim_out, dim_hidden)
        self.fc_v = nn.Linear(dim_out, dim_hidden)

        self.ln0 = nn.LayerNorm(dim_hidden)
        self.ln1 = nn.LayerNorm(dim_hidden)

        self.fc_o = nn.Linear(dim_hidden, dim_hidden)
        self.fc_out = nn.Linear(dim_hidden, dim_out)


    def forward(self, Q, K):
        Q, K, V  = self.fc_q(Q), self.fc_k(K), self.fc_v(K)

        Q_ = torch.cat(Q.split(self.head_dim, -1), dim=0)
        K_ = torch.cat(K.split(self.head_dim, -1), dim=0)
        V_ = torch.cat(V.split(self.head_dim, -1), dim=0)

        logits = Q_ @ K_.transpose(-2,-1) / math.sqrt(self.head_dim)
        A = torch.softmax(logits, dim=-1)
        out = torch.cat((Q_ + A @ V_).split(Q.size(0), dim=0), dim=-1)
        out = self.ln0(out)
        out = self.ln1(out + F.relu(self.fc_o(out)))
        return self.fc_out(out)

class ISAB(nn.Module):
    def __init__(self, dim_out, dim_hidden=128, num_heads=8, num_inds=40):
        super(ISAB, self).__init__()

        self.num_inds = num_inds
        self.dim_out = dim_out

        self.mab0 = MAB(dim_out=dim_out, dim_hidden=dim_hidden, num_heads=num_heads)
        self.mab1 = MAB(dim_out=dim_out, dim_hidden=dim_hidden, num_heads=num_heads)

    def forward(self, context, generate_noise=None):
        if generate_noise is None:
            lead_shape = context.shape[:-2]
            generate_noise = torch.randn(*lead_shape, self.num_inds, self.dim_out, device=context.device)
        h = self.mab0(context, generate_noise)
        return self.mab1(generate_noise, h)


class PMA(nn.Module):
    def __init__(self, dim_out, dim_hidden=128, num_heads=8, num_comp=40, ln=True):
        super(PMA, self).__init__()
        self.s = nn.Parameter(torch.Tensor(1, num_comp, dim_out))
        nn.init.xavier_uniform_(self.s)
        self.mab = MAB(dim_out=dim_out, dim_hidden=dim_hidden, num_heads=num_heads, ln=ln)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            assert x.size(0) == 1, f"Expected x.shape[0]=1, but got {x.size(0)}"
        return self.mab(self.s, x)


if __name__ == '__main__':

    torch.manual_seed(0)

    X = torch.randn(40, 30)
    p_model = PMA(dim_out=30, dim_hidden=128, num_heads=8, num_seeds=10, ln=True)

    H1 = p_model(X)


