from modules import *

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x, y):
        """Encode training set as one vector representation"""
        set_input = torch.cat((x, y), dim=1)
        representation = torch.mean(set_input, dim=0)
        return representation

class Decoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(in_features=dim_in, out_features=dim_out)

    def forward(self, representation, x):
        set_size, dim = x.shape
        representation = representation.unsqueeze(0).repeat(set_size, 1)
        dec_input = torch.cat((x, representation), dim=1)

        out = self.linear(dec_input)
        mu, log_sigma = torch.split(out, 1, dim=-1)

        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)

        return dist, mu, sigma

class MPNP(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(MPNP, self).__init__()
        rep_dim = x_dim + y_dim
        self._encoder = Encoder()
        self._decoder = Decoder(dim_in=rep_dim + x_dim, dim_out=2)

    def forward(self, query, target_y=None):
        (con_x, con_y), val_x = query
        representation = self._encoder(con_x, con_y)
        dist, mu, sigma = self._decoder(representation, val_x)

        log_p = None if target_y is None else dist.log_prob(target_y)
        return log_p, mu, sigma
