from modules import *

class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, h=64, rep_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, rep_dim)
        )

    def forward(self, x, y):
        """Encode training set as one vector representation"""
        if y.dim() == x.dim() - 1:
            y = y.unsqueeze(-1)
        z = self.encoder(torch.cat([x, y], dim=-1))
        return z.mean(dim=-2)

class Decoder(nn.Module):
    def __init__(self, dim_in, dim_out, h=64):
        super(Decoder, self).__init__()
        self.min_sigma = 0.1
        self.dim_out = dim_out
        self.decoder = nn.Sequential(
            nn.Linear(dim_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, dim_out * 2)
        )

    def forward(self, rep, x):
        # rep: [..., rep_dim]
        # x : (..., N, x_dim) or (N, x_dim)
        rep = rep.unsqueeze(-2).expand(*x.shape[:-1], rep.shape[-1]) # (..., N, rep_dim)
        dec_input = torch.cat((x, rep), dim=-1)

        out = self.decoder(dec_input)
        mu, log_sigma = torch.split(out, self.dim_out, dim=-1)

        sigma = self.min_sigma + (1 - self.min_sigma) * torch.nn.functional.softplus(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)

        return dist, mu, sigma

class MPNP(nn.Module):
    def __init__(self, x_dim, y_dim, h=128, rep_dim=128):
        super(MPNP, self).__init__()
        self._encoder = Encoder(x_dim, y_dim, h=h, rep_dim=rep_dim)
        self._decoder = Decoder(dim_in=rep_dim + x_dim, dim_out=y_dim, h=h)

    def forward(self, query, target_y=None):
        (con_x, con_y), val_x = query
        rep = self._encoder(con_x, con_y)
        dist, mu, sigma = self._decoder(rep, val_x)

        if target_y is not None and target_y.dim() == mu.dim() - 1:
            target_y = target_y.unsqueeze(-1)

        log_p = None if target_y is None else dist.log_prob(target_y)

        return log_p, mu, sigma
