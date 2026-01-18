import os
import math
import torch
import numpy as np
from models import MPNP
from modules import ISAB
import torch.optim as optim
from metric_utils import mmd_rbf2
from data_utils.train_data import build_infinite_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)

x_dim = 19
y_dim = 1

bs = 24
test_data = np.load("data/central_dataset.npz")
Xc = torch.from_numpy(test_data["X_test"]).float()
Yc = torch.from_numpy(test_data["y_test"]).float()
W_star = torch.from_numpy(test_data["W_star"]).float().to(device)

dataloader = build_infinite_dataloader(
        batch_size=64,
        x_dim=19,
        Pc=24,
        Pt=24,
        x_dist="normal",
        x_range=(-2, 2),
        w_dist="normal",
        w_scale=1.0,
        noise_dist="uniform",
        noise_scale=0.1,
        seed=42,
        num_workers=0,
    )
it = iter(dataloader)

num_samples = 8
num_steps = int(1e6)
model = MPNP(x_dim=x_dim, y_dim=y_dim, h=128, rep_dim=128).to(device)
isab = ISAB(dim_out=x_dim + y_dim, dim_hidden=128, num_heads=8, num_inds=24).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(isab.parameters()), lr=5e-3)

@torch.no_grad()
def sample_central(bs: int):
    n = Xc.shape[0]
    idx = torch.randint(0, n, (bs,))
    Xb = Xc[idx].to(device)
    yb = Yc[idx].to(device)
    return Xb, yb


for i in range(num_steps):
    model.train()
    isab.train()

    batch = next(it)
    x_ctx, y_ctx = batch["x_ctx"].to(device), batch["y_ctx"].to(device)
    x_tar, y_tar = batch["x_tar"].to(device), batch["y_tar"].to(device)

    target_x = torch.cat([x_ctx, x_tar], dim=1)
    target_y = torch.cat([y_ctx, y_tar], dim=1)
    N_all = target_x.shape[1]

    optimizer.zero_grad()

    # L_amort
    log_prob_real, _, _ = model(((x_ctx, y_ctx), target_x), target_y) # (B, N_all)
    lp_real = log_prob_real.sum(dim=tuple(range(1, log_prob_real.ndim)))
    L_amort = - (lp_real / N_all).mean()

    context = torch.cat((x_ctx, y_ctx), dim=-1) # [B, N_ctx, D]
    contextK = context.unsqueeze(1).expand(-1, num_samples, -1, -1) # [B, K, N_ctx, D]
    context_xK = contextK[..., :x_dim]
    context_yK = contextK[..., x_dim:]

    predK = isab(contextK) # [B, K, N_pred, D]
    pred_xK = predK[..., :x_dim] # [B, K, N_pred, x_dim]
    pred_yK = predK[..., x_dim:] # [B, K, N_pred, y_dim]

    aug_xK = torch.cat((context_xK, pred_xK), dim=2)  # (B, K, N_ctx + N_pred, x_dim)
    aug_yK = torch.cat((context_yK, pred_yK), dim=2)

    target_xK = target_x.unsqueeze(1).expand(-1, num_samples, -1, -1)  # (B, K, N_tar, x_dim)
    target_yK = target_y.unsqueeze(1).expand(-1, num_samples, -1, -1)

    # L_marg
    log_probK, _, _ = model(((aug_xK, aug_yK), target_xK), target_yK) # (B, K, N_all)
    lpK = log_probK.sum(dim=tuple(range(2, log_probK.ndim))) # (B, K)
    logmeanexp_K = torch.logsumexp(lpK, dim=1) - math.log(num_samples) # [B]
    L_marg = - (logmeanexp_K / N_all).mean() # scalar

    # L_pseudo
    log_prob_pseudoK, _, _ = model(((pred_xK, pred_yK), target_xK), target_yK) # (B, K, N_all)
    lpPseudo = log_prob_pseudoK.sum(dim=tuple(range(2, log_prob_pseudoK.ndim))) # [B, K]
    L_pseudo = - (lpPseudo.mean(dim=1) / N_all).mean() # scalar (mean over K, then over B)

    loss = L_marg + L_amort + L_pseudo
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        save_path = "checkpoints/isab_model.pt"
        torch.save(isab.state_dict(), save_path)

        isab.eval()
        with (torch.no_grad()):
            Xb, yb = sample_central(bs)
            context_real = torch.cat([Xb, yb], dim=-1).unsqueeze(0)

            w = W_star[0].view(x_dim, 1)

            pred_xy = isab(context_real).squeeze(0)
            pred_x = pred_xy[..., :x_dim]
            pred_y = pred_xy[..., x_dim:]

            res_real = yb - Xb @ w
            res_pred = pred_y - pred_x @ w

            mmd_res = mmd_rbf2(res_real, res_pred)
            mmd_x = mmd_rbf2(Xb, pred_x)

        print(
            f"[step {i}] "
            f"L_amort={float(L_amort):.6g}  L_marg={float(L_marg):.6g}  L_pseudo={float(L_pseudo):.6g}  loss={float(loss):.6g}  "
            f"MMD^2(residual)={mmd_res:.4g} MMD^2(x)={mmd_x:.4g}"
        )

