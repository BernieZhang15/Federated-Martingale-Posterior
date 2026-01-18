import torch
from typing import Optional, Union, Sequence

def _flatten_samples(z: torch.Tensor) -> torch.Tensor:
    """
    Flatten samples to shape [N, D].
    Accepts:
      [N, D]
      [B, N, D]
      [B, K, N, D]
    """
    if z.dim() == 2:
        return z
    elif z.dim() == 3:
        B, N, D = z.shape
        return z.reshape(B * N, D)
    elif z.dim() == 4:
        B, K, N, D = z.shape
        return z.reshape(B * K * N, D)
    else:
        raise ValueError(f"Unsupported tensor dim: {z.dim()}")

@torch.no_grad()
def _median_heuristic_sigma(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    z = torch.cat([x, y], dim=0)

    # Use float64 for numerical stability in distance computation
    z64 = z.double()
    d2 = torch.cdist(z64, z64, p=2.0) ** 2

    d2 = d2.flatten()
    d2 = d2[d2 > 0]
    if d2.numel() == 0:
        return 1.0

    med = d2.median().item()
    sigma = (max(med, eps) ** 0.5)
    return float(max(sigma, eps))

def _heuristic_multiscale_sigmas(
    x: torch.Tensor,
    y: torch.Tensor,
    scales: Sequence[int] = (-2, -1, 0, 1, 2),
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Auto-generate multi-scale sigmas around median heuristic base:
      sigmas = base * 2^k, k in scales
    """
    base = _median_heuristic_sigma(x, y, eps=eps)
    sigmas = torch.tensor([base * (2.0 ** k) for k in scales], device=x.device, dtype=x.dtype)
    return sigmas.clamp_min(eps)

def _rbf_kernel_mean(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Multi-scale RBF kernel aggregation over sigmas.

    Computes:
      K(x,y) = mean_i exp(-||x-y||^2 / (2*sigma_i^2))   if mode="mean"

    Returns matrix [Nx, Ny].
    """
    # squared distances [Nx, Ny]
    d2 = torch.cdist(x, y, p=2.0) ** 2
    d2 = d2.clamp_min(0.0)  # numerical safety

    sigmas = sigmas.to(device=x.device, dtype=x.dtype).clamp_min(eps).view(-1, 1, 1) # [S, 1, 1]
    denom = 2.0 * (sigmas ** 2)  # [S, 1, 1]

    k = torch.exp(-d2.unsqueeze(0) / denom)  # [S, Nx, Ny]
    return k.mean(dim=0)

@torch.no_grad()
def mmd_rbf2(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Optional[Union[Sequence[float], torch.Tensor]] = None,
    use_unbiased: bool = True,
    scales: Sequence[int] = (-2, -1, 0, 1, 2),
    eps: float = 1e-12,
) -> float:
    """
    Compute MMD^2 between samples x and y with multi-scale RBF kernel.
    x, y can be [N,D] or [B,N,D] or [B,K,N,D].

    - sigmas:
        None or "auto" -> median heuristic + multi-scale (base*2^k)
        list/tuple/tensor -> use provided sigmas
    """
    x = _flatten_samples(x).float()
    y = _flatten_samples(y).float()


    if sigmas is None:
        sigmas_t = _heuristic_multiscale_sigmas(x, y, scales=scales, eps=1e-6)
    else:
        if torch.is_tensor(sigmas):
            sigmas_t = sigmas.to(device=x.device, dtype=x.dtype)
        else:
            sigmas_t = torch.tensor(list(sigmas), device=x.device, dtype=x.dtype)
        sigmas_t = sigmas_t.clamp_min(1e-6)

    Kxx = _rbf_kernel_mean(x, x, sigmas_t, eps=eps)
    Kyy = _rbf_kernel_mean(y, y, sigmas_t, eps=eps)
    Kxy = _rbf_kernel_mean(x, y, sigmas_t, eps=eps)

    nx = x.shape[0]
    ny = y.shape[0]

    if use_unbiased:
        # exclude diagonal for unbiased estimate
        mmd2 = (Kxx.sum() - Kxx.diag().sum()) / (nx * (nx - 1) + eps) \
             + (Kyy.sum() - Kyy.diag().sum()) / (ny * (ny - 1) + eps) \
             - 2.0 * Kxy.mean()
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    mmd2 = mmd2.clamp_min(0.0)
    return float(mmd2.item())
