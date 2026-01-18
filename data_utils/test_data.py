import os
import math
import numpy as np
from typing import Optional, Tuple, List


def _sample_x(
    rng: np.random.Generator,
    n: int,
    d: int,
    x_dist: str,
    mu: Optional[np.ndarray],
    Sigma: Optional[np.ndarray],
    x_range: Tuple[float, float],
) -> np.ndarray:
    if x_dist == "normal":
        assert mu is not None and Sigma is not None
        return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)  # (n,d)
    elif x_dist == "uniform":
        lo, hi = x_range
        return rng.uniform(low=lo, high=hi, size=(n, d))           # (n,d)
    else:
        raise ValueError(f"Unknown x_dist: {x_dist}")

def _sample_w_components(
    rng: np.random.Generator,
    K: int,
    d: int,
    w_dist: str,
    w_scale: float,
) -> np.ndarray:
    """W: (K,d)"""
    if w_dist == "normal":
        return rng.normal(loc=0.0, scale=w_scale, size=(K, d))
    elif w_dist == "uniform":
        return rng.uniform(low=-w_scale, high=w_scale, size=(K, d))
    else:
        raise ValueError(f"Unknown w_dist: {w_dist}")


def _sample_noise(
    rng: np.random.Generator,
    n: int,
    noise_dist: str,
    noise_var: float,
) -> np.ndarray:
    """eps: (n,1) with Var(eps)=noise_var"""
    if noise_var < 0:
        raise ValueError("noise_var must be non-negative.")
    if noise_var == 0.0:
        return np.zeros((n, 1), dtype=float)

    if noise_dist == "normal":
        sigma = math.sqrt(noise_var)
        return rng.normal(loc=0.0, scale=sigma, size=(n, 1))
    elif noise_dist == "uniform":
        a = math.sqrt(3.0 * noise_var)
        return rng.uniform(low=-a, high=a, size=(n, 1))
    else:
        raise ValueError(f"Unknown noise_dist: {noise_dist}")


def _gen_y(
    rng: np.random.Generator,
    X: np.ndarray,           # (n,d)
    W: np.ndarray,           # (K,d)
    eps: np.ndarray,         # (n,1)
    mixing_prob: Optional[np.ndarray] = None,  # (K,)
):
    n, d = X.shape
    K = W.shape[0]

    if K == 1:
        return (X @ W[0])[:, None] + eps

    if mixing_prob is None:
        mixing_prob = np.ones(K) / K
    else:
        mixing_prob = np.array(mixing_prob, dtype=float)
        mixing_prob /= mixing_prob.sum()

    z = rng.choice(K, size=n, p=mixing_prob)
    w_per_sample = W[z]
    return np.sum(X * w_per_sample, axis=1)[:, None] + eps  # (n,1)

def generate_federated_linear_regression_iid_train_test(
    M: int,
    d: int,
    n_train_list: List[int],
    n_test: int,
    # x
    x_dist: str,
    mu: Optional[np.ndarray],
    Sigma: Optional[np.ndarray],
    x_range: Tuple[float, float],
    # w
    w_dist: str,
    w_scale: float,
    K: int,
    mixing_prob: Optional[np.ndarray],
    # noise
    noise_dist: str,
    noise_var: float,
    # seed
    seed: int,
):
    """
    IID across clients:
      - K == 1: all clients share one w_star (d,)
      - K > 1: all clients share W_star (K,d) and mixing_prob; each SAMPLE draws component independently
    """
    rng = np.random.default_rng(seed)
    assert len(n_train_list) == M
    assert K >= 1

    W_star = _sample_w_components(rng, K=K, d=d, w_dist=w_dist, w_scale=w_scale)  # (K,d)

    clients = []
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for m in range(M):
        n_train = int(n_train_list[m])

        X_tr = _sample_x(rng, n_train, d, x_dist, mu, Sigma, x_range)
        eps_tr = _sample_noise(rng, n_train, noise_dist, noise_var)
        y_tr = _gen_y(rng, X_tr, W_star, eps_tr, mixing_prob)

        X_te = _sample_x(rng, n_test, d, x_dist, mu, Sigma, x_range)
        eps_te = _sample_noise(rng, n_test, noise_dist, noise_var)
        y_te = _gen_y(rng, X_te, W_star, eps_te, mixing_prob)

        c = {"id": m, "X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}
        clients.append(c)

        X_train_all.append(X_tr)
        y_train_all.append(y_tr)
        X_test_all.append(X_te)
        y_test_all.append(y_te)

    X_train_all = np.vstack(X_train_all)
    y_train_all = np.vstack(y_train_all)
    X_test_all = np.vstack(X_test_all)
    y_test_all = np.vstack(y_test_all)

    return clients, X_train_all, y_train_all, X_test_all, y_test_all, W_star


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # ---------- CONFIG ----------
    M = 5
    d = 19
    n_train_list = [27, 19, 22, 20, 24]
    n_test = 1000

    x_dist = "normal"          # "normal" or "uniform"
    mu = np.zeros(d)
    Sigma = np.eye(d)
    x_range = (-2.0, 2.0)

    w_dist = "normal"          # "normal" or "uniform"
    w_scale = 1.0

    #   K=1  -> shared single w
    #   K>1  -> per-sample mixture over K shared w-components
    K = 1
    mixing_prob = np.array([0.2, 0.5, 0.3]) if K > 1 else None

    noise_dist = "uniform"     # "uniform" or "normal"
    noise_var = 0.1
    seed = 45
    # ----------------------------

    clients, X_train, y_train, X_test, y_test, W_star = generate_federated_linear_regression_iid_train_test(
        M=M, d=d, n_train_list=n_train_list, n_test=n_test,
        x_dist=x_dist, mu=mu, Sigma=Sigma, x_range=x_range,
        w_dist=w_dist, w_scale=w_scale,
        K=K, mixing_prob=mixing_prob,
        noise_dist=noise_dist, noise_var=noise_var,
        seed=seed,
    )

    print("Centralized X_train:", X_train.shape, "y_train:", y_train.shape)
    print("Centralized X_test :", X_test.shape,  "y_test :", y_test.shape)

    # Save centralized dataset
    central_path = "../data/central_dataset.npz"
    payload = {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "K": np.array(K),
        "M": np.array(M), "d": np.array(d),
        "n_train_list": np.array(n_train_list),
        "n_test": np.array(n_test),
        "x_dist": np.array(x_dist),
        "w_dist": np.array(w_dist),
        "noise_dist": np.array(noise_dist),
        "noise_var": np.array(noise_var),
        "seed": np.array(seed),
        "W_star": np.array(W_star),
    }

    np.savez(central_path, **payload)
    print("Saved:", central_path)

    # Save per-client datasets
    for c in clients:
        m = c["id"]
        client_path = f"../data/client_{m}.npz"
        p = {
            "id": np.array(m),
            "X_train": c["X_train"], "y_train": c["y_train"],
            "X_test":  c["X_test"],  "y_test":  c["y_test"],
            "K": np.array(K),
            "seed": np.array(seed),
        }
        np.savez(client_path, **p)

    print(f"Saved {M} client files: data/client_*.npz")
