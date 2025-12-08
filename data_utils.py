import numpy as np

def generate_federated_linear_regression(M, d, n_list, mu, Sigma, sigma2, mixing_prob=None, K=2, seed: int = 0):

    rng = np.random.default_rng(seed)

    if mixing_prob is None:
        mixing_prob = np.ones(K) / K
    else:
        mixing_prob = np.array(mixing_prob)
        mixing_prob /= mixing_prob.sum()

    w_components = rng.multivariate_normal(mean=np.zeros(d), cov=(1.0 / d) * np.eye(d), size=K)

    clients = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for m in range(M):

        n_m = n_list[m]

        X_m = rng.multivariate_normal(mean=mu, cov=Sigma, size=n_m)  # (n_m, d)
        X_m_test = rng.multivariate_normal(mean=mu, cov=Sigma, size=200)

        z_m = rng.choice(K, size=n_m, p=mixing_prob)
        z_m_test = rng.choice(K, size=200, p=mixing_prob)

        y_mean_m = np.zeros(n_m)
        y_mean_m_test = np.zeros(200)

        for k in range(K):
            idx_train = (z_m == k)
            idx_test = (z_m_test == k)
            if np.any(idx_train):
                y_mean_m[idx_train] = X_m[idx_train] @ w_components[k]
            if np.any(idx_train):
                y_mean_m_test[idx_test] = X_m_test[idx_test] @ w_components[k]

        eps_m = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=n_m)
        eps_m_test = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=200)

        y_m = y_mean_m + eps_m  # (n_m,)
        y_m_test = y_mean_m_test + eps_m_test

        clients.append({
            "id": m,
            "X": X_m,
            "y": y_m,
            "X_test": X_m_test,
            "y_test": y_m_test
        })

        X_train.append(X_m)
        y_train.append(y_m)
        X_test.append(X_m_test)
        y_test.append(y_m_test)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    X_isab = rng.multivariate_normal(mean=mu, cov=Sigma, size=400)
    z_isab = rng.choice(K, size=400, p=mixing_prob)
    y_mean_isab = np.zeros(400)
    for k in range(K):
        idx_k = (z_isab == k)
        if np.any(idx_k):
            y_mean_isab[idx_k] = X_isab[idx_k] @ w_components[k]

    eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=400)
    y_isab = y_mean_isab + eps

    return clients, w_components, X_train, y_train, X_test, y_test, X_isab, y_isab


if __name__ == "__main__":
    M = 10
    d = 29
    n_list = [40, 48, 30, 34, 50, 52, 28, 42, 64, 24]

    mu = np.zeros(d)
    Sigma = np.eye(d)
    sigma2 = 0.1

    clients, w_comps, X_train, y_train, X_test, y_test, X_isab, y_isab = (
        generate_federated_linear_regression(M=M, d=d, n_list=n_list, mu=mu, Sigma=Sigma, sigma2=sigma2, K=2, seed=45))

    print("True w* shape:", w_comps.shape)
    print("Client 0 X shape:", clients[0]["X"].shape, "y shape:", clients[0]["y"].shape)
    print("Centralized X shape:", X_train.shape, "y shape:", y_train.shape)

    np.savez("data/central_dataset.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
             w_star=w_comps, M=M, d=d, n_list=np.array(n_list), mu=mu, Sigma=Sigma, sigma2=np.array(sigma2))

    np.savez("data/isab_dataset.npz", X_isab=X_isab, y_isab=y_isab)

    for c in clients:
        m = c["id"]
        np.savez(f"data/client_{m}.npz", X_train=c["X"], y_train=c["y"], X_test=c['X_test'], y_test=c['y_test'],
                 id=np.array(m))

    print("Saved centralized_dataset.npz and client_*.npz")
