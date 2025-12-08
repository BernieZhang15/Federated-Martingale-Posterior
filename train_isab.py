import torch
import numpy as np
from models import MPNP
from modules import ISAB
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load("data/isab_dataset.npz")
X_isab = data["X_isab"]
y_isab = data["y_isab"]

X_tensor = torch.from_numpy(X_isab).float()
y_tensor = torch.from_numpy(y_isab).unsqueeze(1).float()

full_dataset = TensorDataset(X_tensor, y_tensor)

val_ratio = 0.7
n_total = len(full_dataset)
n_val = int(val_ratio * n_total)
n_train = int(n_total - n_val)

train_dataset, val_dataset = random_split(full_dataset, lengths=[n_train, n_val], generator=torch.Generator().manual_seed(42))

print(f"Total: {n_total}, Train: {n_train}, Val: {n_val}")
train_loader = DataLoader(train_dataset, batch_size=120, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)

context_x, context_y = next(iter(train_loader))
context_x, context_y = context_x.to(device), context_y.to(device)
context = torch.cat((context_x, context_y), dim=1)
val_iter = iter(val_loader)

isab_dim = context.size(1)
x_dim = context_x.size(1)
y_dim = context_y.size(1)

K = 4
epochs = int(5e3)
model = MPNP(x_dim=x_dim, y_dim=y_dim).to(device)
isab = ISAB(isab_dim, isab_dim, 5, 120, ln=True).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(isab.parameters()), lr=5e-3)

for i in range(epochs):
    model.train()
    isab.train()
    try:
        target_x, target_y = next(val_iter)
    except StopIteration:
        val_iter = iter(val_loader)
        target_x, target_y = next(val_iter)
    target_x, target_y = target_x.to(device), target_y.to(device)

    optimizer.zero_grad()

    query_real = (context_x, context_y), target_x
    log_prob_real, _, _ = model(query_real, target_y)
    L_amort = -log_prob_real.sum()

    losses_aug = []
    losses_pseudo = []

    for k in range(K):
        predictive = isab(context)

        predictive_x = predictive[:, :x_dim]
        predictive_y = predictive[:, x_dim:]

        aug_x = torch.cat((context_x, predictive_x), dim=0)
        aug_y = torch.cat((context_y, predictive_y), dim=0)

        query = (aug_x, aug_y), target_x
        query_pseudo = (predictive_x, predictive_y), target_x

        log_prob, _, _ = model(query, target_y)
        log_prob_pseudo, _, _ = model(query_pseudo, target_y)

        losses_aug.append(-log_prob.sum())
        losses_pseudo.append(-log_prob_pseudo.sum())

    L_marg = - torch.logsumexp(torch.stack([-L for L in losses_aug]), dim=0) - torch.log(torch.tensor(K, device=device))
    L_pseudo = sum(losses_pseudo) / K

    loss = L_marg + L_pseudo + L_amort

    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        print(f'Iteration {i + 1}: Loss: {loss.item():.4f}')
        save_path = "checkpoint/isab_model.pt"
        torch.save(isab.state_dict(), save_path)


isab.eval()

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

with torch.no_grad():
    for k in range(4):
        predictive = isab(context)

        X_vis = torch.cat([context, predictive], dim=0)
        labels = np.concatenate([np.zeros(context.size(0)), np.ones(predictive.size(0))])

        X_vis = X_vis.cpu().detach().numpy()

        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, init='pca', random_state=42)

        X_2d = tsne.fit_transform(X_vis)

        ax = axes[k]

        ax.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], alpha=0.7, label="real context", c='tab:blue')
        ax.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], alpha=0.7, label="predictive", c='tab:orange')

        ax.set_title(f'Predictive sample {k + 1}')

plt.tight_layout()
plt.legend()
plt.show()
