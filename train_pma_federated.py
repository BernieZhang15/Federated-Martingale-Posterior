import torch
import numpy as np
from modules import ISAB, PMA
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M = 10
num_seeds = 5
num_induces = 120

client_contexts = []

for m in range(M):
    data = np.load(f"data/client_{m}.npz")
    X_m = torch.from_numpy(data["X_train"]).float().to(device)
    y_m = torch.from_numpy(data["y_train"]).float().unsqueeze(-1).to(device)
    context_m = torch.cat((X_m, y_m), dim=1)
    client_contexts.append(context_m)

data_c = np.load("data/central_dataset.npz")
X_train = torch.from_numpy(data_c["X_train"]).float().to(device)
y_train = torch.from_numpy(data_c["y_train"]).float().unsqueeze(-1).to(device)
context = torch.cat((X_train, y_train), dim=1)
d_ctx = context.size(1)
d_x = X_train.size(1)

isab = ISAB(d_ctx, d_ctx, 5, num_induces, ln=True).to(device)
isab_dict = torch.load("checkpoint/isab_model.pt")
isab.load_state_dict(isab_dict)
for p in isab.parameters():
    p.requires_grad = False
isab.eval()

pma = PMA(d_ctx, 5, num_seeds, ln=True).to(device)
optimizer = optim.Adam(pma.parameters(), lr=5e-3)

epochs = int(1e4)

for i in range(epochs):

    pma.train()

    optimizer.zero_grad()

    e = torch.randn(1, num_induces, d_ctx, device=device)

    client_induces = []

    for m in range(M):
        context_m = client_contexts[m]
        induces_m = pma(context_m)
        client_induces.append(induces_m)

    induce_points = torch.cat(client_induces, dim=0)

    with torch.no_grad():
        pred_client_induces = isab(induce_points, e)

    X_induce = induce_points[:, :d_x]
    y_induce = induce_points[:, d_x:]
    X_pred_induce = pred_client_induces[:, :d_x]
    y_pred_induce = pred_client_induces[:, d_x:]

    X_induce_aug = torch.cat((X_induce, X_pred_induce), dim=0)
    y_induce_aug = torch.cat((y_induce, y_pred_induce), dim=0)

    XTX_induce = X_induce_aug.T@X_induce_aug
    XTy_induce = X_induce_aug.T@y_induce_aug
    W_induce = torch.linalg.solve(XTX_induce, XTy_induce)

    pred_real_points = isab(context, e)
    X_pred_real = pred_real_points[:, :d_x]
    y_pred_real = pred_real_points[:, d_x:]
    X_real_aug = torch.cat((X_train, X_pred_real), dim=0)
    y_real_aug = torch.cat((y_train, y_pred_real), dim=0)

    XTX_real = X_real_aug.T @ X_real_aug
    XTy_real = X_real_aug.T @ y_real_aug
    W_real = torch.linalg.solve(XTX_real, XTy_real)

    loss = torch.norm(W_induce - W_real)

    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        print(f'Iteration {i + 1}: Loss: {loss.item():.4f}')
        save_path = "checkpoint/pma_model_federated_5.pt"
        torch.save(pma.state_dict(), save_path)
