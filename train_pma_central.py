import torch
import numpy as np
import torch.optim as optim
from modules import ISAB, PMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_seeds = 20
num_induces = 120

data = np.load("./data/central_dataset.npz")
context_x = torch.from_numpy(data["X_train"]).float().to(device)
context_y = torch.from_numpy(data["y_train"]).float().unsqueeze(-1).to(device)
context = torch.cat((context_x, context_y), dim=1)
d = context.size(1)

isab = ISAB(d, d, 5, num_induces, ln=True).to(device)
isab_dict = torch.load("checkpoint/isab_model.pt")
isab.load_state_dict(isab_dict)
for p in isab.parameters():
    p.requires_grad = False
isab.eval()

pma = PMA(d, 5, num_seeds, ln=True).to(device)
pma.train()

epochs = int(1e4)
optimizer = optim.Adam(pma.parameters(), lr=5e-3)

for i in range(epochs):
    optimizer.zero_grad()

    e = torch.randn(1, num_induces, d, device=device)

    induce_points = pma(context)
    pred_induce_points = isab(induce_points, e)
    X_induce = induce_points[:, :d - 1]
    y_induce = induce_points[:, d - 1:]
    X_pred_induce = pred_induce_points[:, :d - 1]
    y_pred_induce = pred_induce_points[:, d - 1:]

    X_induce_aug = torch.cat((X_induce, X_pred_induce), dim=0)
    y_induce_aug = torch.cat((y_induce, y_pred_induce), dim=0)

    XTX_induce = X_induce_aug.T@X_induce_aug
    XTy_induce = X_induce_aug.T@y_induce_aug
    W_induce = torch.linalg.solve(XTX_induce, XTy_induce)

    pred_real_points = isab(context, e)
    X_pred_real = pred_real_points[:, :d - 1]
    y_pred_real = pred_real_points[:, d - 1:]
    X_real_aug = torch.cat((context_x, X_pred_real), dim=0)
    y_real_aug = torch.cat((context_y, y_pred_real), dim=0)

    XTX_real = X_real_aug.T@X_real_aug
    XTy_real = X_real_aug.T@y_real_aug
    W_real = torch.linalg.solve(XTX_real, XTy_real)

    loss = torch.norm(W_induce - W_real)

    loss.backward()
    optimizer.step()

    if (i + 1) % 500 == 0:
        print(f'Iteration {i + 1}: Loss: {loss.item():.4f}')
        save_path = "checkpoint/pma_model_central_10.pt"
        torch.save(pma.state_dict(), save_path)
