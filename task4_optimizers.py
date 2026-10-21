import copy, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42); random.seed(42); torch.manual_seed(42)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
Xtr, Xv = scaler.fit_transform(Xtr), scaler.transform(Xv)
Xtr, Xv = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xv, dtype=torch.float32)
ytr, yv = torch.tensor(ytr, dtype=torch.long), torch.tensor(yv, dtype=torch.long)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()

class MoonNet(nn.Module):
    def __init__(self, regularized=False):
        super().__init__()
        layers = [nn.Linear(2, 32), nn.ReLU()]
        if regularized: layers += [nn.BatchNorm1d(32), nn.Dropout(0.3)]
        layers += [nn.Linear(32, 16), nn.ReLU()]
        if regularized: layers += [nn.BatchNorm1d(16), nn.Dropout(0.3)]
        layers += [nn.Linear(16, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def evaluate(model):
    model.eval()
    with torch.no_grad(): return criterion(model(Xv), yv).item()

def train(name, regularized=False, early_stop=False, step_lr=False):
    model = MoonNet(regularized)
    if name == "SGD": opt = torch.optim.SGD(model.parameters(), lr=0.05)
    elif name == "SGD Momentum": opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    elif name == "Adam": opt = torch.optim.Adam(model.parameters(), lr=0.01)
    elif name == "AdamW": opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5) if step_lr else None
    train_losses, val_losses, rates = [], [], []
    best, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(100):
        model.train(); batch_losses = []
        for xb, yb in loader:
            opt.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); opt.step(); batch_losses.append(loss.item())
        tr_loss, val_loss = float(np.mean(batch_losses)), evaluate(model)
        train_losses.append(tr_loss); val_losses.append(val_loss)
        rates.append(opt.param_groups[0]["lr"])
        if val_loss < best - 1e-5:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else: wait += 1
        if scheduler: scheduler.step()
        if early_stop and wait >= 10:
            print(f"Early stopping at epoch {epoch+1}")
            break
    model.load_state_dict(best_state)
    return {"train": train_losses, "val": val_losses, "rates": rates, "best": best}

results = {}
for name in ["SGD", "SGD Momentum", "Adam"]:
    torch.manual_seed(42); results[name] = train(name)
    print(name, "best validation loss:", round(results[name]["best"], 4))
plt.figure()
for name, r in results.items(): plt.plot(r["train"], label=name)
plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Optimizer Comparison")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("task4_optimizer_loss_curves.png"); plt.show()

torch.manual_seed(42); baseline = train("Adam")
torch.manual_seed(42); reg = train("Adam", regularized=True, early_stop=True, step_lr=True)
print("Adam baseline validation loss:", round(baseline["best"], 4))
print("Dropout + BatchNorm validation loss:", round(reg["best"], 4))
plt.figure(); plt.plot(baseline["val"], label="Adam baseline")
plt.plot(reg["val"], label="Adam + Dropout + BatchNorm + StepLR")
plt.xlabel("Epoch"); plt.ylabel("Validation loss"); plt.title("Validation Loss Comparison")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("task4_validation_loss_comparison.png"); plt.show()
plt.figure(); plt.plot(reg["rates"])
plt.xlabel("Epoch"); plt.ylabel("Learning rate"); plt.title("StepLR Schedule")
plt.grid(True); plt.tight_layout(); plt.savefig("task4_step_lr_schedule.png"); plt.show()
