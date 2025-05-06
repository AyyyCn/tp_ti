import numpy as np, scipy.io, torch, torch.nn as nn, torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

# -------------------- data --------------------
PATCH = 15                       # smaller patch → fewer parameters
COMP  = 20                       # 20 PCs are enough on Pavia U
DATA  = scipy.io.loadmat('PaviaU.mat')['paviaU']
GT    = scipy.io.loadmat('PaviaU_gt.mat')['paviaU_gt']
h, w, b = DATA.shape

X = StandardScaler().fit_transform(DATA.reshape(-1, b)).reshape(h, w, b)
X  = PCA(COMP).fit_transform(X.reshape(-1, b)).reshape(h, w, COMP)

def make_patches(img, lab, skip0=True, k=PATCH):
    m = k // 2
    xs, ys = [], []
    for i in range(m, h-m):
        for j in range(m, w-m):
            if skip0 and lab[i, j] == 0: continue
            xs.append(img[i-m:i+m+1, j-m:j+m+1])
            ys.append(lab[i, j]-1)       # shift to 0‑based
    xs = np.stack(xs).astype('float32')
    xs = np.transpose(xs, (0, 3, 1, 2))  # NCHW
    return xs, np.array(ys, dtype='int64')

X_p, y_p = make_patches(X, GT)
n_cls     = GT.max()                     # 9 for PaviaU

# -------------------- model --------------------
class TinyCNN(nn.Module):
    def __init__(self, ch_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                    # 15→7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                    # 7→3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, n_cls)
        )
    def forward(self, x): return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = TinyCNN(COMP, n_cls).to(device)

# -------------------- loaders & weights --------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_p, y_p, test_size=0.15, stratify=y_p, random_state=0)

tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

tr_loader  = DataLoader(tr_ds, 64, shuffle=True,  drop_last=True)
val_loader = DataLoader(val_ds, 256, shuffle=False)

counts  = np.bincount(y_tr, minlength=n_cls)
weights = torch.tensor(1. / np.maximum(counts, 1), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=weights)
opt       = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.4)

# -------------------- training loop --------------------
best, patience, PATIENCE = float('inf'), 0, 8
for epoch in range(20):
    # train
    model.train(); losses = []
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); opt.step()
        losses.append(loss.item())
    # val
    model.eval(); vloss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out    = model(xb); vloss += criterion(out, yb).item()
            correct += (out.argmax(1)==yb).sum().item()
            total   += yb.size(0)
    vloss /= len(val_loader); acc = correct/total
    scheduler.step(vloss)
    print(f'E{epoch:02d}  loss={np.mean(losses):.4f}  val={vloss:.4f}  acc={acc:.3f}')

    if vloss < best:
        best, patience = vloss, 0
        torch.save(model.state_dict(), 'tinycnn_paviau.pt')
    else:
        patience += 1
        if patience > PATIENCE: break      # early stopping
