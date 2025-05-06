
import numpy as np, scipy.io, torch, torch.nn as nn, torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ------------ hyper‑paramètres ------------
PATCH      = 11          # 11×11 suffit sur Indian Pines
N_PCA      = 30          # 30 composantes ≈ >99 % variance
BATCH_TR   = 64
BATCH_VAL  = 256
MAX_EPOCHS = 15
PATIENCE   = 10          # early‑stopping
LR         = 1e-3
WD         = 1e-4
MODEL_FILE = 'tinycnn_indianpines.pt'

# ------------ données ------------
data   = scipy.io.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
gt     = scipy.io.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
h, w, b = data.shape
nb_cls  = gt.max()                 # 16 vraies classes (0 = fond)

# (1) standardisation + PCA
X = StandardScaler().fit_transform(data.reshape(-1, b)).reshape(h, w, b)
X = PCA(N_PCA).fit_transform(X.reshape(-1, b)).reshape(h, w, N_PCA)

# (2) génération des patches
m = PATCH // 2
def make_patches(img, lab):
    xs, ys = [], []
    for i in range(m, h-m):
        for j in range(m, w-m):
            lbl = lab[i, j]
            if lbl == 0:     # on saute la classe fond
                continue
            patch = img[i-m:i+m+1, j-m:j+m+1]      # (P,P,C)
            xs.append(patch)
            ys.append(lbl-1)                      # passe en 0‑based
    xs = np.stack(xs).astype('float32').transpose(0,3,1,2)  # N,C,P,P
    ys = np.array(ys, dtype='int64')
    return xs, ys

X_p, y_p = make_patches(X, gt)

# (3) split train/val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_p, y_p, test_size=0.15, stratify=y_p, random_state=1)

tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
tr_ld  = DataLoader(tr_ds, BATCH_TR,  shuffle=True,  drop_last=True)
val_ld = DataLoader(val_ds, BATCH_VAL, shuffle=False)

# ------------ modèle ------------
class TinyCNN(nn.Module):
    def __init__(self, ch_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 11→5
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 5→2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, n_cls)
        )
    def forward(self, x): return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = TinyCNN(N_PCA, nb_cls).to(device)

# ------------ loss équilibré ------------
cnts      = np.bincount(y_tr, minlength=nb_cls)
weights   = torch.tensor(1/np.maximum(cnts,1), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=weights)
opt       = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched     = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.4)

# ------------ training ------------
best, wait = float('inf'), 0
for epoch in range(MAX_EPOCHS):
    # — train
    model.train(); tr_loss = []
    for xb, yb in tr_ld:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); opt.step()
        tr_loss.append(loss.item())
    # — validation
    model.eval(); v_loss = correct = total = 0
    with torch.no_grad():
        for xb, yb in val_ld:
            xb, yb = xb.to(device), yb.to(device)
            out    = model(xb)
            v_loss += criterion(out, yb).item()
            correct += (out.argmax(1)==yb).sum().item()
            total   += yb.size(0)
    v_loss /= len(val_ld); acc = correct/total
    sched.step(v_loss)
    print(f'E{epoch:02d}  tr={np.mean(tr_loss):.4f}  val={v_loss:.4f}  acc={acc:.3f}')
    # — early‑stopping
    if v_loss < best:
        best, wait = v_loss, 0
        torch.save(model.state_dict(), MODEL_FILE)
    else:
        wait += 1
        if wait >= PATIENCE:
            print('⏹ early stop'); break
