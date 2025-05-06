import numpy as np, scipy.io, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

PATCH, COMP = 15, 20
DATA  = scipy.io.loadmat('PaviaU.mat')['paviaU']
GT    = scipy.io.loadmat('PaviaU_gt.mat')['paviaU_gt']
h, w, b = DATA.shape

X = StandardScaler().fit_transform(DATA.reshape(-1, b)).reshape(h, w, b)
X  = PCA(COMP).fit_transform(X.reshape(-1, b)).reshape(h, w, COMP)

def pad(img, m): return np.pad(img, ((m,m),(m,m),(0,0)), mode='reflect')
m  = PATCH//2
Xp = pad(X, m)

class TinyCNN(nn.Module):
    # same as above …
    def __init__(self, ch_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, GT.max())
        )
    def forward(self, x): return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = TinyCNN(COMP, GT.max()).to(device)
model.load_state_dict(torch.load('tinycnn_paviau.pt', map_location=device))
model.eval()

out_map = np.zeros((h, w), dtype='uint8')
with torch.no_grad():
    for i in range(h):
        batch = []
        for j in range(w):
            patch = Xp[i:i+PATCH, j:j+PATCH]
            batch.append(patch.transpose(2,0,1))          # C,H,W
        batch_np = np.stack(batch, axis=0).astype(np.float32)       # -> (w, C, H, W)
        batch_t  = torch.from_numpy(batch_np).to(device)            # conversion une seule fois
        pred     = model(batch_t).argmax(1).cpu().numpy() + 1
        out_map[i] = pred

out_map[GT==0] = 0

# ---------- show ----------
colors = np.random.rand(GT.max()+1, 3); colors[0] = .5,.5,.5
cmap = ListedColormap(colors)
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].imshow(GT, cmap=cmap);  ax[0].set_title('Ground truth');   ax[0].axis('off')
ax[1].imshow(out_map, cmap=cmap); ax[1].set_title('Prediction'); ax[1].axis('off')
plt.tight_layout(); plt.show()
plt.savefig("paviau_full_prediction.png")