
import numpy as np, scipy.io, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# ------------------------------ paramètres ------------------------------------
PATCH, N_PCA = 11, 30
MODEL_FILE   = 'tinycnn_indianpines.pt'

# ------------------------------ données ---------------------------------------
data = scipy.io.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
gt   = scipy.io.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
h, w, b = data.shape
n_cls   = gt.max()

# Standardisation + PCA
X = StandardScaler().fit_transform(data.reshape(-1, b)).reshape(h, w, b)
X = PCA(N_PCA).fit_transform(X.reshape(-1, b)).reshape(h, w, N_PCA)

# Padding
m   = PATCH // 2
Xp  = np.pad(X, ((m, m), (m, m), (0, 0)), mode='reflect')  # (h+2m, w+2m, C)

# ------------------------------ modèle ----------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, ch_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, n_cls)
        )
    def forward(self, x):
        return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = TinyCNN(N_PCA, n_cls).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

# ------------------------------ extraction patches GPU (unfold) ---------------
# Xp -> (1, C, H+2m, W+2m)
X_tensor = torch.from_numpy(Xp.transpose(2,0,1)).unsqueeze(0)  # float64 by default
X_tensor = X_tensor.to(device, dtype=torch.float32)

# Unfold: (1, C*P*P, h*w)
patches = torch.nn.functional.unfold(X_tensor, PATCH)
patches = patches.squeeze(0).T                                 # (h*w, C*P*P)
patches = patches.view(-1, N_PCA, PATCH, PATCH)               # (N, C, P, P)

BATCH = 2048
pred_map = torch.zeros(h*w, dtype=torch.uint8, device='cpu')

with torch.no_grad():
    for start in range(0, h*w, BATCH):
        out = model(patches[start:start+BATCH]).argmax(1).cpu() + 1
        pred_map[start:start+len(out)] = out

pred_map = pred_map.view(h, w).numpy()
pred_map[gt == 0] = 0   # ré‑insère la classe fond

# ------------------------------ métriques -------------------------------------
mask = gt.ravel() > 0
print(classification_report(gt.ravel()[mask]-1, pred_map.ravel()[mask]-1,
                            zero_division=0, digits=4))

# ------------------------------ visuel ----------------------------------------
colors = np.random.rand(n_cls+1, 3); colors[0] = (.5,.5,.5)
cmap = ListedColormap(colors)
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].imshow(gt, cmap=cmap);        ax[0].set_title('Ground truth');   ax[0].axis('off')
ax[1].imshow(pred_map, cmap=cmap);  ax[1].set_title('Prediction');     ax[1].axis('off')
plt.tight_layout(); plt.show()
plt.savefig('indian_pines.png', dpi=300, bbox_inches='tight')