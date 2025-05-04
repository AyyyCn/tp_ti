import numpy as np
import scipy.io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from collections import Counter

# === Load and normalize the data ===
data = scipy.io.loadmat('PaviaU.mat')['paviaU']

labels = scipy.io.loadmat('PaviaU_gt.mat')['paviaU_gt']
h, w, bands = data.shape
X = data.reshape(-1, bands)
X = StandardScaler().fit_transform(X).reshape(h, w, bands)

# === PCA reduction ===
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X.reshape(-1, bands)).reshape(h, w, 30)

# === Patch extraction ===
def extract_patches(data, labels, patch_size=25):
    margin = patch_size // 2
    data_patches = []
    patch_labels = []
    for i in range(margin, h - margin):
        for j in range(margin, w - margin):
            label = labels[i, j]
            if label == 0:
                continue
            patch = data[i-margin:i+margin+1, j-margin:j+margin+1, :]
            data_patches.append(patch)
            patch_labels.append(label - 1)
    return np.array(data_patches), np.array(patch_labels)

X_patches, y_patches = extract_patches(X_pca, labels)
print("Patches:", X_patches.shape)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X_patches, y_patches, test_size=0.3, stratify=y_patches, random_state=42)
X_train = torch.tensor(X_train).permute(0, 3, 1, 2).float()
X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

# === DataLoader ===
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# === Class weights ===
class_counts = Counter(y_train.numpy())
weights = torch.tensor([1.0 / class_counts[i] for i in range(len(class_counts))])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = weights.to(device)

# === CNN Model ===
class BetterCSCNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.model(x)

n_classes = len(np.unique(y_patches))
model = BetterCSCNN(X_train.shape[1], n_classes).to(device)

# === Training ===
loss_fn = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(25):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Train Acc: {correct / total:.4f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(1).cpu()
    print("Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

# === Generate full prediction map ===
margin = 12  # 25x25 patch
pred_map = np.zeros((h, w), dtype=np.uint8)

model.eval()
with torch.no_grad():
    for i in range(margin, h - margin):
        for j in range(margin, w - margin):
            if labels[i, j] == 0:
                continue
            patch = X_pca[i - margin:i + margin + 1, j - margin:j + margin + 1, :]
            patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(patch_tensor).argmax(1).item()
            pred_map[i, j] = pred + 1
# === Adjust labels to match prediction size ===
gt_cropped = labels[margin:h - margin, margin:w - margin]
pred_cropped = pred_map[margin:h - margin, margin:w - margin]

# === Visualization ===
colors = np.random.rand(n_classes + 1, 3)
colors[0] = [0, 0, 0]  # black for background
cmap = ListedColormap(colors)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].imshow(gt_cropped, cmap=cmap)
axs[0].set_title("Ground Truth (Cropped)")
axs[0].axis('off')

axs[1].imshow(pred_cropped, cmap=cmap)
axs[1].set_title("Predicted Classes (CNN2D)")
axs[1].axis('off')

plt.tight_layout()
plt.savefig("pavia_comparison_cropped.png")
plt.show()
