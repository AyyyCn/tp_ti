import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

torch.backends.cudnn.benchmark = True

# === Dataset paresseux ===
class IndianPines3DDataset(Dataset):
    def __init__(self, data, labels, patch_size=25):
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.margin = patch_size // 2
        self.indices = [
            (i, j)
            for i in range(self.margin, data.shape[0] - self.margin)
            for j in range(self.margin, data.shape[1] - self.margin)
            if labels[i, j] > 0
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        patch = self.data[i - self.margin:i + self.margin + 1,
                          j - self.margin:j + self.margin + 1, :]
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        label = self.labels[i, j] - 1
        return patch, label

# === Modèle CNN 3D simple ===
class CNN3D(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3), nn.BatchNorm3d(8), nn.ReLU(),
            nn.Conv3d(8, 16, 3), nn.BatchNorm3d(16), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(16, n_classes)
        )
    def forward(self, x):
        return self.model(x)

def main():
    # === Chargement données ===
    data = scipy.io.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = scipy.io.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
    h, w, bands = data.shape

    # === Normalisation ===
    X = data.reshape(-1, bands)
    X = StandardScaler().fit_transform(X).reshape(h, w, bands)

    # === Dataset et DataLoader ===
    dataset = IndianPines3DDataset(X, labels, patch_size=25)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # === Détection GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # === Pondération des classes
    all_labels = [label for _, label in dataset]
    class_counts = Counter(all_labels)
    weights = torch.tensor([1.0 / class_counts[i] for i in range(len(class_counts))], dtype=torch.float32).to(device)

    # === Modèle
    n_classes = len(np.unique(labels)) - 1
    model = CNN3D(in_ch=1, n_classes=n_classes).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()  # AMP compatible PyTorch <= 2.x

    # === Entraînement
    for epoch in range(25):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(batch_x)
                loss = loss_fn(output, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

        print(f"[{epoch+1}/25] Loss: {total_loss:.4f} - Acc: {correct/total:.4f}")

    # === Évaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())

    print("Classification Report:")
    print(classification_report(all_targets, all_preds, zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (3D CNN)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix_3dcnn.png")
    plt.show()

if __name__ == "__main__":
    main()
