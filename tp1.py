# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
# import math

# # === Load the data ===
# data = loadmat('Indian_pines.mat')  # or 'Indian_pines.mat'
# cube = data['indian_pines']         # or 'indian_pines'
# H, W, B = cube.shape

# # === Normalize each band (optional for better visualization) ===
# cube = cube.astype(np.float32)
# for b in range(B):
#     band = cube[:, :, b]
#     cube[:, :, b] = (band - band.min()) / (band.max() - band.min() + 1e-8)

# # === Compute grid size ===
# cols = math.ceil(np.sqrt(B))
# rows = math.ceil(B / cols)

# # === Create the figure ===
# fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
# fig.suptitle("All Spectral Bands", fontsize=16)

# # === Plot each band ===
# for i in range(rows * cols):
#     r = i // cols
#     c = i % cols
#     ax = axes[r, c] if rows > 1 else axes[c]
#     if i < B:
#         ax.imshow(cube[:, :, i], cmap='gray')
#         ax.set_title(f'Band {i}', fontsize=6)
#     ax.axis('off')

# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.savefig("all_bands_grid_indian_pines_corrected.png", dpi=300)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Load the hyperspectral dataset ===
data = loadmat('Indian_pines.mat')  # or 'Indian_pines.mat'
cube = data['indian_pines']         # or 'indian_pines'

# === Reshape the cube: (H, W, B) --> (H*W, B) ===
H, W, B = cube.shape
X = cube.reshape(-1, B).astype(np.float32)

# === Normalize (mean 0, std 1) ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Apply PCA ===
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# === Reshape the PCA output to (H, W, 3) ===
pca_image = X_pca.reshape(H, W, 3)

# === Normalize for visualization (min-max) ===
pca_image -= pca_image.min()
pca_image /= pca_image.max()
print(pca.explained_variance_ratio_)
print("Variance totale expliquée :", np.sum(pca.explained_variance_ratio_)*100)

# === Show and save the result ===
plt.imshow(pca_image)
plt.title("Image RGB issue de l’ACP (3 composantes principales)")
plt.axis('off')
plt.tight_layout()
plt.savefig("pca_projection_Indian_pines.png", dpi=300)
plt.show()

# Optional: print explained variance
print("Variance expliquée par composante :", pca.explained_variance_ratio_)
print("Variance totale expliquée :", np.sum(pca.explained_variance_ratio_))
