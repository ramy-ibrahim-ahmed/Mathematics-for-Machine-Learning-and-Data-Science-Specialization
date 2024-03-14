import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.sparse.linalg import eigsh

np.random.seed(7)

### LOAD THE DATA ###
imgs = utils.load_images("./data/")
height, width = imgs[0].shape
print(f"Images shape: {np.array(imgs).shape}")
print(f"Your dataset has {len(imgs)} images of size {height}x{width} pixels")


### PLOT IMAGE ###
# plt.imshow(imgs[0], cmap="gray")
# plt.show()


### FLAT IMAGES IN ONE ROW FOREACH ###
imgs = np.array([im.flatten() for im in imgs])
print(f"Shape of images after flatting: {imgs.shape}")


### 1. CENTER THE DATA ###
def center_data(Y):
    mean_v = np.mean(Y, axis=0)
    return Y - mean_v


X_mean = center_data(imgs)
# plt.imshow(X_mean[0].reshape(64, 64), cmap="gray")
# plt.show()


### 2. GET COVARIANCE MATRIX ###
def C_matrix(X_mean):
    M = np.dot(X_mean.T, X_mean) / (X_mean.shape[0] - 1)
    return M


C = C_matrix(X_mean)
print(f"Covariance matrix shape: {C.shape}")


### 3. COMPUTE EIGENVECTORS & EIGENVALUES OF C ###
e_val, e_vec = eigsh(C, k=55)
print(f"Ten largest eigenvalues: \n{e_val[-10:]}")
e_val = e_val[::-1]
e_vec = e_vec[:, ::-1]
print(f"Ten largest eigenvalues sorted: \n{e_val[:10]}")


### VISUALIZE COMPONANTS ###
# fig, ax = plt.subplots(4, 4, figsize=(8, 8))
# for n in range(4):
#     for k in range(4):
#         ax[n, k].imshow(e_vec[:, n * 4 + k].reshape(height, width), cmap="gray")
#         ax[n, k].set_title(f"component number {n*4+k+1}", fontsize=8)
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# plt.show()


### 4. PROJECTION ###
def projection(X_mean, e_vec, k):
    V = e_vec[:, :k]
    return np.dot(X_mean, V)


X_pca = projection(X_mean, e_vec, 2)
print(f"Data shape after applying PCA: {X_pca.shape}")


### 5. RECONSTRUCTING ###
def reconstructing(X_pca, e_vec):
    X_mean = np.dot(X_pca, e_vec[:, :X_pca.shape[1]].T)
    return X_mean


X_reconst = reconstructing(X_pca, e_vec)
print(f"Data shape after back from PCA: {X_reconst.shape}")


### 6. EXPLAINED VARIANCE ###
def explained_variance(eigentvalues):
    e_variance = eigentvalues / np.sum(eigentvalues)
    plt.plot(np.arange(1, len(eigentvalues)+1), e_variance, label="Explained variance")
    plt.xlabel("Num of components", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


# explained_variance(e_val)


def explained_variance(eigentvalues, percentage):
    e_variance = eigentvalues / np.sum(eigentvalues)
    e_variance = np.cumsum(e_variance)
    plt.plot(np.arange(1, len(eigentvalues)+1), e_variance, label="Explained variance")
    plt.axhline(y=percentage / 100, c='r', linestyle="--")
    plt.xlabel("Num of components", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", lw=0.5)
    plt.tight_layout()
    plt.show()


# explained_variance(e_val, 95)