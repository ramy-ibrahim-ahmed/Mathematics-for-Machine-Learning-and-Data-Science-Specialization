import numpy as np

### FIND THE RANK OF TRANSFORMATIONS ###
M = np.array(
    [
        [1, 3, -4],
        [2, -1, -3],
        [4, 5, -11],
    ]
)
print(f"The rank is {np.linalg.matrix_rank(M)}")


### det(M@N) = DET(M) * DET(N) ###
M = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
N = np.array([[2, 8, 7], [4, 3, 9], [1, 5, 9]])
det_MN = np.linalg.det(M) * np.linalg.det(N)
print(f"Det M @ N = {det_MN}")


### SPAN OF VECTORS GIVES BY IT'S LINEAR DEPENDENCY ###
v1 = np.array([[2], [1], [1]])
v2 = np.array([[1], [0], [2]])
v3 = np.array([[1], [0], [1]])
M = np.hstack([v1, v2, v3])
if np.linalg.det(M) == 0:
    print("Dependency acheived")
else:
    print("Independency achieved")


### 3D SPACE BASIS ###
M = np.array([[-1, 2, 1], [1, 2, 2], [1, 6, 5]])
M = np.array([[2, 3, 1], [0, 2.5, 3.5], [0, 5.5, 4.5]])
M = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
M = np.array([[-1, 2, 1], [1, 2, 2], [1, 0, 1]])
if np.linalg.det(M) == 0:
    print("Dependency acheived")
else:
    print("Independency achieved")


### COVARIANCE MATRIX ###
M = np.array([[3, 2], [5, 8]])
mean_v = np.mean(M, axis=0)
X_mean = M - mean_v
C = (X_mean.T @ X_mean) / (M.shape[0] - 1)
print(C)


### X - MEAN ###
M = np.array([[70, 2, 2], [110, 4, 2]])
X_mean = M - np.mean(M, axis=0)
print(X_mean)
C = (X_mean.T @ X_mean) / (M.shape[0] - 1)
C_eig = np.linalg.eigvals(C)
print(f"Eigenvalues: {C_eig}")