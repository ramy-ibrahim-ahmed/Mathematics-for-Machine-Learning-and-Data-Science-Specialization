import numpy as np

### GETTING X1 ###
### X0 THAT PRESERVE VALUE 4 ###
P = np.array(
    [
        [0, 0.75, 0.35, 0.25, 0.85],
        [0.15, 0, 0.35, 0.25, 0.05],
        [0.15, 0.15, 0, 0.25, 0.05],
        [0.15, 0.05, 0.05, 0, 0.05],
        [0.55, 0.05, 0.25, 0.25, 0],
    ]
)
X0 = np.array([[0], [0], [0], [1], [0]])
X1 = P @ X0
print(f"Sum of columns of P: {sum(P)}")
print(f"X1:\n{X1}")


### FIND VECTOR X20 ###
X = np.array([[0], [0], [0], [1], [0]])
m = 20
for t in range(m):
    X = P @ X
print(X)


### USE EIGENVICTORES & EIGENVALUES ###
eigenvals, eigenvecs = np.linalg.eig(P)
print(f"Eigenvalues of P:\n{eigenvals}\nEigenvectors of P\n{eigenvecs}")
X_inf = eigenvecs[:, np.argmax(eigenvals)]  # FOR THE BIGGEST EIGENVALUE
print(f"Eigenvector corresponding to the eigenvalue 1:\n{X_inf[:,np.newaxis]}")


def check_eigenvector(P, X_inf):
    X_check = P @ X_inf
    return X_check


X_check = check_eigenvector(P, X_inf)
print("Original eigenvector corresponding to the eigenvalue 1:\n" + str(X_inf))
print("Result of multiplication:" + str(X_check))