import numpy as np

A = np.array(
    [
        [-1, 3],
        [3, 2],
    ],
    dtype=float,
)

b = np.array([7, 1], dtype=float)


### SOLVE ###
x = np.linalg.solve(A, b)
print(x)


### DETERMINANT ###
d = np.linalg.det(A)
print(f"{d:.2f}")


### STACK ###
M = np.hstack((A, b.reshape(-1, 1)))
print(M)
