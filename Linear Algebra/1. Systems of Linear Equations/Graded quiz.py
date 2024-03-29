import numpy as np

singularity_condition = lambda d: "Singular" if d == 0 else "Non Singular"

singularity = lambda M: "Singular" if np.linalg.det(M) == 0 else "Non Singular"

### Q3 ###
M = np.array(
    [
        [1, 2, 1],
        [2, 1, 1],
        [-1, 2, 1],
    ],
    dtype=float,
)

d = 1 * (1 * 1 - 1 * 2) - 2 * (2 * 1 - 1 * -1) + 1 * (2 * 2 - 1 * -1)
print(f"Determinant by np.linalg = {np.linalg.det(M):.2f}")
print(f"Determinant by calculations = {d:.2f}")
print(f"Matrix is {singularity_condition(d)}")


### Q4 ###
M = np.array(
    [
        [1, 2, 3],
        [3, 2, 1],
        [2, 2, 1],
    ],
    dtype=float,
)

print(f"Matrix is {singularity(M)} so it's Lineary indeprndent")


### Q5 ###
M = np.array(
    [
        [2, 1, 5],
        [1, 2, 1],
        [3, 3, 6],
    ],
    dtype=float,
)
print(singularity(M))


### Q6 ###
M = np.array(
    [
        [1, 2, 3],
        [0, 2, 2],
        [1, 4, 5],
    ],
    dtype=float,
)
d = np.linalg.det(M)
print(d, singularity_condition(d))