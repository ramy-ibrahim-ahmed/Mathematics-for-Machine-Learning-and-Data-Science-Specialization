import numpy as np

### Q1 ###
M = np.array(
    [
        [1, 1, 4],
        [-6, 2, 16],
    ],
    dtype=float,
)

M[1] /= 6
# print(M)
M[1] += M[0]
# print(M)
M[1] /= M[1, 1]
# print(M)
y = 5
x = 4 - y
print(f"x = {x}, y = {y}")


### Q2 ###
d = 4 * -8 - -3 * 7
print(d)


### Q3 ###
M = np.array(
    [
        [-3, 8, 1],
        [2, 2, -1],
        [-5, 6, 2],
    ],
    dtype=float,
)
d = np.linalg.det(M)
print(f"{d:.2f}")