import numpy as np

### Q1 ###
M = np.array(
    [
        [1, 3, 15],
        [3, 12, 3],
    ],
    dtype=float,
)
M[1] = M[0] * -3 + M[1]
M[1] /= 3
M[0] = M[1] * -3 + M[0]
print(M)


### Q2 ###
M = np.array(
    [
        [2, 1, 5],
        [4, 2, 10],
    ],
    dtype=float,
)
M[0] /= 2
M[1] = M[0] * -4 + M[1]
print(M)


### Q3 ###
M = np.array(
    [
        [1, 2, 3, 10],
        [2, 6, 12, 4],
        [4, -8, 4, 8],
    ],
    dtype=float,
)
M[1] = M[0] * -2 + M[1]
M[2] = M[0] * -4 + M[2]
M[1] /= 2
M[2] = M[1] * 16 + M[2]
M[2] /= 40
print(M)  ## z = -4


### Q7 ###
M = np.array(
    [
        [2, 1, 5],
        [1, 3, 1],
        [3, 4, 6],
    ],
    dtype=float,
)
M[0] /= 2
M[1] = M[0] * -1 + M[1]
M[2] = M[0] * -3 + M[2]
M[1] /= 2.5
M[2] /= 2.5
M[2] = M[1] * -1 + M[2]
print(M)  ## rank = 2


### Q8 ###
M = np.array([[2, 3], [1, 7]])
print(np.linalg.det(M))
M[[0, 1]] = M[[1, 0]]
print(np.linalg.det(M))
M[0] *= 5
print(np.linalg.det(M))