import numpy as np


### Q1 ###
u = np.array([[1], [0], [7]])
w = np.array([[0], [-1], [2]])
diff = []
d2 = 0.0
for i in range(len(u)):
    diff.append(u[i] - w[i])
for i in range(len(u)):
    d2 += diff[i] * diff[i]
d = np.sqrt(d2)
# print("Distance between v and w:", d[0])


### Q2 ###
p = np.array(
    [
        [1],
        [0],
        [-3],
    ]
)
q = np.array(
    [
        [-1],
        [0],
        [-3],
    ]
)
magnitude = np.linalg.norm(p - q, ord=2)
# print(magnitude)


### Q4 ###
u = np.array(
    [
        [1],
        [-5],
        [2],
        [0],
        [-3],
    ]
)
norm2 = 1**2 + (5**2) + 2**2 + 0**2 + (3**2)
norm = np.sqrt(norm2)
norm = np.linalg.norm(u, ord=2)
# print(norm)


### Q6 ###
a = np.array(
    [
        [3],
        [7],
        [1],
    ]
)
b = np.array(
    [
        [4],
        [0],
        [3],
    ]
)
a = a.T
dot_ab = 0.0
for i in range(len(b)):
    dot_ab += a[0, i] * b[i, 0]
# print(dot_ab)


### Q7 ###
m1 = np.array(
    [
        [2, -1],
        [3, -3],
    ]
)
m2 = np.array(
    [
        [5, -2],
        [0, 1],
    ]
)
# print(m1 @ m2)


### Q8 ###
w = np.array(
    [
        [-9],
        [-1],
    ]
)
z = np.array(
    [
        [-3],
        [-5],
    ]
)
w = w.T
dot_wz = 0.0
for i in range(len(z)):
    dot_wz += w[0, i] * z[i, 0]
# print(dot_wz)