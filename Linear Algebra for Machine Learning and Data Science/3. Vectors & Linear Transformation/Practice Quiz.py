import numpy as np

### Q2 ###
u = np.array([1, 3])
v = np.array([6, 2])
sum_uv = np.zeros(2)
sum_uv[0] = u[0] + v[0]
sum_uv[1] = u[1] + v[1]
# print(sum_uv)


### Q3 ###
diff = np.zeros(2)
diff[0] = u[0] - v[0]
diff[1] = u[1] - v[1]
# print(diff)


### Q4 ###
a = np.array(
    [
        [-1],
        [5],
        [2],
    ]
)
b = np.array(
    [
        [-3],
        [6],
        [-4],
    ]
)
dot_ab = np.zeros(1)
a = a.T
dot_ab[0] = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0]
# print(dot_ab)