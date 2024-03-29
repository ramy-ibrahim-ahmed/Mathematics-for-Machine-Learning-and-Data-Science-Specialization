import numpy as np


### GENERAL TRANSFORMATION ###
### VECTOR V TO W ###
def T(v):
    w = np.zeros(shape=(3, 1))
    w[0, 0] = 3 * v[0, 0]
    w[2, 0] = 2 * v[1, 0]
    return w


v = np.array(
    [
        [3],
        [5],
    ]
)
w = T(v)
# print("Original vector:\n", v, "\n\nResult of the transformation:\n", w)


### LINEAR TRANSFORMATION ###
u = np.array(
    [
        [1],
        [-2],
    ]
)
v = np.array(
    [
        [2],
        [4],
    ]
)
k = 7
# print("T(k*v):\n", T(k * v), "\n\nk * T(v):\n", k * T(v), "\n\n")
# print("T(u+v):\n", T(u + v), "\n\n T(u) + T(v):\n", T(u) + T(v))


### TRANSFORMATION DEFINED AS A MATRIX MULTIPLICATION ###
def L(v):
    A = np.array(
        [
            [3, 0],
            [0, 0],
            [0, -2],
        ]
    )
    print("Transformation matrix:\n", A, "\n")
    w = A @ v
    return w


v = np.array([[3], [5]])
# w = L(v)
# print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)


### HORIZONTAL SCALING (DILTATION) ###
def T_scaler(V):
    A = np.array([[2,0], [0,1]])
    W = A @ V
    return W


e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])
V = np.hstack((e1, e2))
T_scaling = T_scaler(V)
# print("Original vectors:\n e1= \n", e1, "\n\n e2=\n", e2, "\n\n Result of the transformation (matrix form):\n", result)


### REFLECTION ABOUT Y_AXIS ###
def T_reflection_yaxis(v):
    A = np.array([[-1,0], [0,1]])
    w = A @ v
    return w


e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])
V = np.hstack((e1, e2))
T_yaxis = T_reflection_yaxis(V)
# print("Original vectors:\n e1= \n", e1,"\n\n e2=\n", e2, "\n\n Result of the transformation (matrix form):\n", T_yaxis)