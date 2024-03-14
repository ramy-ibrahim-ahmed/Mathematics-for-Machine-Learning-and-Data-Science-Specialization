import numpy as np

### GET THE RANK OF THE MATRIX ###
### APPLY LINEAR TRANSFORMATION TO THE STANDARD BASES BY THE MATRIX ###
### SEE ITS IMAGE DIMANTIONS ###
M = np.array(
    [
        [1, 0],
        [2, 3],
    ]
)
S = np.array(
    [
        [1, 0],
        [0, 1],
    ]
)
linear_T = M @ S
print(f"Rank = {linear_T.ndim}")


### THE AREA OF PARALLELOGRAM SPANNED BY TRANSFORMATION ###
### GET DETERMINANT OF IMAGE BASES TO GET THE AREA ###
### T(0,1)=(2,5)
### T(1,0)=(3,1)
### X(3, 2) | Y(1, 5)
M = np.array(
    [
        [3, 2],
        [1, 5],
    ]
)
print(f"Area = {np.linalg.det(M)}")


### DETERMINANT OF M1 @ M2 @ M3 = det1 * det2 * det3 ###
A = np.array([[2, 1], [3, 1]])
B = np.array([[3, 5], [1, 1]])
C = np.array([[2, 3], [4, 5]])
det1 = np.linalg.det(A)
det2 = np.linalg.det(B)
det3 = np.linalg.det(C)
print(f"Determinant = {det1*det2*det3}")


### DETERMINANT OF INVERSE M IS EQUAL TO THE INVERSE OF DETERMINANT OF M ###
M = np.array(
    [
        [0, 0, 1],
        [2, 2, 1],
        [1, 0, 0],
    ]
)
print(f"Determinant of M^-1 = {1 / np.linalg.det(M)}")