import numpy as np
import matplotlib.pyplot as plt
import utils


### EIGENVICTORS & EIGENVALUES ###
A = np.array(
    [
        [2, 3],
        [2, 1],
    ]
)
A_eig = np.linalg.eig(A)
eigenvalues = A_eig[0]
eigenvectors = A_eig[1]
# print(f"Matrix A:\n{A}\n")
# print(f"Eigenvalues of matrix A:\n{eigenvalues}\n")
# print(f"Eigenvectors of matrix A:\n{eigenvectors}\n")
# print(f"First eigenvector of matrix A:\n{eigenvectors[0]}\n")


### REFLECTION ABOUT Y AXIS ###
### EIGENVICTORS HAVE DIFFERENT DIRECTIONS ###
### EIGENVALUES HAVE DIFFERENT VALUES ###
reflection_y = np.array(
    [
        [-1, 0],
        [0, 1],
    ]
)
reflection_y_eig = np.linalg.eig(reflection_y)
# print(f"Matrix A:\n{reflection_y}\n")
# print(f"Eigenvalues of matrix A:\n{reflection_y_eig[0]}\n")
# print(f"Eigenvectors of matrix A:\n{reflection_y_eig[1]}\n")


### SHEAR ON X DIRECTION ###
### GETTING EIGENVECTORS & VALUES DOSEN'T PRESERVE THE ROLES ###
### CHANGE DIRECTIONS ###
shear_x = np.array(
    [
        [1, 0.5],
        [0, 1],
    ]
)
shear_x_eig = np.linalg.eig(shear_x)
# print(f"Matrix A:\n{shear_x}\n")
# print(f"Eigenvalues of matrix A:\n{shear_x_eig[0]}\n")
# print(f"Eigenvectors of matrix A:\n{shear_x_eig[1]}\n")


### ROTATION ###
### GETTING EIGENVECTORS UNREAL NUMS DOSEN'T PRESERVE THE ROLES ###
### CHANGE DIRECTIONS ###
rotation = np.array(
    [
        [0, 1],
        [-1, 0],
    ]
)
rotation_eig = np.linalg.eig(rotation)
# print(f"Matrix A:\n{rotation}\n")
# print(f"Eigenvalues of matrix A:\n{rotation_eig[0]}\n")
# print(f"Eigenvectors of matrix A:\n{rotation_eig[1]}\n")


### IDENTITY MATRIX ###
### PRESERVE EIGENVECTORS & VALUES WITH VALUES ALWATS = 1 ###
I = np.array(
    [
        [1, 0],
        [0, 1],
    ]
)
I_eig = np.linalg.eig(I)
# print(f"Matrix A:\n{I}\n")
# print(f"Eigenvalues of matrix A:\n{I_eig[0]}\n")
# print(f"Eigenvectors of matrix A:\n{I_eig[1]}\n")


### SCALING ###
### PRESERVE THE EIGEN ROLE ###
scaling = np.array(
    [
        [2, 0],
        [0, 2],
    ]
)
scaling_eig = np.linalg.eig(scaling)
# print(f"Matrix A:\n{scaling}\n")
# print(f"Eigenvalues of matrix A:\n{scaling_eig[0]}\n")
# print(f"Eigenvectors of matrix A:\n{scaling_eig[1]}\n")


### PROJECTION ON X AXIS ###
projection = np.array(
    [
        [1, 0],
        [0, 0],
    ]
)
projection_eig = np.linalg.eig(projection)
# print(f"Matrix A:\n{projection}\n")
# print(f"Eigenvalues of matrix A:\n{projection_eig[0]}\n")
# print(f"Eigenvectors of matrix A:\n{projection_eig[1]}\n")