import numpy as np

# Solving Systems of Linear Equations using Matrices #
M = np.array(
    [
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5],
    ],
    dtype=float,
)

b = np.array([-10, 0, 17], dtype=float)


### SOLVE ###
x = np.linalg.solve(M, b)
print(f"Solutions: {x}")


### DETERMINANT ###
d = np.linalg.det(M)
print(f"Determinant is {d:.2f}")


# Solving System of Linear Equations using Row Reduction #

### PREPARE ###
A = np.hstack((M, b.reshape(-1, 1)))


### MULTIPLY ROW ###
def MultiplyRow(M, row, num):
    if num != 0:
        new = M.copy()
        new[row] = new[row] * num
    else:
        raise ValueError("num mustn't be equal to zero")
    return new


### ADD ROWS ###
def AddRows(M, row1, row2, num):
    if num != 0:
        new = M.copy()
        new[row1] = new[row1] + new[row2] * num 
    else:
        raise ValueError("num mustn't be equal to zero")
    return new


### SWAP ROWS ###
def SwapRows(M, row1, row2):
    new = M.copy()
    new[[row1, row2]] = new[[row2, row1]]
    return new


### SOLVE ###
# print(A)
B = SwapRows(A, 0, 2)
# print(B)
B = AddRows(B, 1, 0, 2)
# print(B)
B = AddRows(B, 2, 0, 4)
# print(B)
B = AddRows(B, 2, 1, -1)
# print(B)
B = MultiplyRow(B, 0, -1)
# print(B)
B = MultiplyRow(B, 1, 1/5)
# print(B)
B = MultiplyRow(B, 2, -1/12)
print(B)


### SOLUTION ###
z = -2
y = 6.8 + 1.4 * z
x = -17 + 2 * y - 5 * z
print(f"Solutions: {([x, y, z])}")