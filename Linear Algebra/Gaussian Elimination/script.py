import numpy as np
import w2_unittest


### SWAP ROWS ###
def SwapRows(M, row1, row2):
    new = M.copy()
    new[[row1, row2]] = new[[row2, row1]]
    return new


### INDEX OF NON_ZERO VAL IN COLUMN ###
def NonZeroColumnIndex(M, column, start_row):
    column = M[start_row:, column]
    for i, val in enumerate(column):
        if not np.isclose(val, 0, atol=1e-5):  # 0.00001
            index = start_row + i
            return index
    return None


M = np.array(
    [
        [0, 5, -3, 6, 8],
        [0, 6, 3, 8, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 7],
        [0, 2, 1, 0, 4],
    ]
)
print(M)
print(NonZeroColumnIndex(M, 0, 0))
print(NonZeroColumnIndex(M, -1, 2))


### PIVOT IN ROW ###
def pivot(M, row):
    for i, val in enumerate(M[row]):
        if not np.isclose(val, 0, 0.00001):
            return i
    return None


print(M)
print(f"for row 2: {pivot(M, 2)}")
print(f"for row 3: {pivot(M, 3)}")


### MOVE ZEROES ROW BOTTOM ###
def ZeroesToBottom(M, row):
    new = M.copy()
    zero_row = new[row]
    new = np.delete(new, row, axis=0)
    new = np.vstack([new, zero_row])
    return new


M = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
print(f"Matrix before:\n{M}")
print(f"Matrix after moving index 1:\n{ZeroesToBottom(M, 1)}")


### AUGMENTED ###
def augmented(M, B):
    new = []
    for i in range(len(M)):
        new.append(np.insert(M[i], len(M[0]), B[i]))
    return np.array(new)


A = np.array(
    [
        [1, 2, 3],
        [3, 4, 5],
        [4, 5, 6],
    ]
)
B = np.array(
    [
        [1],
        [5],
        [7],
    ]
)
print(augmented(A, B))


### REDUCE ROW ECHELON FORM ###
def ReducedRowEchelon(A, B):
    # PRERARE A & B #
    A = A.copy()
    B = B.copy()
    A = A.astype("float64")
    B = B.astype("float64")

    # AGUMENT MATRICES #
    M = augmented(A, B)

    # INITIALIZE LIST OF ZEROES ROWS #
    rows_to_move = []

    # ITERATIONS FOR ALL ROWS #
    for i in range(len(A)):

        # DEFAULT PIVOT #
        curr_pivot = M[i, i]
        column = i

        # IF DEFAULT PIVOT IS ZERO #
        if np.isclose(curr_pivot, 0):

            # SUPPOSE THE COLUMN HAS A NON_ZERO VALUE #
            real_row_to_swap = NonZeroColumnIndex(M, column, start_row=i)

            # IF NON_ZERO VAL IN CURRENT COLUMN #
            # SWAP THE ROW WITH NON_ZERO & CURRENT ROW #
            # UPDATE CURRENT PIVOT #
            if real_row_to_swap is not None:
                M = SwapRows(M, i, real_row_to_swap)
                curr_pivot = M[i, i]

            # IF NO NON_ZERO VAL IN CURRENT COLUMN #
            # SEARCH IN CURRENT ROW FOR NON_ZERO VALUE'S INDEX #
            if real_row_to_swap is None:
                pivot_index = pivot(M, i)

                # IF NO NON_ZERO VALUE IN CURRENT ROW #
                # OR NON_ZERO VALUE IS IN THE LAST COLUMN #
                # APPEND CURRENT ROW TO LIST OF ZEROES ROWS #
                # GO NEXT ITERATION FOR SEARCHING ON NEXT ROW #
                if pivot_index is None or pivot_index >= len(A):
                    rows_to_move.append(i)
                    continue

                # IF WE GOT NON_ZERO VALUE IN CURRENT ROW #
                # UPDATE PIVOT WITH THAT VALUE #
                # UPDATE COLUMN WITH THE NEW PIVOT INDEX #
                else:
                    curr_pivot = M[pivot_index, i]
                    column = pivot_index

        # SUPTRACT CURRENT ROW WITH PIVOT TO GET PIVOT = 1 #
        M[i] /= curr_pivot

        # ITERATE FOR ALL ROWS ABOVE CURRENT ROW #
        # GET VALUE ABOVE PIVOT TO USE IT IN CALCULATIONS FOR MAKING IT = 0 #
        for j in range(i + 1, len(A)):
            value_below_pivot = M[j, column]
            M[j] = M[j] - value_below_pivot * M[i]

    # MOVE ALL ZEROES ROWS TO BOTTOM #
    for row in rows_to_move:
        M = ZeroesToBottom(M, row)

    # RETURN REDUCEC ROW ECHELON FORM #
    return M


w2_unittest.test_reduced_row_echelon_form(ReducedRowEchelon)


### CHECKING SOLUTIONS ###
def CheckSolution(M):
    M = M.copy()

    # PREPARE COEFFICIENT & CONSTANT #
    coefficient_matrix = M[:, :-1]
    constant_vector = M[:, -1]
    singular = False

    # ITERATE OVER ROWS #
    for i in range(len(M)):

        # IF NO NON_ZERO VALUE IN ROW SET AS SINGULAR #
        if pivot(coefficient_matrix, i) is None:
            singular = True

            # IF CONSTANT != SET AS NO SOLUTIONS #
            if not np.isclose(constant_vector[i], 0):
                return "No solution."

    # CHECK FOR SINGULARITY #
    if singular:
        return "Infinitely many solutions."
    else:
        return "Unique solution."


w2_unittest.test_check_solution(CheckSolution)


### BACK SUBSTITUTION ###
def BackSubstitution(M):
    M = M.copy()

    # ITERATE FROM LAST ROW #
    # GET SUBSTITUTION ROW #
    for i in range(len(M) - 1, -1, -1):
        substitution_row = M[i]

        # ITERATE FROM SUBSTITUTION TO ABOVE #
        # GET SUBSTITUTION PIVOT INDEX TO UPDATE ABOVE #
        for j in range(i - 1, -1, -1):
            pivot_index = pivot(M, i)

            # CALCULATE ABOVE USING PIVOT VALUE #
            # UPDATE ABOVE ROW #
            row_to_reduce = M[j]
            value_to_reduce = row_to_reduce[pivot_index]
            row_to_reduce = row_to_reduce - value_to_reduce * substitution_row
            M[j] = row_to_reduce

    # GET SOLUTIONS FINAL COLUMN #
    solutions = M[:, -1]
    return solutions


w2_unittest.test_back_substitution(BackSubstitution)


### GAUSSIAN ELIMINATION ###
def GaussianElimination(A, B):
    # GET MATRIX IN ROW ECHELON FORM #
    M = ReducedRowEchelon(A, B)
    # CHECK THE TYPE OF SOLUTION #
    solution_type = CheckSolution(M)
    if solution_type == "Unique solution.":
        # GET SOLUTIONS BY BACK SUBSTITUTION METHOD #
        solutions = BackSubstitution(M)
        return solutions
    return solution_type


w2_unittest.test_gaussian_elimination(GaussianElimination)