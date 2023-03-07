import numpy as np

n = input("Enter the value of n: ")
m = input("Enter the value of m: ")
n = int(n)
m = int(m)

A = np.random.randint(-10, 10, (n, m))

print("A = \n", A)
print("\n")

# Singular Value Decomposition using numpy
U, singular_values, V_T = np.linalg.svd(A, full_matrices=True)

print("U = \n", U)
print("singular_values = \n", singular_values)
print("V_T = \n", V_T)

# Moore-Penrose Pseudoinverse using numpy
A_pseudo_inv = np.linalg.pinv(A)

print("\nA_pseudo_inv = \n", A_pseudo_inv)
print("\n")

#------------------ Moore-Penrose Pseudoinverse using eqn 2.47 ------------------------

# generate a pseudo inverse diagonal matrix
D_pseudo_inv = np.zeros((n, m))

for i in range(min(n, m)):
    if singular_values[i] != 0:
        D_pseudo_inv[i, i] = 1/singular_values[i]


D_pseudo_inv = D_pseudo_inv.T
# print("D_pseudo_inv = \n", D_pseudo_inv)
# print("\n")


A_inv_reconstructed = np.dot(np.dot(V_T.T, D_pseudo_inv), U.T)

print("A_inv_reconstructed = \n", A_inv_reconstructed)
print("\n")

# Check if A_reconstructed is equal to A_pseudo_inv
print("is reconstruction perfect?: ", np.allclose(A_pseudo_inv, A_inv_reconstructed))