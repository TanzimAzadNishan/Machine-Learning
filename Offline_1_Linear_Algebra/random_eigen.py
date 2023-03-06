import numpy as np

n = input("Enter the value of n: ")
n = int(n)

# generate a random n*n invertible matrix with integer entries
A = np.random.randint(-10, 10, (n, n))

while True:
    det_A = np.linalg.det(A)
    if det_A != 0:
        break
    
    A = np.random.randint(-10, 10, (n, n))


print("A = \n", A)
print("\n")


# Perform Eigen Decomposition using NumPy’s library function
eigenvalues, eigenvectors = np.linalg.eig(A)

# print("Eigenvalues = \n", eigenvalues)
# print("Eigenvectors = \n", eigenvectors)
# print("\n")

# Reconstruct A from eigenvalue and eigenvectors
V = eigenvectors
diag_lambda = np.diag(eigenvalues)
V_inv = np.linalg.inv(eigenvectors)
A_reconstructed = V.dot(diag_lambda).dot(V_inv)

print("A_reconstructed = \n", A_reconstructed)
print("\n")

# Check if A_reconstructed is equal to A
print("is reconstruction perfect?: ", np.allclose(A, A_reconstructed))

