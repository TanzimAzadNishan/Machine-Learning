import numpy as np

n = input("Enter the value of n: ")
n = int(n)

# generate a random n*n invertible symmetric matrix with integer entries
A = np.random.randint(-10, 10, (n, n))

while True:
    det_A = np.linalg.det(A)
    if det_A != 0:
        break
    
    A = np.random.randint(-10, 10, (n, n))

# make A symmetric in the efficient way
A = A + A.T
# A = A.dot(A.T) # this is costly than the above line

print("A = \n", A)
print("\n")

# Perform Eigen Decomposition using NumPy’s library function
eigenvalues, eigenvectors = np.linalg.eig(A)

# print("Eigenvalues = \n", eigenvalues)
# print("Eigenvectors = \n", eigenvectors)
# print("\n")


# Reconstruct A from eigenvalue and eigenvectors
Q = eigenvectors
cap_Lambda = np.diag(eigenvalues)

A_reconstructed = Q.dot(cap_Lambda).dot(Q.T)

print("A_reconstructed = \n", A_reconstructed)
print("\n")

# Check if A_reconstructed is equal to A
print("is reconstruction perfect?: ", np.allclose(A, A_reconstructed))

