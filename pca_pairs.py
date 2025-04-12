import numpy as np

# Data (5 patients, 2 variables)
"""
X = np.array([
    [120, 200],
    [130, 220],
    [110, 190],
    [140, 210],
    [125, 235]
])

"""

def pca_pair_dim_reduction(X):
    # Step 1: Center the data (subtract mean)
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Step 3: Find eigenvectors/eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # PC1 is the eigenvector with largest eigenvalue
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]

    # Step 4: Project data onto PC1
    x_star = X_centered.dot(pc1)
    print("Reduced 1D vector x*:", x_star)

    return x_star
