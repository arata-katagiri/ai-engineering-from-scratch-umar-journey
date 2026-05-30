import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)
    
    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V

# np.random.seed(42)
# A = np.random.randn(5, 4)

# U_ours, S_ours, V_ours = svd_from_scratch(A)
# U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

# print("Our singular values:", np.round(S_ours, 4))
# print("Numpy singular values:", np.round(S_np, 4))

# A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
# print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")

def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

# image = np.random.seed(42)
# rows, cols = 200, 300
# image = np.random.randn(rows, cols)

# for k in [1, 5, 10, 20, 50]:
#     compressed = compress_image_svd(image, k)
#     error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
#     original_size = rows * cols
#     compressed_size = k * (rows + cols + 1)
#     ratio = compressed_size / original_size
#     print(f"k={k:>3d} error={error:.4f} storage={ratio:.1%}")

# np.random.seed(42)
# clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
#                  np.cos(np.linspace(0, 2*np.pi, 80)))
# noise = 0.3 * np.random.randn(100, 80)
# noisy = clean + noise

# U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
# denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

# print(f"Noisy error: {np.linalg.norm(noisy - clean):.4f}")
# print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
# print(f"Improvement: {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")

# A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
# b = np.array([3, 5, 6], dtype=float)

# U, S, Vt = np.linalg.svd(A, full_matrices=False)
# S_inv = np.diag(1.0 / S)
# A_pinv = Vt.T @ S_inv @ U.T

# x_svd = A_pinv @ b
# x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
# x_pinv = np.linalg.pinv(A) @ b

# print(f"SVD pseudoinverse solution: {x_svd}")
# print(f"np.linalg.lstsq solution: {x_lstsq}")
# print(f"np.linalg.pinv solution: {x_pinv}")

def svd_via_eigh(A, k=None):
    m = A.shape[0]
    AtA = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eigh(AtA)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    k_valid = np.sum(eigenvalues > 1e-10)

    if k is None:
        k = k_valid

    k = min(k, k_valid)
    V = eigenvectors[:, :k]
    us = []
    for i in range(k):
        u = A @ V[:, i] / np.sqrt(eigenvalues[i])
        us.append(u)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.sqrt(eigenvalues[:k])

    return U, S, V

# np.random.seed(42)
# A = np.random.randn(5, 4)

# U1, S1, V1 = svd_from_scratch(A)
# U2, S2, V2 = svd_via_eigh(A)
# U3, S3, V3_t = np.linalg.svd(A, full_matrices=False)

# print("Power iteration:", np.round(S1, 6))
# print("eigh:           ", np.round(S2, 6))
# print("NumPy:          ", np.round(S3, 6))

# A1 = U1 @ np.diag(S1) @ V1.T
# A2 = U2 @ np.diag(S2) @ V2.T

# print(f"Power iteration reconstruction error: {np.linalg.norm(A - A1):.2e}")
# print(f"eigh reconstruction error:            {np.linalg.norm(A - A2):.2e}")

# from skimage import data
# img = data.camera()

import matplotlib.pyplot as plt

# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# axes = axes.flatten()

# axes[0].imshow(img, cmap='gray')
# axes[0].set_title('Original')

# img = img.astype(float)
# rows, cols = img.shape
# for i, k in enumerate([1, 5, 10, 25, 50, 100]):
#     compressed = compress_image_svd(img, k)
#     error = np.linalg.norm(img - compressed) / np.linalg.norm(img)
#     original_size = rows * cols
#     compressed_size = k * (rows + cols + 1)
#     ratio = compressed_size / original_size
#     print(f"k={k:>3d} error={error:.4f} storage={ratio:.1%}")

#     axes[i+1].imshow(compressed, cmap='gray')
#     axes[i+1].set_title(f"k={k}")


# plt.show()

# ratings = np.full((10, 8), np.nan)

# # Users 0-3: action fans (cols 0-2)
# ratings[0, [0,1,2,4]] = [5, 4, 5, 2]
# ratings[1, [0,2,3,6]] = [4, 5, 2, 1]
# ratings[2, [1,2,5,7]] = [5, 4, 1, 2]
# ratings[3, [0,1,6,7]] = [4, 5, 2, 1]

# # Users 4-6: romance fans (cols 3-5)
# ratings[4, [3,4,5,1]] = [5, 5, 4, 2]
# ratings[5, [3,5,0,2]] = [4, 5, 1, 2]
# ratings[6, [3,4,5,6]] = [5, 4, 5, 2]

# # Users 7-9: comedy fans (cols 6-7)
# ratings[7, [6,7,3,4]] = [5, 5, 2, 1]
# ratings[8, [6,7,0,1]] = [4, 5, 1, 2]
# ratings[9, [6,7,4,5]] = [5, 4, 2, 1]

# for i in range(10):
#     row_mean = np.nanmean(ratings[i])

#     ratings[i, np.isnan(ratings[i])] = row_mean

# compressed = compress_image_svd(ratings, 3)
# print("Original ratings:")
# print(np.round(ratings, 2))

# print("\nReconstructed (predicted) ratings:")
# print(np.round(compressed, 2))

# dtm = np.zeros((100, 50))

# dtm[0:33, 0:5] += np.random.rand(33, 5) * 5
# dtm[33:66, 5:10] += np.random.rand(33, 5) * 5
# dtm[66:, 10:15] += np.random.rand(34, 5) * 5

# dtm += np.random.rand(100, 50)

# U, S, V = svd_from_scratch(dtm)

# print(S[:10])

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# coords = U[:, :3]

# ax.scatter(coords[0:33, 0], coords[0:33, 1], coords[0:33, 2], c='red', label='Topic 1')
# ax.scatter(coords[33:66, 0], coords[33:66, 1], coords[33:66, 2], c='blue', label='Topic 2')
# ax.scatter(coords[66:, 0], coords[66:, 1], coords[66:, 2], c='green', label='Topic 3')

# ax.legend()
# plt.show()

A = 0.5 * np.random.randn(50, 3) @ np.random.randn(3, 40)

optimal_ks = []

for sigma in [0.1, 0.5, 1.0, 2.0]:
    noisy_A = A + sigma * np.random.randn(50, 40)

    errors = []

    for k in range(1, 41):
        compressed = compress_image_svd(noisy_A, k)
        error = np.linalg.norm(A - compressed)
        errors.append(error)
    
    optimal_k = np.argmin(errors) + 1

    print(f"sigma={sigma} optimal_k={optimal_k}")

    optimal_ks.append(optimal_k)

plt.scatter([0.1, 0.5, 1.0, 2.0], optimal_ks)
plt.xlabel('Noise level (sigma)')
plt.ylabel('Optimal k')
plt.title('Optimal truncation rank vs noise level')
plt.show()