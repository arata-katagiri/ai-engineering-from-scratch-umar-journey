import math
import random

def l1_norm(x):
    return sum(abs(xi) for xi in x)

def l2_norm(x):
    return math.sqrt(sum(xi**2 for xi in x))

def lp_norm(x, p):
    return sum(abs(xi)**p for xi in x) ** (1/p)

def linf_norm(x):
    return max(abs(xi) for xi in x)

def l1_distance(x, y):
    return sum(abs(xi - yi) for xi, yi in zip(x, y))

def l2_distance(x, y):
    return math.sqrt(sum((xi - yi)**2 for xi, yi in zip(x, y)))

def lp_distance(x, y, p):
    diff = [xi - yi for xi, yi in zip(x, y)]
    return lp_norm(diff, p)

def linf_distance(x, y):
    return max(abs(xi - yi) for xi, yi in zip(x, y))

def dot_product(x, y):
    return sum(xi * yi for xi, yi in zip(x, y))

def cosine_similarity(x, y):
    dot = dot_product(x, y)
    norm_x = l2_norm(x)
    norm_y = l2_norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return dot / (norm_x * norm_y)

def cosine_distance(x, y):
    return 1 - cosine_similarity(x, y)

def mahalanobis_distance(x, y, cov_matrix):
    n = len(x)
    diff = [xi - yi for xi, yi in zip(x, y)]

    inv_cov = invert_matrix(cov_matrix)

    temp = [0.0] * n
    for i in range(n):
        for j in range(n):
            temp[i] += diff[j] * inv_cov[j][i]

    result = sum(temp[i] * diff[i] for i in range(n))
    return math.sqrt(max(0, result))

def invert_matrix(matrix):
    n = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(augmented[row][col]) > abs(augmented[max_row][col]):
                max_row = row
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]

            pivot = augmented[col][col]
            if abs(pivot) < 1e-12:
                raise ValueError("Matrix is singular or near-singular")
            for j in range(2 * n):
                augmented[col][j] /= pivot

            for row in range(n):
                if row != col:
                    factor = augmented[row][col]
                    for j in range(2 * n):
                        augmented[row][j] -= factor * augmented[col][j]
            
            return [row[n:] for row in augmented]
        
def jaccard_similarity(x, y):
    if not x and not y:
        return 1.0
    intersection = len(x & y)
    union = len(x | y)
    return intersection / union

def jaccard_distance(x, y):
    return 1.0 - jaccard_similarity(x, y)

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        df[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = df[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )
    return dp[m][n]

def kl_divergence(p, q):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * math.log(pi / qi)
    return total

def wasserstein_1d(p, q):
    assert len(p) == len(q), "Distributions must have the same number of bins"
    n = len(p)
    cdf_p = [0.0] * n
    cdf_q = [0.0] * n

    cdf_p[0] = p[0]
    cdf_q[0] = q[0]
    for i in range(1, n):
        cdf_p[i] = cdf_p[i - 1] + p[i]
        cdf_q[i] = cdf_q[i - 1] + q[i]
    
    return sum(abs(cdf_p[i] - cdf_q[i]) for i in range(n))

def compute_covariance(data):
    n = len(data)
    d = len(data[0])
    means = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
    centered = [[data[i][j] - means[j] for j in range(d)] for i in range(n)]
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            cov[i][j] = sum(centered[k][j] * centered[k][j] for k in range(n)) / (n - 1)
    return cov

def normalize_vector(v):
    norm = l2_norm(v)
    if norm == 0:
        return v[:]
    return [vi / norm for vi in v]

def find_nearest_neighbor(query, dataset, distance_fn, **kwargs):
    best_idx = 0
    best_dist = float('inf')
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist

def find_k_nearest(query, dataset, distance_fn, k=5, **kwargs):
    distances = []
    for i, point in enumerate(dataset):
        d = distance_fn(query, point, **kwargs)
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return distances[:k]