import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T
    
    def inverse_transform(self, trans):
        trans = trans @ self.components
        trans += self.mean
        return trans
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    


# np.random.seed(42)
# n_samples = 500

# t = np.random.uniform(0, 2 * np.pi, n_samples)
# x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
# x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
# x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

# X_synthetic = np.column_stack([x1, x2, x3])

# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X_synthetic)

# print(f"Original shape: {X_synthetic.shape}")
# print(f"Reduced shape:  {X_reduced.shape}")
# print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
# print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")



from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

# for i in [5, 30, 100]:
#     pca_mnist = PCA(n_components=i)
#     X_pca = pca_mnist.fit_transform(X_mnist)
#     X = pca_mnist.inverse_transform(X_pca)
#     print(f"Reconstruction error: {np.mean((X - X_mnist)**2)}")



# from sklearn.decomposition import PCA as SklearnPCA
# from sklearn.manifold import TSNE

# sklearn_pca = SklearnPCA(n_components=2)
# X_sklearn_pca = sklearn_pca.fit_transform(X_mnist)

# print(f"\nOur PCA explained variance:     {pca_2d.explained_variance_ratio_}")
# print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")

# diff = np.abs(np.abs(X_pca2d) - np.abs(X_sklearn_pca))
# print(f"Max absolute difference: {diff.max():.10f}")

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# X_tsne = tsne.fit_transform(X_mnist)
# print(f"\nt-SNE output shape: {X_tsne.shape}")



# try:
#     from umap import UMAP

#     reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
#     X_umap = reducer.fit_transform(X_mnist)
#     print(f"UMAP output shape: {X_umap.shape}")
# except ImportError:
#     print("Install umap-learn: pip install umap-learn")


# from sklearn.decomposition import PCA as SklearnPCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# X_train, X_test, y_train, y_test = train_test_split(
#     X_mnist, y_mnist, test_size=0.2, random_state=42
# )

# results = {}
# for k in [10, 30, 50, 100, 200]:
#     pca_k = SklearnPCA(n_components=k)
#     X_tr = pca_k.fit_transform(X_train)
#     X_te = pca_k.transform(X_test)

#     clf = LogisticRegression(max_iter=1000, random_state=42)
#     clf.fit(X_tr, y_train)
#     acc = accuracy_score(y_test, clf.predict(X_te))
#     var_captured = sum(pca_k.explained_variance_ratio_)
#     results[k] = (acc, var_captured)
#     print(f"k={k:>3d} accuracy={acc:.4f} variance={var_captured:.4f}")

# from sklearn.manifold import TSNE

# for p in [5, 30, 100]:
#     tsne = TSNE(n_components=2, perplexity=p, random_state=42)
#     X_tsne = tsne.fit_transform(X_mnist)
#     print(f"\nt-SNE std: {X_tsne.std():.4f}")


from sklearn.datasets import make_classification

X, y = make_classification(n_features=50, n_informative=5, n_redundant=0, n_repeated=0, n_clusters_per_class=1)

pca = PCA(n_components=50)

dataset_pca = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))
