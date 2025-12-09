import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load features
df = pd.read_csv("../data/dataset_features.csv")

# Features we will use (ratios + counts)
X = df[[
    "counts_free", "counts_Al", "counts_Cu", "counts_Pb",
    "Al_ratio", "Cu_ratio", "Pb_ratio"
]]

y = df["isotope"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame with PCA components
df_pca = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "isotope": y
})

# Save
df_pca.to_csv("../data/dataset_pca.csv", index=False)
print("dataset_pca.csv generado. Shape:", df_pca.shape)

# Quick scatter plot
plt.figure(figsize=(7, 6))
for iso in df_pca["isotope"].unique():
    subset = df_pca[df_pca["isotope"] == iso]
    plt.scatter(subset["PC1"], subset["PC2"], label=iso, alpha=0.5, s=20)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA — Componentes principales (2D)")
plt.legend()
plt.tight_layout()
plt.show()
