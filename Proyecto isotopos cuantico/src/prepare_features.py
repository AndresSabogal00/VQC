import pandas as pd

df = pd.read_csv("../data/dataset_counts.csv")

# Avoid division by zero
df["counts_free"] = df["counts_free"].replace(0, 1)

df["Al_ratio"] = df["counts_Al"] / df["counts_free"]
df["Cu_ratio"] = df["counts_Cu"] / df["counts_free"]
df["Pb_ratio"] = df["counts_Pb"] / df["counts_free"]

df_features = df[[
    "isotope", "distance_cm",
    "counts_free", "counts_Al", "counts_Cu", "counts_Pb",
    "Al_ratio", "Cu_ratio", "Pb_ratio"
]]

df_features.to_csv("../data/dataset_features.csv", index=False)

print("dataset_features.csv generado con shape:", df_features.shape)
