import pandas as pd

df = pd.read_csv("../data/dataset_counts.csv")

print("Filas totales:", len(df))
print(df.head())

print("\nConteos medios por isótopo:")
print(df.groupby("isotope")[["counts_free"]].mean())

print("\nAtenuación promedio por filtro:")
print(df.groupby("isotope")[["counts_Al","counts_Cu","counts_Pb"]].mean())
