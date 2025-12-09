import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/dataset_counts.csv")
# tomar un par de filas de ejemplo del isótopo Cs137 a 10 cm
ex = df[(df.isotope=="Cs137") & (df.distance_cm==10)].sample(5)
print(ex[["counts_free","counts_Al","counts_Cu","counts_Pb"]])
# boxplot de conteos por filtro para cada isótopo (dist=10cm)
subset = df[df.distance_cm==10]
subset.groupby("isotope")[["counts_free","counts_Al","counts_Cu","counts_Pb"]].boxplot(rot=45)
plt.tight_layout()
plt.show()
