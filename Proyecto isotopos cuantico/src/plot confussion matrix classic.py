import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matriz de confusión (modelo clásico)
cm = np.array([
    [150, 0,   0,  0],
    [0,   101, 25, 24],
    [0,   24,  95, 31],
    [0,   49,  40, 61]
])

labels = ["Cd109", "Co60", "Cs137", "Na22"]

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – Modelo Clásico")
plt.tight_layout()
plt.show()
