import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ====== ESTILO PROFESIONAL PARA POSTER ======
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")  # títulos grandes tipo póster

# ====== DATOS EXTRAÍDOS DEL ENTRENAMIENTO ======

train_loss = [1.1766, 0.9825, 0.8989, 0.8881, 0.8088, 0.8150, 0.8189, 0.8137,
              0.8174, 0.8028, 0.8021, 0.8047, 0.7780, 0.8005, 0.8091, 0.8047,
              0.7985, 0.8020, 0.7530, 0.8000, 0.7778, 0.7720, 0.7817, 0.8146,
              0.7939, 0.7834, 0.7896, 0.7681, 0.7765, 0.7776, 0.7505, 0.7567,
              0.7569, 0.7466, 0.7550, 0.7626, 0.7326, 0.7567, 0.7621, 0.7442,
              0.7517, 0.7512, 0.7553, 0.7383, 0.7421, 0.7632, 0.7691, 0.7581]

val_loss = [0.9529, 0.8834, 0.8059, 0.7638, 0.7372, 0.7426, 0.7255, 0.7488,
            0.7277, 0.7362, 0.7229, 0.7387, 0.7232, 0.7183, 0.7084, 0.7211,
            0.7167, 0.7045, 0.7284, 0.6929, 0.7066, 0.6826, 0.7039, 0.7031,
            0.7052, 0.7416, 0.7072, 0.7212, 0.6937, 0.6809, 0.6908, 0.6795,
            0.7033, 0.6760, 0.6835, 0.6837, 0.6753, 0.6889, 0.6985, 0.6851,
            0.6841, 0.6911, 0.6852, 0.6711, 0.6683, 0.6736, 0.6737, 0.6666]

train_acc = [42.40, 50.68, 55.94, 57.97, 59.27, 60.00, 59.01, 59.22, 58.39,
             59.43, 59.38, 59.74, 60.73, 61.61, 60.62, 60.83, 60.57, 60.10,
             62.60, 60.62, 59.69, 60.21, 60.68, 59.79, 59.84, 61.25, 58.65,
             62.81, 61.72, 60.89, 60.36, 62.08, 62.40, 62.29, 62.71, 59.43,
             62.19, 62.55, 61.46, 62.71, 61.46, 62.03, 61.82, 63.59, 62.14,
             62.03, 62.29, 61.09]

val_acc = [54.38, 55.21, 59.79, 61.67, 61.25, 63.12, 65.21, 61.88, 59.38,
           60.62, 64.58, 63.54, 60.83, 63.33, 63.96, 64.17, 61.46, 64.17,
           64.38, 65.42, 63.75, 64.17, 62.08, 63.96, 63.96, 61.46, 63.12,
           61.04, 65.83, 65.21, 64.38, 65.00, 68.33, 65.00, 65.83, 66.04,
           65.62, 64.79, 65.62, 67.50, 65.42, 63.54, 64.58, 64.38, 64.58,
           66.46, 65.00, 65.83]

# CONFUSION MATRIX EXTRAÍDA
cm = np.array([
    [116, 1, 2, 1],
    [0, 91, 23, 6],
    [3, 10, 84, 23],
    [1, 23, 75, 21]
])

isotopes = ["Cd109", "Co60", "Cs137", "Na22"]
epochs = range(1, len(train_loss) + 1)

# =========================================================
# 1. LOSS CURVES
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss, label="Train Loss", linewidth=3)
plt.plot(epochs, val_loss, label="Validation Loss", linewidth=3)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (VQC Hybrid Model)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 2. ACCURACY CURVES
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=3)
plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy (VQC Hybrid Model)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# 3. CONFUSION MATRIX (heatmap estilo paper)
# =========================================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=isotopes, yticklabels=isotopes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Quantum-Classical Hybrid Classifier")
plt.tight_layout()
plt.show()

# =========================================================
# 4. BARPLOT — ACCURACY POR CLASE
# =========================================================
class_acc = cm.diagonal() / cm.sum(axis=1) * 100

plt.figure(figsize=(10, 6))
sns.barplot(x=isotopes, y=class_acc, palette="viridis")
plt.ylabel("Accuracy (%)")
plt.title("Per-Class Accuracy")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

# =========================================================
# 5. PCA SCATTER (si deseas mostrar clusters reales)
# =========================================================

ruta_dataset = "../data/dataset_pca.csv"  # cambia si tu ruta es distinta
df = pd.read_csv(ruta_dataset)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x="PC1", y="PC2",
    hue="isotope",
    palette="viridis",
    s=60,
    alpha=0.8
)

plt.title("PCA Projection of Radiation Spectra")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Isotope")
plt.tight_layout()
plt.show()
