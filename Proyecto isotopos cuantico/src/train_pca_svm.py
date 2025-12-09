import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load PCA dataset
df = pd.read_csv("../data/dataset_pca.csv")

X = df[["PC1", "PC2"]]
y = df["isotope"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Build model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("\n=== SVM (PCA) — CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== SVM (PCA) — CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))
