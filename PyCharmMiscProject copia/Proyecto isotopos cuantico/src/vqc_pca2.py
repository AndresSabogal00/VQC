import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')

CONFIG = {
    'n_qubits': 4,
    'feature_map_reps': 2,
    'ansatz_reps': 3,
    'learning_rate': 0.01,
    'epochs': 60,
    'batch_size': 32,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'patience': 15,
    'test_size': 0.2,
    'random_state': 42,
}



def load_and_preprocess_data(filepath, config):
    df = pd.read_csv(filepath)

    X = df[["PC1", "PC2"]].values
    y = df["isotope"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pc1_sq = X_scaled[:, 0] ** 2
    pc2_sq = X_scaled[:, 1] ** 2
    pc_cross = X_scaled[:, 0] * X_scaled[:, 1]
    pc_sum = X_scaled[:, 0] + X_scaled[:, 1]

    X_expanded = np.column_stack([
        X_scaled[:, 0],
        X_scaled[:, 1],
        pc_cross,
        pc_sum
    ])

    X_min = X_expanded.min(axis=0)
    X_max = X_expanded.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_normalized = (X_expanded - X_min) / X_range * np.pi

    return X_normalized, y_enc, le, scaler, n_classes


def create_quantum_circuit(n_qubits, feature_map_reps, ansatz_reps):

    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=feature_map_reps,
        entanglement='circular',
        insert_barriers=False
    )

    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=ansatz_reps,
        entanglement='linear',
        insert_barriers=False,
        skip_final_rotation_layer=False
    )

    qc = feature_map.compose(ansatz)

    return qc, feature_map.parameters, ansatz.parameters


class HybridQNNClassifier(nn.Module):

    def __init__(self, qnn_layer, n_qubits, n_classes, dropout=0.3):
        super().__init__()
        self.qnn = qnn_layer

        # Post-procesamiento clásico multicapa
        self.bn1 = nn.BatchNorm1d(n_qubits)
        self.fc1 = nn.Linear(n_qubits, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(dropout / 2)
        self.fc3 = nn.Linear(16, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        q_out = self.qnn(x)

        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(0)

        x = self.bn1(q_out)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.bn2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    """Crea dataloaders con batching para entrenamiento eficiente"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    return total_loss / len(val_loader), 100. * correct / total


def main():
    print("=" * 70)
    print("VQC MULTICLASS OPTIMIZADO - CLASIFICACIÓN DE ISÓTOPOS")
    print("=" * 70)

    # 1. Cargar y preprocesar datos
    print("\n[1/6] Cargando y preprocesando datos...")
    X, y, le, scaler, n_classes = load_and_preprocess_data(
        "../data/dataset_pca.csv",
        CONFIG
    )
    print(f"   ✓ Datos cargados: {X.shape[0]} muestras, {n_classes} clases")
    print(f"   ✓ Features expandidas: {X.shape[1]} dimensiones")
    print(f"   ✓ Clases: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    print(f"   ✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("\n[2/6] Construyendo circuito cuántico...")
    qc, input_params, weight_params = create_quantum_circuit(
        CONFIG['n_qubits'],
        CONFIG['feature_map_reps'],
        CONFIG['ansatz_reps']
    )

    print(f"   ✓ Qubits: {CONFIG['n_qubits']}")
    print(f"   ✓ Feature map reps: {CONFIG['feature_map_reps']}")
    print(f"   ✓ Ansatz reps: {CONFIG['ansatz_reps']}")
    print(f"   ✓ Parámetros entrenables: {len(weight_params)}")
    print(f"   ✓ Profundidad del circuito: {qc.depth()}")

    print("\n[3/6] Inicializando QNN...")
    estimator = StatevectorEstimator()

    from qiskit.quantum_info import SparsePauliOp
    observables = [SparsePauliOp.from_list([("Z" + "I" * i + "Z" + "I" * (CONFIG['n_qubits'] - i - 2), 1.0)])
                   for i in range(CONFIG['n_qubits'] - 1)]
    observables.append(SparsePauliOp.from_list([("Z" * CONFIG['n_qubits'], 1.0)]))

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=list(input_params),
        weight_params=list(weight_params),
        observables=observables[:CONFIG['n_qubits']],
        estimator=estimator,
    )

    quantum_layer = TorchConnector(qnn)
    print("   ✓ QNN inicializado con éxito")

    print("\n[4/6] Construyendo modelo híbrido cuántico-clásico...")
    model = HybridQNNClassifier(
        quantum_layer,
        CONFIG['n_qubits'],
        n_classes,
        dropout=CONFIG['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Parámetros totales entrenables: {total_params}")

    print("\n[5/6] Configurando entrenamiento...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    train_loader, val_loader = create_dataloaders(
        X_train_t, y_train_t,
        X_test_t, y_test_t,
        CONFIG['batch_size']
    )

    print("\n[6/6] Entrenando modelo...")
    print("-" * 70)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1:3d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠ Early stopping activado en época {epoch + 1}")
            break

    model.load_state_dict(best_model_state)

    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, y_pred = outputs.max(1)
        y_pred_np = y_pred.numpy()

    test_acc = accuracy_score(y_test, y_pred_np)

    print(f"\nAccuracy Final en Test: {test_acc * 100:.2f}%")
    print(f"Mejor Accuracy en Validación: {best_val_acc:.2f}%")

    print("\n" + "-" * 70)
    print("REPORTE DE CLASIFICACIÓN")
    print("-" * 70)
    print(classification_report(
        y_test, y_pred_np,
        target_names=le.classes_,
        digits=4
    ))

    print("\n" + "-" * 70)
    print("MATRIZ DE CONFUSIÓN")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred_np)
    print(cm)

    print("\n" + "-" * 70)
    print("ANÁLISIS POR CLASE")
    print("-" * 70)
    for i, isotope in enumerate(le.classes_):
        class_acc = cm[i, i] / cm[i].sum() * 100
        print(f"{isotope:10s}: {class_acc:.2f}% ({cm[i, i]}/{cm[i].sum()} correctos)")

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
