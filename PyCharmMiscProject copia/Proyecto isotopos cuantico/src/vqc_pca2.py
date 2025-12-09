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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings('ignore')

CONFIG = {
    'n_qubits': 4,
    'feature_map_reps': 3,  # Aumentado de 2 a 3
    'ansatz_reps': 5,  # Aumentado de 3 a 5
    'learning_rate': 0.005,  # Reducido para mejor convergencia
    'epochs': 80,  # Aumentado de 60 a 100
    'batch_size': 16,  # Reducido para mejor gradientes
    'weight_decay': 5e-5,  # Reducido
    'dropout': 0.2,  # Reducido de 0.3
    'patience': 25,  # Aumentado de 15
    'test_size': 0.2,
    'random_state': 42,
    'n_restarts': 1,  # Múltiples inicializaciones
    'min_lr': 1e-6,  # Para learning rate scheduler
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

    # Feature engineering expandido
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

    # Normalización mejorada
    X_min = X_expanded.min(axis=0)
    X_max = X_expanded.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_normalized = (X_expanded - X_min) / X_range * np.pi

    return X_normalized, y_enc, le, scaler, n_classes


def create_quantum_circuit(n_qubits, feature_map_reps, ansatz_reps):
    # Feature map más expresivo
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=feature_map_reps,
        entanglement='full',  # Cambiado de 'circular' a 'full'
        insert_barriers=False
    )

    # Ansatz más expresivo
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=ansatz_reps,
        entanglement='full',  # Cambiado de 'linear' a 'full'
        insert_barriers=False,
        skip_final_rotation_layer=False
    )

    qc = feature_map.compose(ansatz)

    return qc, feature_map.parameters, ansatz.parameters


class HybridQNNClassifier(nn.Module):
    def __init__(self, qnn_layer, n_qubits, n_classes, dropout=0.2):
        super().__init__()
        self.qnn = qnn_layer

        # Classical head más grande y profundo
        self.bn1 = nn.BatchNorm1d(n_qubits)
        self.fc1 = nn.Linear(n_qubits, 64)  # Aumentado de 32 a 64
        self.dropout1 = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)  # Aumentado de 16 a 32
        self.dropout2 = nn.Dropout(dropout)

        self.bn3 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)  # Nueva capa adicional
        self.dropout3 = nn.Dropout(dropout / 2)

        self.fc4 = nn.Linear(16, n_classes)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        q_out = self.qnn(x)

        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(0)

        x = self.bn1(q_out)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.bn2(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.bn3(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)

        return x


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
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


def train_single_model(X_train_t, y_train_t, X_test_t, y_test_t, n_classes, restart_idx):
    """Entrena un modelo con una inicialización aleatoria diferente"""
    print(f"\n{'=' * 70}")
    print(f"RESTART {restart_idx + 1}/{CONFIG['n_restarts']}")
    print(f"{'=' * 70}")

    # Construir circuito cuántico
    print("\n[1/4] Construyendo circuito cuántico...")
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

    # Inicializar QNN
    print("\n[2/4] Inicializando QNN...")
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
    print("   ✓ QNN inicializado")

    # Construir modelo
    print("\n[3/4] Construyendo modelo híbrido...")
    torch.manual_seed(CONFIG['random_state'] + restart_idx)  # Seed diferente por restart
    model = HybridQNNClassifier(
        quantum_layer,
        CONFIG['n_qubits'],
        n_classes,
        dropout=CONFIG['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Parámetros totales: {total_params}")

    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Scheduler con warmup y cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=CONFIG['min_lr']
    )

    train_loader, val_loader = create_dataloaders(
        X_train_t, y_train_t,
        X_test_t, y_test_t,
        CONFIG['batch_size']
    )

    # Entrenar
    print("\n[4/4] Entrenando...")
    print("-" * 70)

    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠ Early stopping en época {epoch + 1}")
            break

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n✓ Mejor accuracy de validación: {best_val_acc:.2f}%")

    return model, best_val_acc


def main():
    print("=" * 70)
    print("VQC MULTICLASS OPTIMIZADO - CLASIFICACIÓN DE ISÓTOPOS")
    print("CON MÚLTIPLES REINICIOS Y ARQUITECTURA MEJORADA")
    print("=" * 70)

    # Cargar datos
    print("\n[FASE 1] Cargando y preprocesando datos...")
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

    # Entrenar múltiples modelos con diferentes inicializaciones
    print(f"\n[FASE 2] Entrenando {CONFIG['n_restarts']} modelos con diferentes inicializaciones...")

    models = []
    val_accs = []

    for i in range(CONFIG['n_restarts']):
        model, val_acc = train_single_model(
            X_train_t, y_train_t, X_test_t, y_test_t, n_classes, i
        )
        models.append(model)
        val_accs.append(val_acc)

    # Seleccionar mejor modelo
    best_idx = np.argmax(val_accs)
    best_model = models[best_idx]

    print(f"\n{'=' * 70}")
    print("RESUMEN DE ENTRENAMIENTOS")
    print(f"{'=' * 70}")
    for i, acc in enumerate(val_accs):
        marker = " ← MEJOR" if i == best_idx else ""
        print(f"Restart {i + 1}: {acc:.2f}%{marker}")

    # Evaluación final
    print(f"\n{'=' * 70}")
    print("EVALUACIÓN FINAL CON MEJOR MODELO")
    print(f"{'=' * 70}")

    best_model.eval()
    with torch.no_grad():
        outputs = best_model(X_test_t)
        _, y_pred = outputs.max(1)
        y_pred_np = y_pred.numpy()

    test_acc = accuracy_score(y_test, y_pred_np)

    print(f"\nAccuracy Final en Test: {test_acc * 100:.2f}%")
    print(f"Mejor Accuracy en Validación: {val_accs[best_idx]:.2f}%")

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
        class_acc = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
        print(f"{isotope:10s}: {class_acc:.2f}% ({cm[i, i]}/{cm[i].sum()} correctos)")

    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()