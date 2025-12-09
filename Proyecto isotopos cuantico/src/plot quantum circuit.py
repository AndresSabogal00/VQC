import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.visualization import circuit_drawer

# Configuración del circuito
CONFIG = {
    'n_qubits': 4,
    'feature_map_reps': 2,
    'ansatz_reps': 3,
}


def create_quantum_circuit(n_qubits, feature_map_reps, ansatz_reps):
    """Crea el circuito cuántico con feature map y ansatz"""

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

    # Composición del circuito completo
    qc = feature_map.compose(ansatz)

    return qc, feature_map, ansatz


# Crear el circuito
print("Creando circuito cuántico...")
qc, feature_map, ansatz = create_quantum_circuit(
    CONFIG['n_qubits'],
    CONFIG['feature_map_reps'],
    CONFIG['ansatz_reps']
)

# Información del circuito
print(f"\n{'=' * 70}")
print("INFORMACIÓN DEL CIRCUITO CUÁNTICO")
print(f"{'=' * 70}")
print(f"Número de qubits: {qc.num_qubits}")
print(f"Profundidad: {qc.depth()}")
print(f"Número de parámetros: {qc.num_parameters}")
print(f"Operaciones totales: {len(qc.data)}")
print(f"{'=' * 70}\n")

# ============================================================================
# VISUALIZACIÓN 1: Circuito DESCOMPUESTO (muestra todas las puertas)
# ============================================================================
print("Generando visualización DESCOMPUESTA (todas las puertas visibles)...")

# Descomponer el circuito para mostrar todas las operaciones individuales
qc_decomposed = qc.decompose().decompose()

fig1 = plt.figure(figsize=(20, 8))
circuit_drawer(
    qc_decomposed,
    output='mpl',
    style={
        'backgroundcolor': '#FFFFFF',
        'textcolor': '#000000',
        'gatetextcolor': '#000000',
        'linecolor': '#000000',
        'creglinecolor': '#000000',
        'gatefacecolor': '#BB8FCE',
        'barrierfacecolor': '#DDDDDD'
    },
    fold=-1,  # Sin doblar
    scale=0.6,
    plot_barriers=False
)
plt.savefig('circuito_descompuesto.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Guardado: circuito_descompuesto.png (TODAS LAS PUERTAS VISIBLES)")
plt.close()

# ============================================================================
# VISUALIZACIÓN 2: Circuito con estilo IQP mejorado
# ============================================================================
print("Generando visualización con estilo mejorado...")

fig2 = plt.figure(figsize=(24, 8))
circuit_drawer(
    qc_decomposed,
    output='mpl',
    style='iqp',
    fold=25,  # Doblar cada 25 operaciones para mejor legibilidad
    scale=0.8,
    plot_barriers=False
)
plt.savefig('circuito_estilo_iqp.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: circuito_estilo_iqp.png")
plt.close()

# ============================================================================
# VISUALIZACIÓN 3: Feature Map descompuesto
# ============================================================================
print("Generando visualización del Feature Map descompuesto...")

feature_map_decomposed = feature_map.decompose().decompose()

fig3 = plt.figure(figsize=(12, 6))
circuit_drawer(
    feature_map_decomposed,
    output='mpl',
    style={
        'backgroundcolor': '#FFFFFF',
        'gatefacecolor': '#85C1E9'
    },
    fold=-1,
    scale=0.8,
    plot_barriers=False
)
plt.savefig('feature_map_detallado.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Guardado: feature_map_detallado.png")
plt.close()

# ============================================================================
# VISUALIZACIÓN 4: Ansatz descompuesto
# ============================================================================
print("Generando visualización del Ansatz descompuesto...")

ansatz_decomposed = ansatz.decompose().decompose()

fig4 = plt.figure(figsize=(16, 6))
circuit_drawer(
    ansatz_decomposed,
    output='mpl',
    style={
        'backgroundcolor': '#FFFFFF',
        'gatefacecolor': '#F8BBD0'
    },
    fold=-1,
    scale=0.7,
    plot_barriers=False
)
plt.savefig('ansatz_detallado.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Guardado: ansatz_detallado.png")
plt.close()

# ============================================================================
# VISUALIZACIÓN 5: Representación en texto (ASCII)
# ============================================================================
print("\nGenerando visualización en texto ASCII...")
circuit_text = qc_decomposed.draw(output='text', fold=100)
print("\n" + "=" * 70)
print("CIRCUITO EN FORMATO TEXTO:")
print("=" * 70)
print(circuit_text)

with open('circuito_texto.txt', 'w', encoding='utf-8') as f:
    f.write(str(circuit_text))
print("\n✓ Guardado: circuito_texto.txt")

# ============================================================================
# ESTADÍSTICAS DETALLADAS
# ============================================================================
print(f"\n{'=' * 70}")
print("ESTADÍSTICAS DETALLADAS DEL CIRCUITO")
print(f"{'=' * 70}")

# Contar tipos de puertas en el circuito descompuesto
gate_counts = {}
for instruction in qc_decomposed.data:
    gate_name = instruction.operation.name
    gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

print("\nDistribución de puertas (circuito descompuesto):")
for gate, count in sorted(gate_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {gate:15s}: {count:4d} operaciones")

print(f"\nProfundidad del circuito descompuesto: {qc_decomposed.depth()}")
print(f"Operaciones totales: {len(qc_decomposed.data)}")

# Análisis por qubit
print(f"\n{'=' * 70}")
print("OPERACIONES POR QUBIT")
print(f"{'=' * 70}")

for qubit_idx in range(qc.num_qubits):
    ops_on_qubit = sum(1 for inst in qc_decomposed.data
                       if qubit_idx in [q.index for q in inst.qubits])
    print(f"  Qubit {qubit_idx}: {ops_on_qubit} operaciones")

print(f"\n{'=' * 70}")
print("✅ VISUALIZACIONES GENERADAS EXITOSAMENTE")
print(f"{'=' * 70}")
print("\nArchivos creados:")
print("  1. circuito_descompuesto.png    - ⭐ RECOMENDADO: Todas las puertas visibles")
print("  2. circuito_estilo_iqp.png      - Estilo profesional con fold")
print("  3. feature_map_detallado.png    - Feature Map con todas las puertas")
print("  4. ansatz_detallado.png         - Ansatz con todas las puertas")
print("  5. circuito_texto.txt           - Representación ASCII")
print("\n💡 IMPORTANTE: 'circuito_descompuesto.png' muestra TODAS las puertas")
print("   individuales (H, RZ, RY, CX, etc.) aplicadas en cada qubit.")