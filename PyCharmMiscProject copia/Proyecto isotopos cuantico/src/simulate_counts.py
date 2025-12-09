
import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_SAMPLES_PER_CONFIG = 200   # muestras por combinación (isótopo x distancia)
MEASUREMENT_TIME = 30.0      # segundos de medida por muestra (tiempo de conteo)
THICKNESS_CM = 0.2           # 2 mm = 0.2 cm, grosor común para los 3 filtros
DISTANCES_CM = [10.0, 20.0, 30.0]

# Isótopos: (lista de energias en keV, intensidades relativas)
ISOTOPES = {
    "Co60": ([1173.2, 1332.5], [1.0, 1.0]),
    "Cs137": ([662.0], [1.0]),
    "Na22": ([511.0, 1274.0], [1.0, 1.0]),
    "Cd109": ([22.6], [1.0])   # rayos x/low-energy gamma
}

# Intensidad de la fuente base (escala arbitraria que produce cps realistas a 10 cm sin filtro)
# Estos son ajustables pero se eligieron para que sean físicamente plausibles para una pequeña fuente de laboratorio a 10 cm
BASE_STRENGTH_AT_10CM = {
    "Co60": 1500.0,
    "Cs137": 800.0,
    "Na22": 1000.0,
    "Cd109": 150.0
}

# Background (counts/s)
BASE_BACKGROUND_CPS = 20.0

# Densidades / mu(E) 
def mu_aluminum(E_keV):
    # aluminio: baja atenuacion relativa
    return 0.12 + 0.00006 * (1500.0 - E_keV)  # 1/cm (approx)

def mu_copper(E_keV):
    # cobre: mayor atenuacion que el aluminio
    return 0.9 + 0.0009 * np.sqrt(E_keV + 1)

def mu_lead(E_keV):
    # plomo: atenuacion grande, especialmente para bajas energias
    return 3.0 + 0.0025 * (1500.0 - E_keV) ** 0.35

MU_FUNCTIONS = {"Al": mu_aluminum, "Cu": mu_copper, "Pb": mu_lead}
FILTERS = ["Al", "Cu", "Pb"]

# Eficiencia del detector vs energía (muy aproximada para un GM pequeño + volumen sensible pequeño)
def detector_efficiency(E_keV):
    if E_keV < 50:
        return 0.05
    if E_keV < 300:
        return 0.15
    if E_keV < 900:
        return 0.30
    return 0.20


def transmission_through_filter(energies, intensities, filter_name, thickness_cm):
    """Return weighted transmission (0..1) for a multi-line source through a filter."""
    mu_fn = MU_FUNCTIONS[filter_name]
    trans = []
    for E in energies:
        mu = mu_fn(E)  # 1/cm
        t = np.exp(-mu * thickness_cm)
        trans.append(t)
    # peso por intensidades de línea y eficiencia del detector
    weights = np.array(intensities, dtype=float)
    effs = np.array([detector_efficiency(E) for E in energies])
    weighted_trans = weights * effs * np.array(trans)
    # fracción normalizada de recuentos detectados después del filtro en relación con los recuentos detectados sin filtro
    denom = np.sum(weights * effs)
    if denom <= 0:
        return 0.0
    return np.sum(weighted_trans) / denom

def counts_at_distance_no_filter(isotope, distance_cm):
    """Estimate expected mean counts (lambda) for measurement time at given distance without filter."""
    base_at_10 = BASE_STRENGTH_AT_10CM[isotope]
    scale = (10.0 / distance_cm) ** 2
    mean_cps = base_at_10 * scale
    return mean_cps

def simulate_one_measurement(isotope, distance_cm, filter_name=None, thickness_cm=THICKNESS_CM):
    energies, intensities = ISOTOPES[isotope]
    mean_cps_free = counts_at_distance_no_filter(isotope, distance_cm)

    mean_cps_free *= np.random.normal(1.0, 0.03)  # 3% source strength jitter

    if filter_name is None:
        trans_frac = 1.0
    else:
        trans_frac = transmission_through_filter(energies, intensities, filter_name, thickness_cm)
    if filter_name is None:
        effs = np.array([detector_efficiency(E) for E in energies])
        weights = np.array(intensities)
        detection_fraction = np.sum(weights * effs) / np.sum(weights)
        mean_cps_detected = mean_cps_free * detection_fraction
    else:
        effs = np.array([detector_efficiency(E) for E in energies])
        weights = np.array(intensities)
        detection_fraction = np.sum(weights * effs) / np.sum(weights)
        mean_cps_detected = mean_cps_free * detection_fraction * trans_frac

    mean_cps_detected += BASE_BACKGROUND_CPS

    mean_counts_in_time = max(0.0, mean_cps_detected * MEASUREMENT_TIME)
    observed_counts = np.random.poisson(mean_counts_in_time)

    return {
        "isotope": isotope,
        "distance_cm": distance_cm,
        "filter": filter_name if filter_name is not None else "none",
        "mean_cps_model": mean_cps_detected,
        "counts": int(observed_counts)
    }

def generate_dataset(n_per_config=N_SAMPLES_PER_CONFIG, out_csv="data/dataset_counts.csv"):
    rows = []

    for isotope in ISOTOPES.keys():
        for d in DISTANCES_CM:
            for _ in range(n_per_config):
                meas_free = simulate_one_measurement(isotope, d, filter_name=None)
                meas_al = simulate_one_measurement(isotope, d, filter_name="Al")
                meas_cu = simulate_one_measurement(isotope, d, filter_name="Cu")
                meas_pb = simulate_one_measurement(isotope, d, filter_name="Pb")

                row = {
                    "isotope": isotope,
                    "distance_cm": d,
                    "counts_free": meas_free["counts"],
                    "mean_cps_free_model": meas_free["mean_cps_model"],
                    "counts_Al": meas_al["counts"],
                    "mean_cps_Al_model": meas_al["mean_cps_model"],
                    "counts_Cu": meas_cu["counts"],
                    "mean_cps_Cu_model": meas_cu["mean_cps_model"],
                    "counts_Pb": meas_pb["counts"],
                    "mean_cps_Pb_model": meas_pb["mean_cps_model"]
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Dataset generado: {out_csv}  —  filas: {df.shape[0]}")
    return df

if __name__ == "__main__":
    generate_dataset()
