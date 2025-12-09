# src/simulate_counts.py
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ---------- configuración ----------
N_SAMPLES_PER_CONFIG = 200   # muestras por combinación (isótopo x distancia)
MEASUREMENT_TIME = 30.0      # segundos de medida por muestra (tiempo de conteo)
THICKNESS_CM = 0.2           # 2 mm = 0.2 cm, grosor común para los 3 filtros
DISTANCES_CM = [10.0, 20.0, 30.0]

# Isótopos: (list of energies keV, relative intensities)
ISOTOPES = {
    "Co60": ([1173.2, 1332.5], [1.0, 1.0]),
    "Cs137": ([662.0], [1.0]),
    "Na22": ([511.0, 1274.0], [1.0, 1.0]),
    "Cd109": ([22.6], [1.0])   # rayos x/low-energy gamma
}

# Base source strength (arbitrary scale that produces realistic cps at 10 cm without filter)
# These are tunable but chosen to be physically plausible for a small laboratory source at 10 cm
BASE_STRENGTH_AT_10CM = {
    "Co60": 1500.0,
    "Cs137": 800.0,
    "Na22": 1000.0,
    "Cd109": 150.0
}

# Background (counts/s) typical baseline of environment and detector noise
BASE_BACKGROUND_CPS = 20.0

# Densities / mu(E) approximations (mass attenuation behaviour simplified)
def mu_aluminum(E_keV):
    # aluminium: relatively low attenuation, weak E dependence
    return 0.12 + 0.00006 * (1500.0 - E_keV)  # 1/cm (approx)

def mu_copper(E_keV):
    # copper: stronger attenuation than Al
    return 0.9 + 0.0009 * np.sqrt(E_keV + 1)

def mu_lead(E_keV):
    # lead: strong attenuation, especially for lower energies
    return 3.0 + 0.0025 * (1500.0 - E_keV) ** 0.35

MU_FUNCTIONS = {"Al": mu_aluminum, "Cu": mu_copper, "Pb": mu_lead}
FILTERS = ["Al", "Cu", "Pb"]

# Detector efficiency vs energy (very approximate for a small GM + small sensitive volume)
def detector_efficiency(E_keV):
    # Geiger tubes are not energy resolving and have a complex efficiency vs E.
    # Use a plausible energy-dependent efficiency shape: low at very low energies (window losses),
    # moderate in mid-energies, and slightly lower at high energies.
    if E_keV < 50:
        return 0.05
    if E_keV < 300:
        return 0.15
    if E_keV < 900:
        return 0.30
    return 0.20

# ---------- funciones físicas ----------
def transmission_through_filter(energies, intensities, filter_name, thickness_cm):
    """Return weighted transmission (0..1) for a multi-line source through a filter."""
    mu_fn = MU_FUNCTIONS[filter_name]
    trans = []
    for E in energies:
        mu = mu_fn(E)  # 1/cm
        t = np.exp(-mu * thickness_cm)
        trans.append(t)
    # weight by line intensities and detector efficiency
    weights = np.array(intensities, dtype=float)
    effs = np.array([detector_efficiency(E) for E in energies])
    weighted_trans = weights * effs * np.array(trans)
    # normalized fraction of detected counts after filter relative to detected counts without filter
    denom = np.sum(weights * effs)
    if denom <= 0:
        return 0.0
    return np.sum(weighted_trans) / denom

def counts_at_distance_no_filter(isotope, distance_cm):
    """Estimate expected mean counts (lambda) for measurement time at given distance without filter."""
    base_at_10 = BASE_STRENGTH_AT_10CM[isotope]
    # inverse square scaling (normalized so base_at_10cm corresponds to distance=10 cm)
    scale = (10.0 / distance_cm) ** 2
    mean_cps = base_at_10 * scale
    # add background (assumed independent)
    return mean_cps

# ---------- simulador de una medida ----------
def simulate_one_measurement(isotope, distance_cm, filter_name=None, thickness_cm=THICKNESS_CM):
    energies, intensities = ISOTOPES[isotope]
    # mean counts per second w/o filter
    mean_cps_free = counts_at_distance_no_filter(isotope, distance_cm)

    # apply small random fluctuation in source intensity (real sources vary)
    mean_cps_free *= np.random.normal(1.0, 0.03)  # 3% source strength jitter

    # fraction transmitted by filter (if any)
    if filter_name is None:
        trans_frac = 1.0
    else:
        trans_frac = transmission_through_filter(energies, intensities, filter_name, thickness_cm)

    # detector efficiency integrated effect already in transmission calculation for filtered path;
    # for free case, approximate detection fraction by weighting efficiencies
    if filter_name is None:
        effs = np.array([detector_efficiency(E) for E in energies])
        weights = np.array(intensities)
        detection_fraction = np.sum(weights * effs) / np.sum(weights)
        # apply that to mean cps (so base strength approximates source emission, not detected counts)
        mean_cps_detected = mean_cps_free * detection_fraction
    else:
        # when using transmission_through_filter we already folded eff into that call, so:
        # compute the free-detected rate first, then multiply by transmission fraction
        effs = np.array([detector_efficiency(E) for E in energies])
        weights = np.array(intensities)
        detection_fraction = np.sum(weights * effs) / np.sum(weights)
        mean_cps_detected = mean_cps_free * detection_fraction * trans_frac

    # add background (this is cps baseline)
    mean_cps_detected += BASE_BACKGROUND_CPS

    # measurement: scale by time and apply Poisson noise
    mean_counts_in_time = max(0.0, mean_cps_detected * MEASUREMENT_TIME)
    observed_counts = np.random.poisson(mean_counts_in_time)

    return {
        "isotope": isotope,
        "distance_cm": distance_cm,
        "filter": filter_name if filter_name is not None else "none",
        "mean_cps_model": mean_cps_detected,
        "counts": int(observed_counts)
    }

# ---------- generar dataset completo ----------
def generate_dataset(n_per_config=N_SAMPLES_PER_CONFIG, out_csv="data/dataset_counts.csv"):
    rows = []

    for isotope in ISOTOPES.keys():
        for d in DISTANCES_CM:
            for _ in range(n_per_config):
                # measurement without filter
                meas_free = simulate_one_measurement(isotope, d, filter_name=None)
                # measurements with each filter (Al, Cu, Pb)
                meas_al = simulate_one_measurement(isotope, d, filter_name="Al")
                meas_cu = simulate_one_measurement(isotope, d, filter_name="Cu")
                meas_pb = simulate_one_measurement(isotope, d, filter_name="Pb")

                # Build a single row combining the four readings (one sample)
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
