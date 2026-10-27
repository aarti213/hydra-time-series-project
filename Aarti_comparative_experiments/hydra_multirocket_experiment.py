from pathlib import Path
from time import perf_counter
import importlib.util
import sys
import numpy as np
import pandas as pd
import torch
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

DATASET_NAMES = ["Coffee", "ECG200", "GunPoint", "ItalyPowerDemand", "TwoLeadECG", "Wafer"]
RANDOM_STATE = 42
BATCH_SIZE = 256
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HYDRA_SOURCE = PROJECT_ROOT / "hydra-authors-repo" / "code" / "hydra.py"
RESULTS_DIR = PROJECT_ROOT / "results" / "Aarti"
RESULTS_FILE = RESULTS_DIR / "hydra_multirocket_results.csv"

if not HYDRA_SOURCE.exists():
    raise FileNotFoundError(f"Could not find Hydra implementation: {HYDRA_SOURCE}")
spec = importlib.util.spec_from_file_location("hydra_official_impl", HYDRA_SOURCE)
hydra_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hydra_module
spec.loader.exec_module(hydra_module)
Hydra = hydra_module.Hydra
SparseScaler = hydra_module.SparseScaler

def run_dataset(dataset_name):
    print(f"\n{'=' * 60}\nHydra + MultiRocket | {dataset_name}\n{'=' * 60}")
    X_train, y_train = load_classification(dataset_name, split="train", return_metadata=False)
    X_test, y_test = load_classification(dataset_name, split="test", return_metadata=False)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).astype(str)
    y_test = np.asarray(y_test).astype(str)
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Expected arrays shaped (cases, channels, timepoints).")
    if X_train.shape[1] != 1 or X_test.shape[1] != 1:
        raise ValueError(f"{dataset_name}: this script supports univariate datasets only.")
    if X_train.shape[2] != X_test.shape[2]:
        raise ValueError(f"{dataset_name}: train/test series lengths differ.")

    X_train_tensor = torch.as_tensor(np.ascontiguousarray(X_train), dtype=torch.float32)
    X_test_tensor = torch.as_tensor(np.ascontiguousarray(X_test), dtype=torch.float32)
    hydra = Hydra(input_length=X_train.shape[-1], k=8, g=64, seed=RANDOM_STATE)

    start = perf_counter()
    with torch.inference_mode():
        H_train_tensor = hydra.batch(X_train_tensor, batch_size=BATCH_SIZE)
    hydra_train_time = perf_counter() - start

    start = perf_counter()
    with torch.inference_mode():
        H_test_tensor = hydra.batch(X_test_tensor, batch_size=BATCH_SIZE)
    hydra_test_time = perf_counter() - start

    scaler = SparseScaler()
    H_train_tensor = scaler.fit_transform(H_train_tensor)
    H_test_tensor = scaler.transform(H_test_tensor)
    H_train = H_train_tensor.detach().cpu().numpy()
    H_test = H_test_tensor.detach().cpu().numpy()

    mr = MultiRocket(n_kernels=6250, max_dilations_per_kernel=32,
                     n_features_per_kernel=4, normalise=False,
                     n_jobs=1, random_state=RANDOM_STATE)
    start = perf_counter()
    M_train = np.asarray(mr.fit_transform(X_train), dtype=np.float32)
    mr_train_time = perf_counter() - start
    start = perf_counter()
    M_test = np.asarray(mr.transform(X_test), dtype=np.float32)
    mr_test_time = perf_counter() - start

    Z_train = np.concatenate([H_train, M_train], axis=1)
    Z_test = np.concatenate([H_test, M_test], axis=1)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    start = perf_counter()
    classifier.fit(Z_train, y_train)
    classifier_fit = perf_counter() - start
    start = perf_counter()
    y_pred = classifier.predict(Z_test)
    prediction = perf_counter() - start

    test_transform = hydra_test_time + mr_test_time
    row = {
        "dataset": dataset_name, "model": "Hydra+MultiRocket", "seed": RANDOM_STATE,
        "n_train": len(y_train), "n_test": len(y_test),
        "hydra_features": H_train.shape[1], "multirocket_features": M_train.shape[1],
        "combined_features": Z_train.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "hydra_train_transform_seconds": hydra_train_time,
        "multirocket_train_transform_seconds": mr_train_time,
        "classifier_fit_seconds": classifier_fit,
        "total_training_seconds": hydra_train_time + mr_train_time + classifier_fit,
        "hydra_test_transform_seconds": hydra_test_time,
        "multirocket_test_transform_seconds": mr_test_time,
        "classifier_prediction_seconds": prediction,
        "end_to_end_test_seconds": test_transform + prediction,
        "prediction_seconds_per_case": prediction / len(y_test),
    }
    print(f"Accuracy={row['accuracy']:.4f} | Macro F1={row['macro_f1']:.4f} | "
          f"Train={row['total_training_seconds']:.3f}s | "
          f"End-to-end test={row['end_to_end_test_seconds']:.3f}s")
    return row

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASET_NAMES:
        try:
            rows.append(run_dataset(name))
        except Exception as exc:
            print(f"ERROR on {name}: {type(exc).__name__}: {exc}")
            raise
    new = pd.DataFrame(rows)
    if RESULTS_FILE.exists():
        old = pd.read_csv(RESULTS_FILE)
        old = old[~((old["dataset"].isin(DATASET_NAMES)) &
                    (old["seed"] == RANDOM_STATE) &
                    (old["model"] == "Hydra+MultiRocket"))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {len(rows)} dataset results to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
