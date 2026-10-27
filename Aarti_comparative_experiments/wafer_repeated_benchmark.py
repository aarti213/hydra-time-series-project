from pathlib import Path
from time import perf_counter
import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

DATASET = "Wafer"
RANDOM_STATE = 42
N_REPEATS = 5
BATCH_SIZE = 256
ALPHAS = np.logspace(-3, 3, 10)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HYDRA_SOURCE = PROJECT_ROOT / "hydra-authors-repo" / "code" / "hydra.py"
OUTPUT_DIR = PROJECT_ROOT / "results" / "Aarti"
RAW_OUTPUT = OUTPUT_DIR / "wafer_repeated_benchmark_runs.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "wafer_repeated_benchmark_summary.csv"


def load_hydra_classes():
    if not HYDRA_SOURCE.exists():
        raise FileNotFoundError(f"Could not find Hydra implementation: {HYDRA_SOURCE}")
    spec = importlib.util.spec_from_file_location("hydra_official_benchmark_impl", HYDRA_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Hydra, module.SparseScaler


def load_data():
    X_train, y_train = load_classification(DATASET, split="train", return_metadata=False)
    X_test, y_test = load_classification(DATASET, split="test", return_metadata=False)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).astype(str)
    y_test = np.asarray(y_test).astype(str)
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Expected arrays shaped (cases, channels, timepoints).")
    if X_train.shape[1] != 1 or X_test.shape[1] != 1:
        raise ValueError("This script supports univariate datasets only.")
    return X_train, y_train, X_test, y_test


def fit_and_score(model_name, run_number, X_train, y_train, X_test, y_test,
                  Hydra=None, SparseScaler=None):
    row = {
        "dataset": DATASET, "model": model_name, "run_number": run_number,
        "random_state": RANDOM_STATE, "n_train": len(y_train), "n_test": len(y_test),
    }

    if model_name == "MiniRocket":
        transformer = MiniRocket(
            n_kernels=10000, max_dilations_per_kernel=32,
            n_jobs=1, random_state=RANDOM_STATE,
        )
        start = perf_counter()
        X_train_features = transformer.fit_transform(X_train)
        train_transform = perf_counter() - start
        start = perf_counter()
        X_test_features = transformer.transform(X_test)
        test_transform = perf_counter() - start
        row.update({
            "hydra_train_transform_seconds": 0.0,
            "rocket_train_transform_seconds": train_transform,
            "hydra_test_transform_seconds": 0.0,
            "rocket_test_transform_seconds": test_transform,
            "n_features": X_train_features.shape[1],
        })

    elif model_name == "MultiRocket":
        transformer = MultiRocket(
            n_kernels=6250, max_dilations_per_kernel=32,
            n_features_per_kernel=4, normalise=False,
            n_jobs=1, random_state=RANDOM_STATE,
        )
        start = perf_counter()
        X_train_features = transformer.fit_transform(X_train)
        train_transform = perf_counter() - start
        start = perf_counter()
        X_test_features = transformer.transform(X_test)
        test_transform = perf_counter() - start
        row.update({
            "hydra_train_transform_seconds": 0.0,
            "rocket_train_transform_seconds": train_transform,
            "hydra_test_transform_seconds": 0.0,
            "rocket_test_transform_seconds": test_transform,
            "n_features": X_train_features.shape[1],
        })

    elif model_name == "Hydra+MultiRocket":
        if Hydra is None or SparseScaler is None:
            raise ValueError("Hydra and SparseScaler classes are required.")
        X_train_tensor = torch.as_tensor(np.ascontiguousarray(X_train), dtype=torch.float32)
        X_test_tensor = torch.as_tensor(np.ascontiguousarray(X_test), dtype=torch.float32)
        hydra = Hydra(input_length=X_train.shape[-1], k=8, g=64, seed=RANDOM_STATE)

        start = perf_counter()
        with torch.inference_mode():
            H_train_tensor = hydra.batch(X_train_tensor, batch_size=BATCH_SIZE)
        hydra_train_transform = perf_counter() - start
        start = perf_counter()
        with torch.inference_mode():
            H_test_tensor = hydra.batch(X_test_tensor, batch_size=BATCH_SIZE)
        hydra_test_transform = perf_counter() - start

        scaler = SparseScaler()
        H_train_tensor = scaler.fit_transform(H_train_tensor)
        H_test_tensor = scaler.transform(H_test_tensor)
        H_train = H_train_tensor.detach().cpu().numpy()
        H_test = H_test_tensor.detach().cpu().numpy()

        transformer = MultiRocket(
            n_kernels=6250, max_dilations_per_kernel=32,
            n_features_per_kernel=4, normalise=False,
            n_jobs=1, random_state=RANDOM_STATE,
        )
        start = perf_counter()
        M_train = np.asarray(transformer.fit_transform(X_train), dtype=np.float32)
        rocket_train_transform = perf_counter() - start
        start = perf_counter()
        M_test = np.asarray(transformer.transform(X_test), dtype=np.float32)
        rocket_test_transform = perf_counter() - start

        X_train_features = np.concatenate([H_train, M_train], axis=1)
        X_test_features = np.concatenate([H_test, M_test], axis=1)
        row.update({
            "hydra_train_transform_seconds": hydra_train_transform,
            "rocket_train_transform_seconds": rocket_train_transform,
            "hydra_test_transform_seconds": hydra_test_transform,
            "rocket_test_transform_seconds": rocket_test_transform,
            "n_features": X_train_features.shape[1],
        })
    else:
        raise ValueError(f"Unknown model: {model_name}")

    classifier = RidgeClassifierCV(alphas=ALPHAS)
    start = perf_counter()
    classifier.fit(X_train_features, y_train)
    classifier_fit = perf_counter() - start
    start = perf_counter()
    predictions = classifier.predict(X_test_features)
    prediction_time = perf_counter() - start

    row.update({
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "classifier_fit_seconds": classifier_fit,
        "total_training_seconds": row["hydra_train_transform_seconds"]
            + row["rocket_train_transform_seconds"] + classifier_fit,
        "classifier_prediction_seconds": prediction_time,
        "end_to_end_test_seconds": row["hydra_test_transform_seconds"]
            + row["rocket_test_transform_seconds"] + prediction_time,
        "prediction_seconds_per_case": prediction_time / len(y_test),
    })
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test = load_data()
    Hydra, SparseScaler = load_hydra_classes()

    existing = pd.read_csv(RAW_OUTPUT) if RAW_OUTPUT.exists() else pd.DataFrame()
    start_run = int(existing["run_number"].max()) + 1 if not existing.empty else 1
    rows = []
    models = ["MiniRocket", "MultiRocket", "Hydra+MultiRocket"]

    print(f"Dataset: {DATASET}; train={len(y_train)}; test={len(y_test)}")
    print(f"Repeats this session: {N_REPEATS}")
    print("Existing model result CSVs will not be modified.")

    for run_number in range(start_run, start_run + N_REPEATS):
        for model_name in models:
            print(f"\nRun {run_number} | {model_name}")
            row = fit_and_score(
                model_name, run_number, X_train, y_train, X_test, y_test,
                Hydra=Hydra, SparseScaler=SparseScaler,
            )
            rows.append(row)
            print(
                f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}, "
                f"train={row['total_training_seconds']:.3f}s, "
                f"end-to-end test={row['end_to_end_test_seconds']:.3f}s"
            )

    new_runs = pd.DataFrame(rows)
    combined_runs = pd.concat([existing, new_runs], ignore_index=True)
    combined_runs.to_csv(RAW_OUTPUT, index=False)

    metrics = [
        "accuracy", "macro_f1", "total_training_seconds", "end_to_end_test_seconds",
        "hydra_train_transform_seconds", "rocket_train_transform_seconds",
        "classifier_fit_seconds", "hydra_test_transform_seconds",
        "rocket_test_transform_seconds", "classifier_prediction_seconds",
    ]
    summary = combined_runs.groupby(["dataset", "model"])[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print(f"\nRaw runs: {RAW_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
