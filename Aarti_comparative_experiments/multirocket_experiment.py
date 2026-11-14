from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score

DATASET_NAMES = ["Coffee", "ECG200", "GunPoint", "ItalyPowerDemand", "TwoLeadECG", "Wafer"]
RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "Aarti"
RESULTS_FILE = RESULTS_DIR / "multirocket_results.csv"

def run_dataset(dataset_name):
    print(f"\n{'=' * 60}\nMultiRocket | {dataset_name}\n{'=' * 60}")
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    model = MultiRocket(n_kernels=6250, max_dilations_per_kernel=32,
                       n_features_per_kernel=4, normalise=False,
                       n_jobs=1, random_state=RANDOM_STATE)

    start = perf_counter()
    X_train_features = model.fit_transform(X_train)
    train_transform = perf_counter() - start

    start = perf_counter()
    X_test_features = model.transform(X_test)
    test_transform = perf_counter() - start

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    start = perf_counter()
    classifier.fit(X_train_features, y_train)
    classifier_fit = perf_counter() - start

    start = perf_counter()
    y_pred = classifier.predict(X_test_features)
    prediction = perf_counter() - start

    row = {
        "dataset": dataset_name, "model": "MultiRocket",
        "random_state": RANDOM_STATE, "n_train": len(y_train), "n_test": len(y_test),
        "n_features": X_train_features.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "feature_extraction_train_time_seconds": train_transform,
        "classifier_train_time_seconds": classifier_fit,
        "total_training_time_seconds": train_transform + classifier_fit,
        "test_feature_extraction_time_seconds": test_transform,
        "prediction_time_seconds": prediction,
        "prediction_time_per_case_seconds": prediction / len(y_test),
        "end_to_end_test_time_seconds": test_transform + prediction,
    }
    print(f"Accuracy={row['accuracy']:.4f} | Macro F1={row['macro_f1']:.4f} | "
          f"Train={row['total_training_time_seconds']:.3f}s | "
          f"End-to-end test={row['end_to_end_test_time_seconds']:.3f}s")
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
                    (old["random_state"] == RANDOM_STATE) &
                    (old["model"] == "MultiRocket"))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {len(rows)} dataset results to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
