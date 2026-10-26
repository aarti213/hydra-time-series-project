from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42
DATASET_NAME = "Coffee"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "Aarti"
RESULTS_FILE = RESULTS_DIR / "minirocket_results.csv"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}")

    X_train, y_train = load_classification(
        DATASET_NAME,
        split="train",
    )

    X_test, y_test = load_classification(
        DATASET_NAME,
        split="test",
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of training cases: {len(y_train)}")
    print(f"Number of test cases: {len(y_test)}")

    # --------------------------------------------------
    # MiniRocket feature extraction
    # --------------------------------------------------

    minirocket = MiniRocket(
        n_kernels=10000,
        max_dilations_per_kernel=32,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    print("\nFitting MiniRocket and transforming training data...")

    start = perf_counter()
    X_train_features = minirocket.fit_transform(X_train)
    feature_extraction_train_time = perf_counter() - start

    print("Transforming test data...")

    start = perf_counter()
    X_test_features = minirocket.transform(X_test)
    feature_extraction_test_time = perf_counter() - start

    print(f"Training features shape: {X_train_features.shape}")
    print(f"Test features shape: {X_test_features.shape}")

    # --------------------------------------------------
    # Train classifier
    # --------------------------------------------------

    classifier = RidgeClassifierCV(
        alphas=np.logspace(-3, 3, 10)
    )

    print("\nTraining Ridge classifier...")

    start = perf_counter()
    classifier.fit(X_train_features, y_train)
    classifier_train_time = perf_counter() - start

    # --------------------------------------------------
    # Evaluate predictions
    # --------------------------------------------------

    print("Predicting test labels...")

    start = perf_counter()
    y_pred = classifier.predict(X_test_features)
    prediction_time = perf_counter() - start

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    total_training_time = (
        feature_extraction_train_time + classifier_train_time
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    result = {
        "dataset": DATASET_NAME,
        "model": "MiniRocket",
        "random_state": RANDOM_STATE,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X_train_features.shape[1],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "feature_extraction_train_time_seconds":
            feature_extraction_train_time,
        "classifier_train_time_seconds": classifier_train_time,
        "total_training_time_seconds": total_training_time,
        "test_feature_extraction_time_seconds":
            feature_extraction_test_time,
        "prediction_time_seconds": prediction_time,
        "prediction_time_per_case_seconds":
            prediction_time / len(y_test),
    }

    results_df = pd.DataFrame([result])
    results_df.to_csv(RESULTS_FILE, index=False)

    print("\n========== MiniRocket Results ==========")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Total training time: {total_training_time:.4f} seconds")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()