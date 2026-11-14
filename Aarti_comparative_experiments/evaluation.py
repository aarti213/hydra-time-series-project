from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "Aarti"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load experiment outputs
mini = pd.read_csv(RESULTS_DIR / "minirocket_results.csv")
multi = pd.read_csv(RESULTS_DIR / "multirocket_results.csv")
combined = pd.read_csv(RESULTS_DIR / "hydra_multirocket_results.csv")

# Standardise column names across the three scripts
mini_std = pd.DataFrame({
    "dataset": mini["dataset"],
    "model": "MiniRocket",
    "n_train": mini["n_train"],
    "n_test": mini["n_test"],
    "n_features": mini["n_features"],
    "accuracy": mini["accuracy"],
    "macro_f1": mini["macro_f1"],
    "total_training_time_seconds": mini["total_training_time_seconds"],
    "test_feature_extraction_time_seconds": mini["test_feature_extraction_time_seconds"],
    "prediction_time_seconds": mini["prediction_time_seconds"],
})

multi_std = pd.DataFrame({
    "dataset": multi["dataset"],
    "model": "MultiRocket",
    "n_train": multi["n_train"],
    "n_test": multi["n_test"],
    "n_features": multi["n_features"],
    "accuracy": multi["accuracy"],
    "macro_f1": multi["macro_f1"],
    "total_training_time_seconds": multi["total_training_time_seconds"],
    "test_feature_extraction_time_seconds": multi["test_feature_extraction_time_seconds"],
    "prediction_time_seconds": multi["prediction_time_seconds"],
})

combined_std = pd.DataFrame({
    "dataset": combined["dataset"],
    "model": "Hydra+MultiRocket",
    "n_train": combined["n_train"],
    "n_test": combined["n_test"],
    "n_features": combined["combined_features"],
    "accuracy": combined["accuracy"],
    "macro_f1": combined["macro_f1"],
    "total_training_time_seconds": combined["total_training_seconds"],
    "test_feature_extraction_time_seconds": (
        combined["hydra_test_transform_seconds"]
        + combined["multirocket_test_transform_seconds"]
    ),
    "prediction_time_seconds": combined["classifier_prediction_seconds"],
})

comparison = pd.concat([mini_std, multi_std, combined_std], ignore_index=True)
comparison["end_to_end_test_time_seconds"] = (
    comparison["test_feature_extraction_time_seconds"]
    + comparison["prediction_time_seconds"]
)
comparison["prediction_time_per_case_seconds"] = (
    comparison["prediction_time_seconds"] / comparison["n_test"]
)

# Consistent dataset and model ordering
dataset_order = ["Coffee", "ECG200", "GunPoint", "ItalyPowerDemand", "TwoLeadECG", "Wafer"]
model_order = ["MiniRocket", "MultiRocket", "Hydra+MultiRocket"]
comparison["dataset"] = pd.Categorical(comparison["dataset"], categories=dataset_order, ordered=True)
comparison["model"] = pd.Categorical(comparison["model"], categories=model_order, ordered=True)
comparison = comparison.sort_values(["dataset", "model"]).reset_index(drop=True)

output_csv = RESULTS_DIR / "model_comparison.csv"
comparison.to_csv(output_csv, index=False)

print("\nSIX-DATASET MODEL COMPARISON")
print("=" * 110)
columns = [
    "dataset", "model", "n_features", "accuracy", "macro_f1",
    "total_training_time_seconds", "end_to_end_test_time_seconds"
]
print(comparison[columns].round(4).to_string(index=False))
print(f"\nSaved comparison CSV: {output_csv}")

# Print mean metrics across these six datasets (unweighted dataset average)
summary = comparison.groupby("model", observed=False)[
    ["accuracy", "macro_f1", "total_training_time_seconds", "end_to_end_test_time_seconds"]
].mean().reindex(model_order)
summary_file = RESULTS_DIR / "model_comparison_mean_summary.csv"
summary.to_csv(summary_file)
print("\nUNWEIGHTED MEAN ACROSS THE SIX DATASETS")
print(summary.round(4).to_string())
print(f"Saved mean summary: {summary_file}")

def grouped_chart(metric, title, ylabel, filename, ylim=None):
    pivot = comparison.pivot(index="dataset", columns="model", values=metric)
    pivot = pivot.reindex(index=dataset_order, columns=model_order)
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(title="Model")
    plt.tight_layout()
    output = PLOTS_DIR / filename
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {output}")

grouped_chart(
    "accuracy", "Accuracy by Dataset", "Accuracy",
    "accuracy_by_dataset.png", ylim=(0, 1.05)
)
grouped_chart(
    "macro_f1", "Macro F1 by Dataset", "Macro F1",
    "macro_f1_by_dataset.png", ylim=(0, 1.05)
)
grouped_chart(
    "total_training_time_seconds", "Total Training Time by Dataset",
    "Seconds", "training_time_by_dataset.png"
)
grouped_chart(
    "end_to_end_test_time_seconds", "End-to-End Test Time by Dataset",
    "Seconds", "end_to_end_test_time_by_dataset.png"
)

print("\nNote: timing comparisons are most meaningful when all runs use the same machine,")
print("environment, and non-concurrent execution. End-to-end test time includes feature")
print("extraction plus classifier prediction for all three models.")
