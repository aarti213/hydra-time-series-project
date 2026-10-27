import pandas as pd
from pathlib import Path

results_dir = Path("results/Aarti")

files = {
    "MiniRocket": "minirocket_results.csv",
    "MultiRocket": "multirocket_results.csv",
    "Hydra+MultiRocket": "hydra_multirocket_results.csv",
}

for model, filename in files.items():
    print(f"\n{'=' * 15} {model} {'=' * 15}")
    path = results_dir / filename

    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)
    wafer = df[df["dataset"].astype(str).str.strip() == "Wafer"]

    if wafer.empty:
        print("No Wafer results found.")
    else:
        print(wafer.to_string(index=False))
