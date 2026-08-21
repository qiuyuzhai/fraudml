import json

runs = {
    "Baseline (no FE)":         "artifacts/run_20260820_223923_21c4fe",
    "Default (relaxed IV/PSI)": "artifacts/run_20260820_224931_cd07eb",
    "Default (strict IV)":      "artifacts/run_20260820_213633_4088fc",
    "Full config":              "artifacts/run_20260820_221354_354528",
}

print(f"{'':30} {'AUC':>7} {'PR-AUC':>7} {'KS':>7} {'Recall':>7} {'Prec@5%':>7} {'n_feat':>7}")
print("-" * 75)

for name, path in runs.items():
    try:
        m = json.load(open(f"{path}/online_artifacts/metadata.json"))
        mt = m.get("metrics", {})
        n = len(m.get("selected_features", []))
        print(f"{name:30} {mt.get('auc',0):>7.4f} {mt.get('pr_auc',0):>7.4f} {mt.get('ks',0):>7.4f} "
              f"{mt.get('recall',0):>7.4f} {mt.get('precision_at_top_5pct',0):>7.4f} {n:>7}")
    except FileNotFoundError:
        print(f"{name:30} (not found)")
