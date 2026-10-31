"""Collect results/runs/*/seed*/metrics.json into results.csv (per run) and results_summary.csv (mean ± std)."""
import glob
import json
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, "runs", "*", "seed*", "metrics.json")))]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(root, "results.csv"), index=False)

num = df.select_dtypes("number").drop(columns=["seed"])
g = df.groupby("variant")
mean, std = g[num.columns].mean(), g[num.columns].std(ddof=1)
fmt = lambda m, sd: f"{m:.3f} ± {sd:.3f}" if pd.notna(sd) else f"{m:.3f}"
summary = pd.DataFrame({c: [fmt(m, sd) for m, sd in zip(mean[c], std[c])] for c in num.columns}, index=mean.index)
summary.insert(0, "n_seeds", g.size())
order = [v for v in ["baseline", "extra", "corr", "both"] if v in summary.index]
summary.loc[order].to_csv(os.path.join(root, "results_summary.csv"))
print(summary.loc[order].T.to_string())
