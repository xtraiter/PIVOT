import pandas as pd
import numpy as np
import os

seeds = [42, 123, 1234]
dataset = "WN18RR"
method = "PIVOT"
data_dir = "./data/WN18RR/budget_results"

dfs = []
for seed in seeds:
    csv_path = os.path.join(data_dir, f"seed_{seed}", "raw_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        dfs.append(df)
    else:
        print(f"Warning: {csv_path} not found.")

if not dfs:
    print("No seed results found.")
    exit(1)

all_df = pd.concat(dfs, ignore_index=True)

# Group by budget to compute mean and std
agg = all_df.groupby("budget").agg(
    MRR_mean=("MRR", "mean"),
    MRR_std=("MRR", "std"),
    Hits1_mean=("Hits@1", "mean"),
    Hits10_mean=("Hits@10", "mean"),
    eval_total_mean=("eval_total_ms", "mean"),
    data_prep_mean=("data_prep_ms", "mean"),
    forward_mean=("forward_ms", "mean"),
    ranking_mean=("ranking_ms", "mean"),
    latency_mean=("latency_per_query_ms", "mean"),
    throughput_mean=("throughput_qps", "mean"),
    peak_mem_mean=("peak_gpu_mem_mb", "mean"),
).reset_index().sort_values("budget")

formatted = agg.copy()
# Format MRR as Mean ± Std
formatted["Test MRR (Mean ± Std)"] = agg.apply(
    lambda r: f"{r['MRR_mean']:.6f} ± {r['MRR_std']:.6f}" if not pd.isna(r['MRR_std']) else f"{r['MRR_mean']:.6f} ± 0.000000",
    axis=1
)

# Rename columns to match Vietnamese paper requirements
formatted = formatted.rename(columns={
    "budget": "Budget",
    "Hits1_mean": "H@1 (Mean)",
    "Hits10_mean": "H@10 (Mean)",
    "eval_total_mean": "eval_total (ms)",
    "data_prep_mean": "data_prep (ms)",
    "forward_mean": "forward (ms)",
    "ranking_mean": "ranking (ms)",
    "latency_mean": "Latency / query (ms)",
    "throughput_mean": "Throughput (q/s)",
    "peak_mem_mean": "Peak GPU Mem (MB)",
})

formatted.insert(0, "Dataset", dataset)
formatted.insert(1, "Phương pháp", method)

cols_to_keep = [
    "Dataset", "Phương pháp", "Budget", "Test MRR (Mean ± Std)",
    "H@1 (Mean)", "H@10 (Mean)", "eval_total (ms)", "data_prep (ms)",
    "forward (ms)", "ranking (ms)", "Latency / query (ms)",
    "Throughput (q/s)", "Peak GPU Mem (MB)"
]
formatted = formatted[cols_to_keep]

out_csv = os.path.join(data_dir, "pivot_aggregated_summary.csv")
formatted.to_csv(out_csv, index=False)
print(f"Aggregated summary saved to: {out_csv}")
print("\nPIVOT Aggregated Results:")
print(formatted.to_string(index=False))

# Load baseline if available and print comparison
baseline_csv = os.path.join(data_dir, "baseline", "summary.csv")
if os.path.exists(baseline_csv):
    base_df = pd.read_csv(baseline_csv)
    print("\nBaseline PPR-only Results:")
    print(base_df.to_string(index=False))
    
    # Concat both for comparison table
    comparison_df = pd.concat([base_df, formatted], ignore_index=True)
    comparison_csv = os.path.join(data_dir, "final_comparison_summary.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nFinal comparison table saved to: {comparison_csv}")
