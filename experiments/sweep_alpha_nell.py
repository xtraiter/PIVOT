import os
import argparse as _ap
_p = _ap.ArgumentParser(description="Alpha sweep NELL-995")
_p.add_argument("--fact_ratio", type=float, default=0.95)
ARGS = _p.parse_args()

import re
import sys
import subprocess
import time
import pandas as pd
import numpy as np

SEEDS = [42, 123, 1234]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

GNN_CHECKPOINTS = {
    42: "data/nell/saveModel/topk_0.1_layer_8_ValMRR_0.497_seed42.pt",
    123: "data/nell/saveModel/topk_0.1_layer_8_ValMRR_0.504_seed123.pt",
    1234: "data/nell/saveModel/topk_0.1_layer_8_ValMRR_0.498_seed1234.pt"
}

PARSE_REGEX = re.compile(
    r"\[VALID\]\s+MRR:([\d.]+)\s+H@1:([\d.]+)\s+H@10:([\d.]+).*?"
    r"\[TEST\]\s+MRR:([\d.]+)\s+H@1:([\d.]+)\s+H@10:([\d.]+).*?"
    r"\[LATENCY\]\s+eval_total_ms:([\d.]+)\s+data_prep_ms:([\d.]+)\s+forward_ms:([\d.]+)\s+ranking_ms:([\d.]+).*?"
    r"\[PEAK_GPU_MEM\]\s+([\d.]+)MB",
    re.DOTALL,
)

def run_eval(seed, alpha):
    gnn_weight = GNN_CHECKPOINTS[seed]
    mlp_weight = f"data/nell/budget_results/pruning_mlp_v2_best_seed_{seed}.pt"
    
    cmd = [
        sys.executable, "train_auto.py",
        "--data_path", "./data/nell",
        "--batchsize", "8",
        "--only_eval",
        "--gpu", "0",
        "--topk", "0.1",
        "--topm", "-1",
        "--fact_ratio", "0.95",
        "--seed", str(seed),
        "--weight", gnn_weight,
        "--rerank_alpha", str(alpha),
        "--eval_split", "all",
        "--no_amp"
    ]
    
    if alpha > 0.0:
        cmd += ["--pruning_model_path", mlp_weight]

    print(f"--> Running: seed={seed}, alpha={alpha} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    
    m = PARSE_REGEX.search(out)
    if not m:
        # Save output to a debug file
        debug_file = f"data/nell/results/debug_seed_{seed}_alpha_{alpha}.log"
        with open(debug_file, "w") as df:
            df.write(out)
        raise RuntimeError(f"Could not parse metrics for seed={seed}, alpha={alpha}. Debug log saved to {debug_file}")
        
    val_mrr, val_h1, val_h10, test_mrr, test_h1, test_h10, eval_ms, data_prep_ms, forward_ms, ranking_ms, peak_mem = map(float, m.groups())
    
    return {
        "seed": seed,
        "alpha": alpha,
        "valid_mrr": val_mrr,
        "valid_h1": val_h1,
        "valid_h10": val_h10,
        "test_mrr": test_mrr,
        "test_h1": test_h1,
        "test_h10": test_h10,
        "eval_time_ms": eval_ms,
        "peak_gpu_mem_mb": peak_mem
    }

def main():
    results = []
    
    for seed in SEEDS:
        print(f"\n======================================")
        print(f"Sweeping Alpha for Seed {seed}")
        print(f"======================================")
        
        for alpha in ALPHAS:
            try:
                res = run_eval(seed, alpha)
                results.append(res)
                print(f"  Alpha {alpha:.1f} | Valid MRR: {res['valid_mrr']:.6f} | Test MRR: {res['test_mrr']:.6f} | Peak GPU: {res['peak_gpu_mem_mb']:.1f}MB")
            except Exception as e:
                print(f"  [ERROR] {e}")
                
    # Save raw results to CSV
    df = pd.DataFrame(results)
    os.makedirs("data/nell/budget_results", exist_ok=True)
    df.to_csv("data/nell/budget_results/alpha_sweep_raw.csv", index=False)
    print("\nSaved raw sweep results to data/nell/budget_results/alpha_sweep_raw.csv")
    
    # Generate markdown report
    report_lines = []
    report_lines.append("# Kết Quả Sweep Tham Số α — Post-hoc Reranking (NELL-995)")
    report_lines.append("")
    report_lines.append("## Bảng kết quả chi tiết từng seed")
    report_lines.append("")
    
    for seed in SEEDS:
        report_lines.append(f"### Seed {seed}")
        report_lines.append("| α | Valid MRR | Test MRR | Test H@1 | Test H@10 | Eval Time (s) | Peak GPU |")
        report_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        
        seed_df = df[df["seed"] == seed].sort_values("alpha")
        for _, r in seed_df.iterrows():
            report_lines.append(
                f"| {r['alpha']:.1f} | {r['valid_mrr']:.6f} | {r['test_mrr']:.6f} | "
                f"{r['test_h1']*100:.2f}% | {r['test_h10']*100:.2f}% | "
                f"{r['eval_time_ms']/1000.0:.1f}s | {r['peak_gpu_mem_mb']:.1f} MB |"
            )
        report_lines.append("")
        
    # Validation-based Protocol Summary
    report_lines.append("## Đánh Giá Đa Seed Theo Quy Trình Chuẩn (Validation-based Protocol)")
    report_lines.append("")
    report_lines.append("1. Tìm $\\alpha^*$ đạt `Valid MRR` lớn nhất cho từng seed độc lập.")
    report_lines.append("2. Báo cáo `Test MRR` duy nhất tại giá trị $\\alpha^*$ tương ứng.")
    report_lines.append("")
    report_lines.append("| Seed | $\\alpha^*$ (Valid-chosen) | Valid MRR | Test MRR | Baseline Test MRR (α=0.0) | Delta |")
    report_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    val_chosen_rows = []
    for seed in SEEDS:
        seed_df = df[df["seed"] == seed]
        best_row = seed_df.loc[seed_df["valid_mrr"].idxmax()]
        baseline_row = seed_df[seed_df["alpha"] == 0.0].iloc[0]
        delta = best_row["test_mrr"] - baseline_row["test_mrr"]
        
        report_lines.append(
            f"| **{seed}** | {best_row['alpha']:.1f} | {best_row['valid_mrr']:.6f} | "
            f"**{best_row['test_mrr']:.6f}** | {baseline_row['test_mrr']:.6f} | "
            f"{'+' if delta >= 0 else ''}{delta:.6f} |"
        )
        val_chosen_rows.append({
            "seed": seed,
            "alpha": best_row['alpha'],
            "test_mrr": best_row['test_mrr'],
            "baseline_mrr": baseline_row['test_mrr']
        })
        
    test_mrrs = [r["test_mrr"] for r in val_chosen_rows]
    base_mrrs = [r["baseline_mrr"] for r in val_chosen_rows]
    
    report_lines.append(
        f"| **Mean ± Std** | — | — | "
        f"**{np.mean(test_mrrs):.6f} ± {np.std(test_mrrs):.6f}** | "
        f"{np.mean(base_mrrs):.6f} ± {np.std(base_mrrs):.6f} | "
        f"+{np.mean(test_mrrs) - np.mean(base_mrrs):.6f} |"
    )
    
    report_text = "\n".join(report_lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/alpha_sweep_results_nell_no_amp.md", "w") as f:
        f.write(report_text)
        
    print("\n======================================")
    print("Summary of Validation-based Protocol:")
    print("======================================")
    print(report_text)

if __name__ == '__main__':
    main()
