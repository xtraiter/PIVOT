import os
import argparse as _ap
_p = _ap.ArgumentParser(description="Alpha sweep WN18RR (hop nhat 0.75 & 0.95)")
_p.add_argument("--fact_ratio", type=float, default=0.95, choices=[0.75, 0.95])
ARGS = _p.parse_args()
_SUF = "fact95" if ARGS.fact_ratio == 0.95 else "v2"
import re
import sys
import subprocess
import numpy as np
import pandas as pd

SEEDS = [42, 123, 1234]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# GNN checkpoints per seed (best Val MRR)
GNN_CHECKPOINTS = {
    42:   "data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.564_seed42.pt",
    123:  "data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.565_seed123.pt",
    1234: "data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.565_seed1234.pt",
}

PARSE_REGEX = re.compile(
    r"\[VALID\]\s+MRR:([\d.]+)\s+H@1:([\d.]+)\s+H@10:([\d.]+).*?"
    r"\[TEST\]\s+MRR:([\d.]+)\s+H@1:([\d.]+)\s+H@10:([\d.]+).*?"
    r"\[LATENCY\]\s+eval_total_ms:([\d.]+).*?"
    r"\[PEAK_GPU_MEM\]\s+([\d.]+)MB",
    re.DOTALL,
)


def run_eval(seed, alpha):
    gnn_weight = GNN_CHECKPOINTS[seed]
    mlp_weight = f"data/WN18RR/budget_results/pruning_mlp_v2_best_seed_{seed}.pt"

    cmd = [
        sys.executable, "train_auto.py",
        "--data_path", "./data/WN18RR/",
        "--batchsize", "16",
        "--only_eval",
        "--gpu", "0",
        "--topk", "0.1",
        "--topm", "-1",
        "--fact_ratio", str(ARGS.fact_ratio),
        "--seed", str(seed),
        "--weight", gnn_weight,
        "--rerank_alpha", str(alpha),
        "--eval_split", "all",
        "--no_amp"
    ]

    if alpha > 0.0:
        cmd += ["--pruning_model_path", mlp_weight]

    print(f"--> Running: seed={seed}, alpha={alpha} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/vanba/KLTN/one-shot-subgraph")
    out = proc.stdout + proc.stderr

    m = PARSE_REGEX.search(out)
    if not m:
        os.makedirs("data/WN18RR/results", exist_ok=True)
        debug_file = f"data/WN18RR/results/debug_seed_{seed}_alpha_{alpha}.log"
        with open(debug_file, "w") as df:
            df.write(out)
        print(f"  [ERROR] Could not parse metrics for seed={seed}, alpha={alpha}. Debug log saved to {debug_file}")
        return None

    val_mrr, val_h1, val_h10, test_mrr, test_h1, test_h10, eval_ms, peak_mem = map(float, m.groups())

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
        "peak_gpu_mem_mb": peak_mem,
    }


def main():
    results = []

    for seed in SEEDS:
        print(f"\n======================================")
        print(f"Sweeping Alpha for Seed {seed}")
        print(f"======================================")

        for alpha in ALPHAS:
            res = run_eval(seed, alpha)
            if res is not None:
                results.append(res)
                print(f"  Alpha {alpha:.1f} | Valid MRR: {res['valid_mrr']:.6f} | Test MRR: {res['test_mrr']:.6f} | Peak GPU: {res['peak_gpu_mem_mb']:.1f}MB")

    # Save raw CSV
    df = pd.DataFrame(results)
    os.makedirs("data/WN18RR/budget_results", exist_ok=True)
    df.to_csv("data/WN18RR/budget_results/alpha_sweep_raw_" + _SUF + ".csv", index=False)
    print("\nSaved raw sweep results to data/WN18RR/budget_results/alpha_sweep_raw_" + _SUF + ".csv")

    # ------------ Build Markdown report ------------
    lines = []
    lines.append("# Báo Cáo Kết Quả Sweep Tham Số \u03b1 \u2014 Post-hoc Reranking (WN18RR) — Phiên Bản Mới (MLP v2, fact_ratio=0.95, no_amp=True)")
    lines.append("")
    lines.append("> **MLP checkpoint:** `data/WN18RR/budget_results/pruning_mlp_v2_best_seed_<seed>.pt` (train lại ngày 10/07/2026, `fact_ratio=0.75` - suy luận trên `fact_ratio=0.95`)")
    lines.append("> **GNN checkpoint:** checkpoint tốt nhất theo Valid MRR cho từng seed.")
    lines.append("> **Protocol:** Validation-based \u2014 ch\u1ecdn \u03b1\u002a theo Valid MRR, b\u00e1o c\u00e1o Test MRR duy nh\u1ea5t t\u1ea1i \u03b1\u002a \u0111\u00f3.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-seed detail tables
    lines.append("## 1. B\u1ea3ng K\u1ebft Qu\u1ea3 Sweep Chi Ti\u1ebft T\u1eebng Seed")
    lines.append("")

    for seed in SEEDS:
        lines.append(f"### Seed {seed}")
        lines.append("")
        lines.append("| \u03b1 | Valid MRR | **Test MRR** | Test H@1 | Test H@10 | Eval Time | Peak GPU |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        seed_df = df[df["seed"] == seed].sort_values("alpha")
        # find best alpha
        best_alpha = seed_df.loc[seed_df["valid_mrr"].idxmax(), "alpha"]
        for _, r in seed_df.iterrows():
            marker = " \u2b50" if r["alpha"] == best_alpha else ""
            lines.append(
                f"| **{r['alpha']:.1f}**{marker} | {r['valid_mrr']:.4f} | **{r['test_mrr']:.4f}** | "
                f"{r['test_h1']*100:.2f}% | {r['test_h10']*100:.2f}% | "
                f"{r['eval_time_ms']/1000.0:.1f}s | {r['peak_gpu_mem_mb']:.1f} MB |"
            )
        lines.append("")

    # Validation-based protocol summary
    lines.append("---")
    lines.append("")
    lines.append("## 2. \u0110\u00e1nh Gi\u00e1 \u0110a Seed Theo Quy Tr\u00ecnh Chu\u1ea9n (Validation-based Protocol)")
    lines.append("")
    lines.append("1. T\u00ecm \u03b1\u002a \u0111\u1ea1t `Valid MRR` l\u1edbn nh\u1ea5t cho t\u1eebng seed m\u1ed9t c\u00e1ch \u0111\u1ed9c l\u1eadp.")
    lines.append("2. B\u00e1o c\u00e1o `Test MRR` duy nh\u1ea5t t\u1ea1i gi\u00e1 tr\u1ecb \u03b1\u002a t\u01b0\u01a1ng \u1ee9ng.")
    lines.append("")
    lines.append("| Seed | \u03b1\u002a (Ch\u1ecdn theo Valid) | Valid MRR t\u1ea1i \u03b1\u002a | Test MRR t\u1ea1i \u03b1\u002a | PPR Baseline Test MRR (\u03b1=0.0) | Delta |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")

    val_chosen = []
    for seed in SEEDS:
        seed_df = df[df["seed"] == seed]
        best_row = seed_df.loc[seed_df["valid_mrr"].idxmax()]
        base_row = seed_df[seed_df["alpha"] == 0.0].iloc[0]
        delta = best_row["test_mrr"] - base_row["test_mrr"]
        lines.append(
            f"| **{seed}** | {best_row['alpha']:.1f} | {best_row['valid_mrr']:.4f} | "
            f"**{best_row['test_mrr']:.4f}** | {base_row['test_mrr']:.4f} | "
            f"{'+' if delta >= 0 else ''}{delta:.4f} |"
        )
        val_chosen.append({"seed": seed, "test_mrr": best_row["test_mrr"], "baseline_mrr": base_row["test_mrr"]})

    test_mrrs = [r["test_mrr"] for r in val_chosen]
    base_mrrs = [r["baseline_mrr"] for r in val_chosen]
    mean_test = np.mean(test_mrrs)
    std_test = np.std(test_mrrs)
    mean_base = np.mean(base_mrrs)
    std_base = np.std(base_mrrs)
    delta_mean = mean_test - mean_base

    lines.append(
        f"| **Mean \u00b1 Std** | \u2014 | \u2014 | "
        f"**{mean_test:.4f} \u00b1 {std_test:.4f}** | "
        f"{mean_base:.4f} \u00b1 {std_base:.4f} | "
        f"+{delta_mean:.4f} |"
    )
    lines.append("")
    lines.append(f"> **K\u1ebft lu\u1eadn:** PIVOT Reranking c\u1ea3i thi\u1ec7n t\u1eeb **{mean_base:.4f} \u00b1 {std_base:.4f}** l\u00ean **{mean_test:.4f} \u00b1 {std_test:.4f}** (t\u0103ng trung b\u00ecnh **+{delta_mean:.4f} MRR**).")
    lines.append("")

    report_text = "\n".join(lines)

    # Save to reports/
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/alpha_sweep_results_" + _SUF + "_no_amp.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nMarkdown report saved to {report_path}")

    print("\n======================================")
    print("Summary (Validation-based Protocol):")
    print("======================================")
    for r in val_chosen:
        print(f"  Seed {r['seed']}: Test MRR = {r['test_mrr']:.4f}  (baseline = {r['baseline_mrr']:.4f})")
    print(f"  Mean: {mean_test:.4f} ± {std_test:.4f}  |  Baseline: {mean_base:.4f} ± {std_base:.4f}  |  Delta: +{delta_mean:.4f}")


if __name__ == "__main__":
    main()
