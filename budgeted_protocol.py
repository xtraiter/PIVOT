import argparse
import re
import sys
import subprocess
import time
from pathlib import Path

import pandas as pd

# Regex khop voi dinh dang log trong file train_auto.py cua ban (da thay
# trong log thuc te ban gui). re.DOTALL de "." khop ca xuong dong.
PARSE_REGEX = re.compile(
    r"\[TEST\]\s+MRR:([\d.]+)\s+H@1:([\d.]+)\s+H@10:([\d.]+).*?"
    r"\[LATENCY\]\s+eval_total_ms:([\d.]+)\s+data_prep_ms:([\d.]+)\s+forward_ms:([\d.]+)\s+ranking_ms:([\d.]+).*?"
    r"\[PEAK_GPU_MEM\]\s+([\d.]+)MB",
    re.DOTALL,
)


def run_one(data_path, weight, gpu, topk, batchsize=16, extra_args=None,
            train_script="train_auto.py"):
    """
    Chay 1 lan --only_eval voi mot muc budget (topk) cu the, tra ve dict
    metric da parse duoc tu stdout/stderr cua train_auto.py.
    """
    cmd = [
        sys.executable, train_script,
        "--data_path", data_path,
        "--batchsize", str(batchsize),
        "--only_eval",
        "--gpu", str(gpu),
        "--topk", str(topk),
        "--topm", "-1",
        "--weight", weight,
    ]
    if extra_args:
        cmd += extra_args

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_s = time.time() - t0

    out = proc.stdout + proc.stderr
    m = PARSE_REGEX.search(out)
    if not m:
        raise RuntimeError(
            f"Khong parse duoc output cho topk={topk}.\n"
            f"--- 1500 ky tu cuoi stdout/stderr ---\n{out[-1500:]}"
        )

    mrr, h1, h10, eval_ms, data_prep_ms, forward_ms, ranking_ms, peak_mem = map(float, m.groups())
    return {
        "budget": topk,
        "MRR": mrr,
        "Hits@1": h1,
        "Hits@10": h10,
        "eval_total_ms": eval_ms,
        "data_prep_ms": data_prep_ms,
        "forward_ms": forward_ms,
        "ranking_ms": ranking_ms,
        "peak_gpu_mem_mb": peak_mem,
        "wall_clock_s": wall_s,
    }


def run_budget_sweep(data_path, weight, gpu, budgets, n_queries, seeds=(0,),
                      batchsize=16, out_csv="budget_results.csv",
                      train_script="train_auto.py"):
    """
    Chay CUNG MOT checkpoint qua nhieu muc budget (va nhieu seed, neu eval
    co thanh phan stochastic -- thuong dropout da tat khi eval nen 1 seed
    la du, nhung giu lai cho an toan va de tinh std).
    """
    rows = []
    for seed in seeds:
        extra = ["--seed", str(seed)] if seed is not None else None
        for b in budgets:
            print(f"[budgeted_protocol] dang chay budget={b} seed={seed} ...")
            r = run_one(data_path, weight, gpu, b, batchsize, extra, train_script)
            r["seed"] = seed
            # throughput = so query / tong thoi gian eval (giay)
            r["throughput_qps"] = n_queries / (r["eval_total_ms"] / 1000.0)
            r["latency_per_query_ms"] = r["eval_total_ms"] / n_queries
            rows.append(r)
            print(
                f"  -> MRR={r['MRR']:.4f}  "
                f"latency/q={r['latency_per_query_ms']:.2f}ms  "
                f"peak_mem={r['peak_gpu_mem_mb']:.0f}MB"
            )

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nDa luu raw results -> {out_csv}")
    return df


def summarize(df, out_csv="budget_summary.csv", dataset_name="WN18RR", method_name="PPR-only"):
    """
    Gop mean +/- std qua cac seed, theo tung budget. Day la output dung de
    dien vao Table 2 cua paper (efficiency table).
    """
    agg = (
        df.groupby("budget")
        .agg(
            MRR_mean=("MRR", "mean"),
            MRR_std=("MRR", "std"),
            Hits1_mean=("Hits@1", "mean"),
            Hits10_mean=("Hits@10", "mean"),
            eval_total_mean=("eval_total_ms", "mean"),
            data_prep_mean=("data_prep_ms", "mean"),
            forward_mean=("forward_ms", "mean"),
            ranking_mean=("ranking_ms", "mean"),
            latency_mean=("latency_per_query_ms", "mean"),
            latency_std=("latency_per_query_ms", "std"),
            throughput_mean=("throughput_qps", "mean"),
            peak_mem_mean=("peak_gpu_mem_mb", "mean"),
        )
        .reset_index()
        .sort_values("budget")
    )
    
    formatted = agg.copy()
    formatted["Test MRR (Mean ± Std)"] = agg.apply(
        lambda r: f"{r['MRR_mean']:.6f} ± {r['MRR_std']:.6f}" if not pd.isna(r['MRR_std']) else f"{r['MRR_mean']:.6f} ± 0.000000",
        axis=1
    )
    
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
    
    formatted.insert(0, "Dataset", dataset_name)
    formatted.insert(1, "Phương pháp", method_name)
    
    cols_to_keep = [
        "Dataset", "Phương pháp", "Budget", "Test MRR (Mean ± Std)",
        "H@1 (Mean)", "H@10 (Mean)", "eval_total (ms)", "data_prep (ms)",
        "forward (ms)", "ranking (ms)", "Latency / query (ms)",
        "Throughput (q/s)", "Peak GPU Mem (MB)"
    ]
    formatted = formatted[cols_to_keep]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(out_csv, index=False)
    print(f"Da luu summary -> {out_csv}")
    print(formatted.to_string(index=False))
    return agg


def plot_pareto(summary_df, out_png="pareto_frontier.png"):
    """Figure 1 trong ke hoach: Accuracy (MRR) vs Latency, 1 diem / budget."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        summary_df["latency_mean"], summary_df["MRR_mean"],
        marker="o", linewidth=1.5,
    )
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"{int(row['budget'] * 100)}%",
            (row["latency_mean"], row["MRR_mean"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )
    ax.set_xlabel("Latency per query (ms)")
    ax.set_ylabel("MRR")
    ax.set_title("Accuracy-Latency Pareto frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    print(f"Da luu figure -> {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PIVOT budgeted inference benchmark")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--weight", required=True, help="duong dan checkpoint .pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--n_queries", type=int, required=True,
        help="so query trong test set, vd 3034 cho WN18RR test",
    )
    ap.add_argument(
        "--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--batchsize", type=int, default=16)
    ap.add_argument("--outdir", default="./budget_results")
    ap.add_argument("--dataset", default="WN18RR")
    ap.add_argument("--method", default="PPR-only")
    ap.add_argument(
        "--train_script", default="train_auto.py",
        help="duong dan train_auto.py cua repo goc (neu khong chay tu repo root)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_budget_sweep(
        args.data_path, args.weight, args.gpu, args.budgets,
        args.n_queries, args.seeds, args.batchsize,
        out_csv=str(outdir / "raw_results.csv"),
        train_script=args.train_script,
    )
    summary = summarize(df, out_csv=str(outdir / "summary.csv"), dataset_name=args.dataset, method_name=args.method)
    plot_pareto(summary, out_png=str(outdir / "pareto_frontier.png"))
