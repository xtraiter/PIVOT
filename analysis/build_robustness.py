"""
build_robustness.py — Tuan 10
=============================
Tong hop log robustness_t10/ + tai dung diem CLEAN tu grid_t78_wn18rr
(ppr_s{S}_tk0.1_test.log, rerank_s{S}_tk0.1_test.log — cung dieu kien do).
Xuat: bang degradation (mean+/-std, ddof=1), retention %, figure 2 panel,
bao cao markdown de dan vao muc Tuan 10 cua walkthrough.

Canh bao tu dong: neu bat ky o nhieu nao cao hon clean > 0.005 thi in
"CANH BAO: nghi lan cache PPR sach" va yeu cau dung lai.

Chay:
  python3 build_robustness.py --dir robustness_t10 --clean_dir grid_t78_wn18rr
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RE_TEST   = re.compile(r"\[TEST\]\s+MRR:([\-\d.]+)\s+H@1:([\-\d.]+)\s+H@10:([\-\d.]+)")
RE_ROB    = re.compile(r"rob_WN18RR_(\w+?)_(ppr|rerank|hybrid)_s(\d+)\.log$")
RE_CLEAN  = re.compile(r"(ppr|rerank|hybrid)_s(\d+)_tk0\.1_test\.log$")
ORDER     = ["clean", "del05", "del10", "del20", "reldel"]
MLAB      = {"ppr": "PPR-only", "rerank": "PIVOT-Rerank", "hybrid": "Hybrid+Rerank"}
COLORS    = {"ppr": "#7f8c8d", "rerank": "#e74c3c", "hybrid": "#2ecc71"}


def parse_mrr(path):
    """Lay Test MRR tu dong [TEST] cuoi cung trong log."""
    ms = RE_TEST.findall(open(path, errors="ignore").read())
    if not ms:
        return None
    val = float(ms[-1][0])
    return val if val > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/robustness_t10")
    ap.add_argument("--clean_dir", default="reports/grid_t78_wn18rr")
    args = ap.parse_args()

    rows = []

    # Load diem CLEAN tu grid_t78_wn18rr
    for p in glob.glob(os.path.join(args.clean_dir, "*_tk0.1_test.log")):
        m = RE_CLEAN.search(os.path.basename(p))
        if m:
            v = parse_mrr(p)
            if v is not None:
                rows.append({"config": "clean", "method": m.group(1),
                             "seed": int(m.group(2)), "mrr": v, "log": p})

    # Load diem NHIEU tu robustness_t10/
    for p in glob.glob(os.path.join(args.dir, "rob_*.log")):
        m = RE_ROB.search(os.path.basename(p))
        if m:
            v = parse_mrr(p)
            if v is not None:
                rows.append({"config": m.group(1), "method": m.group(2),
                             "seed": int(m.group(3)), "mrr": v, "log": p})

    if not rows:
        raise SystemExit("Khong parse duoc log nao — kiem tra duong dan.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.dir, "robustness_points.csv"), index=False)

    # Aggregate mean +/- std (ddof=1)
    agg = df.groupby(["config", "method"]).agg(
        mrr_mean=("mrr", "mean"),
        mrr_std=("mrr", lambda x: float(x.std(ddof=1))),
        n_seeds=("seed", "nunique")
    ).reset_index()

    # Tinh delta va retention
    clean_vals = {m: agg[(agg.config == "clean") & (agg.method == m)]["mrr_mean"].iloc[0]
                  for m in agg.method.unique()}
    agg["delta_vs_clean"] = agg.apply(lambda r: r.mrr_mean - clean_vals[r.method], axis=1)
    agg["retention_pct"]  = agg.apply(lambda r: 100.0 * r.mrr_mean / clean_vals[r.method], axis=1)
    agg.to_csv(os.path.join(args.dir, "robustness_agg.csv"), index=False)

    print("\n=== robustness_agg ===")
    print(agg[["config","method","mrr_mean","mrr_std","delta_vs_clean","retention_pct"]].round(4).to_string(index=False))

    # ── Canh bao lan cache ──
    warn = False
    for _, r in agg.iterrows():
        if r.config != "clean" and r.mrr_mean > clean_vals[r.method] + 0.005:
            print(f"\n!! CANH BAO: {r.config}/{r.method} CAO hon clean "
                  f"({r.mrr_mean:.4f} vs {clean_vals[r.method]:.4f}) — "
                  f"NGHI LAN CACHE PPR SACH. Kiem tra ppr_scores/ trong thu muc "
                  f"WN18RR_{r.config} co phai moi hay khong.")
            warn = True
    if warn:
        raise SystemExit("STOP: co canh bao lan cache — kiem tra truoc khi chap nhan so lieu.")

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150,
                              gridspec_kw={"width_ratios": [2.2, 1]})

    # Panel A: random deletion curve
    rand_configs = {"clean": 0, "del05": 5, "del10": 10, "del20": 20}
    rand = agg[agg.config.isin(rand_configs)].copy()
    rand["x"] = rand.config.map(rand_configs)
    for m in sorted(rand.method.unique()):
        A = rand[rand.method == m].sort_values("x")
        axes[0].errorbar(A.x, A.mrr_mean, yerr=A.mrr_std,
                         marker="o", color=COLORS.get(m, "k"),
                         label=MLAB.get(m, m), capsize=3, ms=6, lw=1.8)
        for _, row in A.iterrows():
            axes[0].annotate(f"{row.retention_pct:.1f}%",
                             (row.x, row.mrr_mean),
                             textcoords="offset points", xytext=(4, 5), fontsize=7)
    axes[0].set_xlabel("Ty le xoa canh ngau nhien tren facts∪train (%)", fontsize=9)
    axes[0].set_ylabel("Test MRR", fontsize=9)
    axes[0].set_xticks([0, 5, 10, 20])
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("(a) Random edge deletion", fontsize=10)

    # Panel B: reldel bar chart
    rel = agg[agg.config.isin(["clean", "reldel"])]
    if (rel.config == "reldel").any():
        xs = {"ppr": -0.25, "rerank": 0, "hybrid": 0.25}
        w = 0.22
        for m in sorted(rel.method.unique()):
            for i, cfg in enumerate(["clean", "reldel"]):
                r = rel[(rel.method == m) & (rel.config == cfg)]
                if r.empty:
                    continue
                r = r.iloc[0]
                bar = axes[1].bar(i + xs[m], r.mrr_mean, width=w,
                                  yerr=r.mrr_std, color=COLORS.get(m, "k"),
                                  alpha=0.5 if cfg == "clean" else 0.9,
                                  capsize=3, label=MLAB.get(m, m) if i == 0 else "_")
                axes[1].text(i + xs[m], r.mrr_mean + r.mrr_std + 0.001,
                             f"{r.mrr_mean:.4f}", ha="center", fontsize=7)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["clean", "reldel"])
        axes[1].set_ylabel("Test MRR", fontsize=9)
        axes[1].grid(alpha=0.3, axis="y")
        lo = max(0, rel.mrr_mean.min() - 0.015)
        axes[1].set_ylim(lo, rel.mrr_mean.max() + 0.02)
        axes[1].set_title("(b) Relation-specific deletion", fontsize=10)
        axes[1].legend(fontsize=8, loc="lower right")
    else:
        axes[1].text(0.5, 0.5, "reldel chua co\n(chua chay)", ha="center",
                     va="center", transform=axes[1].transAxes, fontsize=10)
        axes[1].set_title("(b) Relation-specific deletion", fontsize=10)

    fig.suptitle("Tuan 10 — Robustness: PPR-only vs PIVOT-Rerank (WN18RR, FP32, θ=10%)",
                 fontsize=11)
    fig.tight_layout()
    out_fig = os.path.join(args.dir, "figure_robustness_wn18rr.png")
    fig.savefig(out_fig, bbox_inches="tight")
    print(f"Da luu figure: {out_fig}")

    # ── Bao cao markdown ──
    lines = [
        "## Bảng Degradation — Robustness Suite WN18RR (mean±std, 3 seed, FP32, θ=10%)\n",
        "| Config | Phương pháp | Test MRR | Δ vs clean | Retention % |",
        "|:---:|:---:|:---:|:---:|:---:|"
    ]
    for cfg in ORDER:
        for m in ["ppr", "rerank", "hybrid"]:
            r = agg[(agg.config == cfg) & (agg.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"| {cfg} | {MLAB.get(m, m)} "
                f"| {r.mrr_mean:.4f} ± {r.mrr_std:.4f} "
                f"| {r.delta_vs_clean:+.4f} "
                f"| {r.retention_pct:.1f}% |"
            )

    lines += [
        "\n## Khoảng cách Rerank − PPR theo mức nhiễu\n",
        "| Config | Δ(Rerank − PPR) | Nhận xét |",
        "|:---:|:---:|:---|"
    ]
    for cfg in ORDER:
        a = agg[(agg.config == cfg) & (agg.method == "rerank")]
        b = agg[(agg.config == cfg) & (agg.method == "ppr")]
        if a.empty or b.empty:
            continue
        diff = a.mrr_mean.iloc[0] - b.mrr_mean.iloc[0]
        clean_diff = (agg[(agg.config=="clean") & (agg.method=="rerank")]["mrr_mean"].iloc[0]
                      - agg[(agg.config=="clean") & (agg.method=="ppr")]["mrr_mean"].iloc[0])
        trend = "thu hep" if diff < clean_diff - 0.001 else (
                "giu vung" if abs(diff - clean_diff) <= 0.001 else "mo rong")
        lines.append(f"| {cfg} | {diff:+.4f} | Gap {trend} so voi clean ({clean_diff:+.4f}) |")

    rpath = os.path.join(args.dir, "robustness_report_wn18rr.md")
    open(rpath, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"Da luu bao cao: {rpath}")

    print(f"\n=== CROSS-CHECK ===")
    p_clean = agg[(agg.config=="clean") & (agg.method=="ppr")]
    r_clean = agg[(agg.config=="clean") & (agg.method=="rerank")]
    if not p_clean.empty:
        pv = p_clean.iloc[0]
        print(f"clean/ppr:    {pv.mrr_mean:.4f} +/- {pv.mrr_std:.4f}  (can khop 0.5638 +/- 0.0017)")
    if not r_clean.empty:
        rv = r_clean.iloc[0]
        print(f"clean/rerank: {rv.mrr_mean:.4f} +/- {rv.mrr_std:.4f}  (can khop 0.5685 +/- 0.0021)")
    
    h_clean = agg[(agg.config=="clean") & (agg.method=="hybrid")]
    if not h_clean.empty:
        hv = h_clean.iloc[0]
        print(f"clean/hybrid: {hv.mrr_mean:.4f} +/- {hv.mrr_std:.4f}  (can khop ~0.5684)")


if __name__ == "__main__":
    main()
