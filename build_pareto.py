"""
build_pareto.py — Tuan 7-8
==========================
Doc log grid (grid_t78_<ds>/), tong hop, trich Pareto frontier THEO VALID,
ve Figure 1 (2 panel: latency & VRAM), chay 2 truy van BudgetController,
xuat cache JSON (moi entry kem log_path) va bao cao markdown de dan vao §7.

Cach chay:
  python3 build_pareto.py --dir grid_t78_wn18rr --dataset wn18rr
  python3 build_pareto.py --dir grid_t78_nell   --dataset nell

Quy uoc:
- Frontier duoc TRICH THEO mean Valid MRR (protocol §1.2). Test MRR chi bao cao.
- latency/query = eval_total_ms cua luot TEST-ONLY / n_test_queries.
- std mau (ddof=1) tren 3 seed.
"""

import argparse, json, re, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RE_VALID = re.compile(r"\[VALID\] MRR:([\-\d.]+) H@1:([\-\d.]+) H@10:([\-\d.]+)")
RE_TEST  = re.compile(r"\[TEST\] MRR:([\-\d.]+) H@1:([\-\d.]+) H@10:([\-\d.]+)")
RE_LAT   = re.compile(r"eval_total_ms:([\d.]+)")
RE_MEM   = re.compile(r"\[PEAK_GPU_MEM\] ([\d.]+)MB")
RE_NAME  = re.compile(r"(ppr|rerank|hybrid)_s(\d+)_tk([\d.]+)_(valid|test)\.log$")

# WN18RR: 3134 triple * 2 = 6268; NELL-995: 3992 triple * 2 = 7984
N_TEST_Q = {"wn18rr": 6268, "nell": 7984}
METHOD_LABEL = {"ppr": "PPR-only", "rerank": "PIVOT-Rerank", "hybrid": "Hybrid+Rerank"}


def last(rx, text):
    ms = rx.findall(text)
    return ms[-1] if ms else None


def parse_dir(d):
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "*.log"))):
        m = RE_NAME.search(os.path.basename(path))
        if not m:
            continue
        method, seed, tk, split = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4)
        text = open(path, errors="ignore").read()
        v, t = last(RE_VALID, text), last(RE_TEST, text)
        lat, mem = last(RE_LAT, text), last(RE_MEM, text)
        row = dict(method=method, seed=seed, topk=tk, split=split, log_path=path,
                   eval_total_ms=float(lat) if lat else None,
                   vram_mb=float(mem) if mem else None)
        if split == "valid" and v and float(v[0]) >= 0:
            row["valid_mrr"] = float(v[0])
        if split == "test" and t and float(t[0]) >= 0:
            row["test_mrr"], row["test_h1"], row["test_h10"] = map(float, t)
        rows.append(row)
    return pd.DataFrame(rows)


def merge_points(df):
    """Gop 2 luot valid+test cua cung (method,seed,topk) thanh 1 diem."""
    va = df[df.split == "valid"][["method", "seed", "topk", "valid_mrr", "log_path"]] \
        .rename(columns={"log_path": "valid_log"})
    te = df[df.split == "test"][["method", "seed", "topk", "test_mrr", "test_h1",
                                 "test_h10", "eval_total_ms", "vram_mb", "log_path"]] \
        .rename(columns={"log_path": "test_log"})
    pts = pd.merge(va, te, on=["method", "seed", "topk"], how="outer")
    missing = pts[pts.valid_mrr.isna() | pts.test_mrr.isna()]
    if len(missing):
        print("!! CANH BAO: cac o thieu du lieu (chay bo sung roi build lai):")
        print(missing[["method", "seed", "topk"]].to_string(index=False))
    return pts


def aggregate(pts, n_test_q):
    g = pts.dropna(subset=["valid_mrr", "test_mrr"]) \
           .groupby(["method", "topk"])
    agg = g.agg(valid_mrr_mean=("valid_mrr", "mean"),
                test_mrr_mean=("test_mrr", "mean"),
                test_mrr_std=("test_mrr", lambda x: x.std(ddof=1)),
                h1_mean=("test_h1", "mean"), h10_mean=("test_h10", "mean"),
                lat_ms_mean=("eval_total_ms", "mean"),
                vram_mean=("vram_mb", "mean"),
                n_seeds=("seed", "nunique")).reset_index()
    agg["lat_per_q_ms"] = agg["lat_ms_mean"] / n_test_q
    return agg


def frontier_2d(agg, subset_methods=None):
    """Trich frontier theo (valid_mrr_mean max, lat_per_q_ms min)."""
    A = agg if subset_methods is None else agg[agg.method.isin(subset_methods)]
    A = A.sort_values("lat_per_q_ms")
    front, best = [], -1
    for _, r in A.iterrows():
        if r.valid_mrr_mean > best:
            front.append(r)
            best = r.valid_mrr_mean
    return pd.DataFrame(front)


def controller(front, max_latency=None, min_valid_mrr=None):
    out = {}
    if max_latency is not None:
        elig = front[front.lat_per_q_ms <= max_latency]
        out["latency_constrained"] = None if elig.empty else \
            elig.loc[elig.valid_mrr_mean.idxmax()].to_dict()
    if min_valid_mrr is not None:
        elig = front[front.valid_mrr_mean >= min_valid_mrr]
        out["accuracy_constrained"] = None if elig.empty else \
            elig.loc[elig.lat_per_q_ms.idxmin()].to_dict()
    return out


def plot_fig(agg, fb, fp, out_png, ds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    colors = {"ppr": "#7f8c8d", "rerank": "#e74c3c", "hybrid": "#2980b9"}
    markers = {"ppr": "o", "rerank": "s", "hybrid": "^"}
    for xcol, ax, xlabel in [("lat_per_q_ms", axes[0], "Latency / query (ms)"),
                             ("vram_mean", axes[1], "Peak VRAM (MB)")]:
        for m in sorted(agg.method.unique()):
            A = agg[agg.method == m].sort_values(xcol)
            ax.errorbar(A[xcol], A.test_mrr_mean, yerr=A.test_mrr_std, fmt=markers.get(m, "o"),
                        color=colors.get(m, "k"), label=METHOD_LABEL.get(m, m),
                        capsize=3, ms=6, alpha=0.9)
            for _, r in A.iterrows():
                ax.annotate(f"θ={int(r.topk*100)}%", (r[xcol], r.test_mrr_mean),
                            textcoords="offset points", xytext=(5, 4), fontsize=7)
        if xcol == "lat_per_q_ms":
            for F, c, ls, lbl in [(fb, "#7f8c8d", "--", "Frontier PPR-only"),
                                   (fp, "#e74c3c", "-",  "Frontier đầy đủ")]:
                if len(F) > 0:
                    Fs = F.sort_values(xcol)
                    ax.step(Fs[xcol], Fs.valid_mrr_mean, where="post", color=c,
                            linestyle=ls, lw=1.6, alpha=0.8, label=lbl)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Test MRR", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].legend(fontsize=8, loc="lower right")
    ds_label = "WN18RR" if ds == "wn18rr" else "NELL-995"
    fig.suptitle(f"PIVOT Pareto Frontier — {ds_label} "
                 f"(frontier chon theo Valid MRR, truc tung = Test MRR)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Da luu figure: {out_png}")


def md_table(df, cols, hdrs):
    lines = ["| " + " | ".join(hdrs) + " |",
             "|" + "|".join([":---:"] * len(hdrs)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", required=True, choices=["wn18rr", "nell"])
    ap.add_argument("--max_latency", type=float, default=35.0)
    ap.add_argument("--min_mrr", type=float, default=0.54)
    args = ap.parse_args()
    ds, n_q = args.dataset, N_TEST_Q[args.dataset]

    df = parse_dir(args.dir)
    if df.empty:
        raise SystemExit("Khong tim thay log nao khop quy uoc ten trong " + args.dir)
    pts = merge_points(df)
    pts.to_csv(os.path.join(args.dir, "pareto_points.csv"), index=False)

    agg = aggregate(pts, n_q).round(6)
    agg.to_csv(os.path.join(args.dir, "pareto_agg.csv"), index=False)
    print("\n=== BANG TONG HOP (pareto_agg) ===")
    print(agg.to_string(index=False))

    fb = frontier_2d(agg, ["ppr"])              # frontier co so (T7-8 thuan)
    fp = frontier_2d(agg)                       # frontier day du (tich hop T7-9)
    ctrl = controller(fp, args.max_latency, args.min_mrr)

    cache = []
    for _, r in pts.dropna(subset=["valid_mrr", "test_mrr"]).iterrows():
        cache.append({"method": METHOD_LABEL.get(r.method, r.method), "budget": r.topk,
                      "seed": int(r.seed), "valid_mrr": r.valid_mrr,
                      "test_mrr": r.test_mrr, "eval_total_ms": r.eval_total_ms,
                      "latency_per_query_ms": r.eval_total_ms / n_q,
                      "vram_mb": r.vram_mb,
                      "valid_log": r.valid_log, "test_log": r.test_log})
    cpath = os.path.join(args.dir, f"pareto_cache_{ds}_v2.json")
    with open(cpath, "w") as f:
        json.dump(cache, f, indent=1)
    print(f"Da luu cache (co log_path): {cpath}")

    plot_fig(agg, fb, fp, os.path.join(args.dir, f"figure1_frontier_{ds}.png"), ds)

    # Bao cao markdown de dan vao §7
    disp = agg.copy()
    disp["Test MRR"] = disp.apply(
        lambda r: f"{r.test_mrr_mean:.4f} ± {r.test_mrr_std:.4f}", axis=1)
    disp["method"] = disp["method"].map(METHOD_LABEL)
    disp["theta"] = (disp.topk * 100).astype(int).astype(str) + "%"
    disp["Valid MRR"] = disp.valid_mrr_mean.map(lambda x: f"{x:.4f}")
    disp["Lat/q (ms)"] = disp.lat_per_q_ms.map(lambda x: f"{x:.2f}")
    disp["VRAM (MB)"] = disp.vram_mean.map(lambda x: f"{x:.0f}")

    rep = [f"## Bang tong hop grid (mean±std, 3 seed, FP32) — {ds.upper()}\n",
           md_table(disp, ["method", "theta", "Valid MRR", "Test MRR",
                           "Lat/q (ms)", "VRAM (MB)"],
                    ["Phương pháp", "θ", "Valid MRR (chọn)", "Test MRR (báo cáo)",
                     "Latency/q (ms)", "VRAM (MB)"]),
           "\n## Frontier co so (PPR-only) — deliverable T7-8 thuan\n"]

    fb_disp = disp[disp["theta"].isin(fb["topk"].apply(lambda x: f"{int(x*100)}%").values) &
                   (disp["method"] == "PPR-only")]
    if len(fb_disp):
        rep.append(md_table(fb_disp,
                    ["theta", "Valid MRR", "Test MRR", "Lat/q (ms)"],
                    ["θ", "Valid MRR", "Test MRR", "Latency/q (ms)"]))
    else:
        rep.append("*(frontier trống — kiểm tra log)*")

    rep.append("\n## Frontier day du — deliverable tich hop T7-9\n")
    fp_methods = fp["method"].map(METHOD_LABEL).values
    fp_topks   = fp["topk"].apply(lambda x: f"{int(x*100)}%").values
    fp_rows = []
    for mi, ti in zip(fp_methods, fp_topks):
        sub = disp[(disp["method"] == mi) & (disp["theta"] == ti)]
        fp_rows.append(sub)
    fp_disp = pd.concat(fp_rows) if fp_rows else pd.DataFrame()
    if len(fp_disp):
        rep.append(md_table(fp_disp,
                    ["method", "theta", "Valid MRR", "Test MRR", "Lat/q (ms)", "VRAM (MB)"],
                    ["Phương pháp", "θ", "Valid MRR", "Test MRR", "Latency/q (ms)", "VRAM (MB)"]))
    else:
        rep.append("*(frontier trống — kiểm tra log)*")

    rep.append("\n## Demo BudgetController (chon theo Valid, bao cao Test)\n")
    for k, v in ctrl.items():
        if v is None:
            rep.append(f"- **{k}**: khong co cau hinh thoa man rang buoc.")
        else:
            meth = v.get("method", "")
            topk = v.get("topk", 0)
            rep.append(f"- **{k}**: {METHOD_LABEL.get(meth, meth)} @ "
                       f"θ={int(topk*100)}% → Valid {v['valid_mrr_mean']:.4f}, "
                       f"Test {v['test_mrr_mean']:.4f} ± {v['test_mrr_std']:.4f}, "
                       f"{v['lat_per_q_ms']:.2f} ms/q, {v['vram_mean']:.0f} MB")

    rpath = os.path.join(args.dir, f"frontier_report_{ds}.md")
    with open(rpath, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print(f"Da luu bao cao markdown: {rpath}")


if __name__ == "__main__":
    main()
