"""
make_perturbed_datasets_v2.py — Tuan 10 (ban sua loi thiet ke)
==============================================================
KHAC BAN CU: ban cu chi xoa canh trong train.txt, nhung do thi quan sat cua
repo nay la facts.txt union train.txt (load_data.py: all_triple = facts + train,
test_sampler xay tu do thi nay). Ban v2 xoa theo CUNG TY LE tren HOP CA HAI
FILE, roi ghi phan song sot ve dung file goc cua no.

Cau hinh nhieu:
  del05 / del10 / del20 : xoa ngau nhien 5/10/20% trieu cua hop facts∪train
  reldel                : xoa 50% trieu thuoc 30% relation HIEM nhat (theo
                          tan suat tren hop facts∪train)

Bat bien:
  - entities.txt / relations.txt / valid.txt / test.txt copy NGUYEN VEN
    (vocab khong doi -> checkpoint GNN & MLP cu van load duoc)
  - KHONG copy bat ky cache nao (ppr_scores/, saveModel/, results/,
    budget_results/, *.pkl) -> PPR se duoc tinh lai tren do thi nhieu

Chay:
  python3 make_perturbed_datasets_v2.py --data_path ./data/WN18RR --seed 42 \
      --configs del05 del10 del20 reldel
"""

import argparse, os, random, shutil
from collections import Counter

CORE_COPY = ["entities.txt", "relations.txt", "valid.txt", "test.txt"]


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def rel_of(line):
    return line.split()[1]          # h r t phan tach bang whitespace


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def make_dir(src, dst):
    if os.path.exists(dst):
        raise SystemExit(f"LOI: {dst} da ton tai — xoa thu muc cu truoc khi tao lai "
                         f"(tranh lan cache PPR cu).")
    os.makedirs(dst)
    for name in CORE_COPY:
        src_file = os.path.join(src, name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dst, name))


def apply_deletion(facts, train, del_idx):
    nf = len(facts)
    keep_f = [facts[i] for i in range(nf) if i not in del_idx]
    keep_t = [train[j - nf] for j in range(nf, nf + len(train)) if j not in del_idx]
    return keep_f, keep_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="./data/WN18RR")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="+",
                    default=["del05", "del10", "del20", "reldel"],
                    choices=["del05", "del10", "del20", "reldel"])
    ap.add_argument("--rel_bottom_frac", type=float, default=0.30)
    ap.add_argument("--rel_del_ratio", type=float, default=0.50)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = args.data_path.rstrip("/")
    base, parent = os.path.basename(src), os.path.dirname(src)
    if not parent:
        parent = "."

    facts_path = os.path.join(src, "facts.txt")
    train_path = os.path.join(src, "train.txt")
    if not os.path.exists(facts_path):
        raise SystemExit(f"LOI: {facts_path} khong ton tai.")
    if not os.path.exists(train_path):
        raise SystemExit(f"LOI: {train_path} khong ton tai.")

    facts = load_lines(facts_path)
    train = load_lines(train_path)
    n_union = len(facts) + len(train)
    print(f"[goc] facts={len(facts)}  train={len(train)}  union={n_union}")

    summary = [f"seed={args.seed}",
               f"goc: facts={len(facts)} train={len(train)} union={n_union}",
               ""]

    for cfg in args.configs:
        dst = os.path.join(parent, f"{base}_{cfg}")
        make_dir(src, dst)

        if cfg.startswith("del"):
            ratio = int(cfg[3:]) / 100.0
            n_del = int(n_union * ratio)
            del_idx = set(rng.sample(range(n_union), n_del))
            kind = f"random {ratio:.0%} tren facts+train"
        else:  # reldel
            rel_count = Counter(rel_of(l) for l in facts + train)
            rels_sorted = sorted(rel_count.items(), key=lambda x: x[1])
            n_target = max(1, int(len(rels_sorted) * args.rel_bottom_frac))
            targets = set(r for r, _ in rels_sorted[:n_target])
            union = facts + train
            tgt_idx = [i for i, l in enumerate(union) if rel_of(l) in targets]
            n_del = int(len(tgt_idx) * args.rel_del_ratio)
            del_idx = set(rng.sample(tgt_idx, n_del))
            kind = (f"rel-specific: {n_target} rel hiem nhat ({len(tgt_idx)} trieu), "
                    f"xoa {args.rel_del_ratio:.0%} trong do")
            summary.append(f"[{cfg}] target_rels ({n_target}): "
                           f"{sorted(targets)[:5]}{'...' if n_target>5 else ''}")

        keep_f, keep_t = apply_deletion(facts, train, del_idx)
        write_lines(os.path.join(dst, "facts.txt"), keep_f)
        write_lines(os.path.join(dst, "train.txt"), keep_t)

        # Kiem tra test.txt khong bi anh huong
        import filecmp
        test_ok = filecmp.cmp(os.path.join(src, "test.txt"),
                               os.path.join(dst, "test.txt"), shallow=False)

        line = (f"[{base}_{cfg}]\t{kind}"
                f"\tdeleted={n_del} ({n_del/n_union:.1%})"
                f"\tfacts_kept={len(keep_f)}"
                f"\ttrain_kept={len(keep_t)}"
                f"\tunion_kept={len(keep_f)+len(keep_t)}"
                f"\ttest_intact={test_ok}")
        print(line)
        summary.append(line)

    spath = os.path.join(parent, f"{base}_perturbation_summary_v2.txt")
    write_lines(spath, summary)
    print(f"\nXong. Summary: {spath}")
    print("\n*** NHAC QUAN TRONG ***")
    print("KHONG copy ppr_scores/ vao cac thu muc moi —")
    print("PPR phai duoc tinh lai tu dau tren do thi nhieu.")
    print("Lan chay dau tien cua moi config se LÂU (precompute ~1-2h).")
    print("Day la hanh vi DUNG — khong kill process.")


if __name__ == "__main__":
    main()
