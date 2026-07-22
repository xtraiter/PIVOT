#!/bin/bash
# ============================================================
# TUAN 10: ROBUSTNESS SUITE — WN18RR (FP32, test-only, resume-safe)
#
#   bash run_robustness_t10.sh sanity   -> 1 luot kiem tra tren del10
#   bash run_robustness_t10.sh run      -> toan bo config x {ppr,rerank} x 3 seed
#
# Diem clean KHONG chay lai — tai dung tu reports/grid_t78_wn18rr (tk0.1).
# LUU Y: luot DAU TIEN cua moi config se kich hoat PPR precompute (lau,
# hang chuc phut den 1-2 gio — DAY LA HANH VI DUNG, KHONG kill process).
# Resume-safe: gian doan thi chay lai cung lenh, skip log da co [TEST].
# ============================================================
set -e
PY=${PY:-python3}
MODE=${1:-run}
OUT=reports/robustness_t10
mkdir -p $OUT

# alpha* per-seed tu Tuan 9 (FP32, valid-chosen), KHONG tune lai tren do thi nhieu
declare -A ALPHA=( [42]=0.8 [123]=0.6 [1234]=0.7 )

ckpt_of() {
    ls data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_*_seed${1}.pt 2>/dev/null | head -1
}
mlp_of() {
    echo data/WN18RR/budget_results/pruning_mlp_v2_best_seed_${1}.pt
}

# Tu phat hien cac thu muc nhieu da tao (WN18RR_del* hoac WN18RR_reldel)
CONFIGS=$(ls -d data/WN18RR_del* data/WN18RR_reldel 2>/dev/null \
          | xargs -n1 basename 2>/dev/null || true)
if [ -z "$CONFIGS" ]; then
    echo "LOI: chua co thu muc nhieu — chay experiments/make_perturbed_datasets_v2.py truoc."
    exit 2
fi
echo "Cac config phat hien: $CONFIGS"

run_one() {  # $1=config_dirname $2=method $3=seed
    local CFG=$1 M=$2 S=$3
    local LOG=$OUT/rob_${CFG}_${M}_s${S}.log
    if [ -s "$LOG" ] && grep -q "\[TEST\]" "$LOG"; then
        echo "SKIP (da co [TEST]): $LOG"; return
    fi
    local CKPT; CKPT=$(ckpt_of $S)
    if [ -z "$CKPT" ]; then
        echo "LOI: khong tim thay checkpoint seed $S"; exit 4
    fi
    local EXTRA=""
    if [ "$M" == "rerank" ] || [ "$M" == "hybrid" ]; then
        local MLP; MLP=$(mlp_of $S)
        if [ ! -f "$MLP" ]; then
            echo "LOI: khong tim thay MLP checkpoint $MLP"; exit 5
        fi
        EXTRA="--pruning_model_path $MLP --rerank_alpha ${ALPHA[$S]}"
        if [ "$M" == "hybrid" ]; then
            EXTRA="$EXTRA --use_learned_pruning"
        fi
    fi
    echo "===== [$CFG] method=$M seed=$S ====="
    $PY train_auto.py \
        --data_path ./data/${CFG}/ \
        --batchsize 16 \
        --only_eval \
        --no_amp \
        --gpu 0 \
        --seed $S \
        --weight "$CKPT" \
        --topk 0.1 \
        --topm -1 \
        --eval_split test \
        $EXTRA 2>&1 | tee "$LOG"
}

# ── SANITY MODE ──
if [ "$MODE" == "sanity" ]; then
    CFG=$(echo $CONFIGS | tr ' ' '\n' | grep del10 | head -1)
    [ -z "$CFG" ] && CFG=$(echo $CONFIGS | awk '{print $1}')
    echo "=== Sanity check tren: $CFG ==="
    echo "LUU Y: luot nay se kich hoat PPR precompute — co the lau hang chuc phut den 1-2h."
    run_one $CFG ppr 42

    echo "---- KIEM TRA SANITY ----"
    N_PPR=$(ls data/${CFG}/ppr_scores/*.pkl 2>/dev/null | wc -l)
    MRR=$(grep -oP '\[TEST\] MRR:\K[\-\d.]+' $OUT/rob_${CFG}_ppr_s42.log 2>/dev/null | tail -1)
    FLAG=$(grep -oP 'use_learned_pruning=\K[A-Za-z]+' $OUT/rob_${CFG}_ppr_s42.log 2>/dev/null | head -1)

    echo "ppr_scores files trong $CFG: $N_PPR (CAN = 40943 — cache MOI, khong phai copy tu WN18RR sach)"
    echo "Test MRR seed42: $MRR (CAN < 0.5647 — thap hon clean seed42=0.5643)"
    echo "use_learned_pruning flag: $FLAG (CAN = False)"

    FAIL=0
    [ "$N_PPR" != "40943" ] && { echo "SANITY FAIL: PPR cache sai so luong ($N_PPR != 40943)."; FAIL=1; }
    [ -z "$MRR" ] && { echo "SANITY FAIL: khong doc duoc Test MRR."; FAIL=1; }
    if [ "$FAIL" == "0" ]; then
        python3 -c "
import sys
mrr=float('${MRR}')
if mrr >= 0.5647:
    print(f'SANITY FAIL: MRR {mrr:.4f} >= 0.5647 (clean) — nghi lan cache PPR sach!')
    sys.exit(3)
print('SANITY PASS.')
"
    else
        exit 3
    fi
    echo "Chay tiep: bash run_robustness_t10.sh run"
    exit 0
fi

# ── SANITY HYBRID MODE (tren clean) ──
if [ "$MODE" == "sanity_hybrid" ]; then
    echo "=== Sanity check Hybrid tren WN18RR (clean) ==="
    run_one WN18RR hybrid 42
    
    MRR=$(grep -oP '\[TEST\] MRR:\K[\-\d.]+' $OUT/rob_WN18RR_hybrid_s42.log 2>/dev/null | tail -1)
    echo "Test MRR seed42: $MRR (CAN = ~0.5697)"
    
    python3 -c "
import sys
mrr=float('${MRR}')
if abs(mrr - 0.5697) > 0.003:
    print(f'SANITY FAIL: MRR {mrr:.4f} lech qua xa 0.5697.')
    sys.exit(3)
print('SANITY PASS.')
"
    exit 0
fi

# ── RUN MODE ──
echo "=== Bat dau chay day du: $(echo $CONFIGS | wc -w) config x 2 method x 3 seed ==="
for CFG in $CONFIGS; do
    for S in 42 123 1234; do
        for M in ppr rerank hybrid; do
            run_one $CFG $M $S
        done
    done
done
echo ""
echo "XONG TAT CA. Kiem tra so log:"
ls $OUT/rob_*.log 2>/dev/null | wc -l
echo "Tiep: python3 analysis/build_robustness.py --dir $OUT --clean_dir reports/grid_t78_wn18rr"
