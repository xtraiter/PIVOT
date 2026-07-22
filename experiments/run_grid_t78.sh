#!/bin/bash
# ============================================================
# TUAN 7-8: GRID CAMPAIGN (FP32, resume-safe)
#   bash run_grid_t78.sh sanity    -> 2 luot kiem tra co
#   bash run_grid_t78.sh wn18rr    -> 72 luot (~4h)
#   bash run_grid_t78.sh nell      -> 48 luot (~4h)  (them hybrid: INCLUDE_HYBRID=1)
# Log: reports/grid_t78_<ds>/{method}_s{seed}_tk{topk}_{split}.log
# ============================================================
set -e
PY=/home/vanba/miniconda3/envs/pivot/bin/python3
DS=${1:-sanity}
REPO=/home/vanba/KLTN/one-shot-subgraph
cd "$REPO"

if [ "$DS" == "wn18rr" ] || [ "$DS" == "sanity" ]; then
    DATA=./data/WN18RR/;  OUT=reports/grid_t78_wn18rr
    CKPT_GLOB='data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_*_seed%s.pt'
    MLP_DIR=data/WN18RR/budget_results
    declare -A ALPHA=( [42]=0.8 [123]=0.6 [1234]=0.7 )
    METHODS="ppr rerank hybrid"
    BS=16
fi

if [ "$DS" == "nell" ]; then
    DATA=./data/nell/;    OUT=reports/grid_t78_nell
    CKPT_GLOB='data/nell/saveModel/topk_0.1_layer_8_ValMRR_*_seed%s.pt'
    MLP_DIR=data/nell/budget_results
    declare -A ALPHA=( [42]=0.8 [123]=0.6 [1234]=0.9 )
    METHODS="ppr rerank"
    [ "$INCLUDE_HYBRID" == "1" ] && METHODS="ppr rerank hybrid"
    BS=8
fi

if [ "$DS" != "wn18rr" ] && [ "$DS" != "nell" ] && [ "$DS" != "sanity" ]; then
    echo "Usage: bash run_grid_t78.sh [sanity|wn18rr|nell]"; exit 1
fi

mkdir -p $OUT

ckpt_of() {
    local S="$1"
    # Glob for seed-specific checkpoint, excluding mlp and hybrid joint variants
    ls $(printf "$CKPT_GLOB" "$S") 2>/dev/null | grep -v mlp | grep -v hybrid | head -1
}
mlp_of()  { echo $MLP_DIR/pruning_mlp_v2_best_seed_$1.pt; }

run_one() {  # $1=method $2=seed $3=topk $4=split
    local M=$1 S=$2 TK=$3 SP=$4
    local LOG=$OUT/${M}_s${S}_tk${TK}_${SP}.log
    if [ -s "$LOG" ] && grep -q "\[VALID\]" "$LOG"; then
        echo "SKIP (da co): $LOG"; return; fi
    local CKPT=$(ckpt_of $S)
    [ -z "$CKPT" ] && { echo "LOI: khong tim thay checkpoint seed $S"; exit 2; }
    local EXTRA=""
    case $M in
        ppr)    EXTRA="" ;;
        rerank) EXTRA="--pruning_model_path $(mlp_of $S) --rerank_alpha ${ALPHA[$S]}" ;;
        hybrid) EXTRA="--use_learned_pruning --pruning_model_path $(mlp_of $S) --rerank_alpha ${ALPHA[$S]}" ;;
    esac
    echo "===== [$M] seed=$S topk=$TK split=$SP ====="
    $PY train_auto.py --data_path $DATA --batchsize $BS --only_eval --no_amp \
        --gpu 0 --seed $S --weight "$CKPT" --topk $TK --topm -1 \
        --eval_split $SP $EXTRA 2>&1 | tee "$LOG"
}

if [ "$DS" == "sanity" ]; then
    mkdir -p reports/grid_t78_wn18rr
    run_one rerank 42 0.05 valid
    run_one hybrid 42 0.05 valid
    echo "---- KIEM TRA SANITY ----"
    # train_auto.py khong in Namespace; kiem tra bang [VALID] va VRAM
    R_VALID=$(grep -c '\[VALID\]' reports/grid_t78_wn18rr/rerank_s42_tk0.05_valid.log || echo 0)
    H_VALID=$(grep -c '\[VALID\]' reports/grid_t78_wn18rr/hybrid_s42_tk0.05_valid.log || echo 0)
    RM=$(grep -oP '\[PEAK_GPU_MEM\] \K[\d.]+' reports/grid_t78_wn18rr/rerank_s42_tk0.05_valid.log | tail -1)
    HM=$(grep -oP '\[PEAK_GPU_MEM\] \K[\d.]+' reports/grid_t78_wn18rr/hybrid_s42_tk0.05_valid.log | tail -1)
    echo "rerank: [VALID] count=$R_VALID (PHAI >= 1) | VRAM=${RM} MB"
    echo "hybrid: [VALID] count=$H_VALID (PHAI >= 1) | VRAM=${HM} MB (PHAI khac rerank)"
    # Kiem tra co [VALID] va VRAM lay duoc
    if [ "$R_VALID" -lt 1 ] || [ "$H_VALID" -lt 1 ]; then
        echo "SANITY FAIL: thieu [VALID] trong log."; exit 3; fi
    if [ -z "$RM" ] || [ -z "$HM" ]; then
        echo "SANITY FAIL: khong lay duoc VRAM tu log."; exit 3; fi
    # Kiem tra VRAM hai luot khac nhau (chung to hybrid pruning hoat dong)
    if [ "$RM" == "$HM" ]; then
        echo "SANITY FAIL: VRAM rerank=$RM == hybrid=$HM, co the hybrid khong dung --use_learned_pruning."; exit 3
    fi
    echo "SANITY PASS. rerank VRAM=$RM MB, hybrid VRAM=$HM MB (khac nhau = OK)."
    exit 0
fi

for S in 42 123 1234; do
  for TK in 0.01 0.05 0.1 0.2; do
    for M in $METHODS; do
      for SP in valid test; do
        run_one $M $S $TK $SP
      done
    done
  done
done
echo "XONG grid $DS. Tiep: $PY build_pareto.py --dir $OUT --dataset $DS"
