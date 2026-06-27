#!/bin/bash

# Wait for Seed 123 (PID 72078) to finish
TARGET_PID=72078
echo "Waiting for PIVOT Seed 123 (PID $TARGET_PID) to complete..."
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep 30
done
echo "Seed 123 completed. Starting Seed 1234..."

# Run Seed 1234
/home/vanba/miniconda3/envs/pivot/bin/python3 budgeted_protocol.py \
  --data_path ./data/WN18RR/ \
  --weight ./data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.565_seed1234.pt \
  --gpu 0 \
  --n_queries 5716 \
  --budgets 0.01 0.05 0.10 0.20 \
  --dataset WN18RR \
  --method "PIVOT-Seed1234" \
  --outdir ./data/WN18RR/budget_results/seed_1234

echo "Seed 1234 completed. Starting PPR-only baseline (layer 6)..."

# Run PPR-only baseline (layer 6)
/home/vanba/miniconda3/envs/pivot/bin/python3 budgeted_protocol.py \
  --data_path ./data/WN18RR/ \
  --weight ./savedModels/WN18RR_topk_0.1_layer_6_ValMRR_0.569.pt \
  --gpu 0 \
  --n_queries 5716 \
  --budgets 0.01 0.05 0.10 0.20 \
  --dataset WN18RR \
  --method "PPR-only" \
  --outdir ./data/WN18RR/budget_results/baseline

echo "Baseline completed. Running aggregation script..."

# Run aggregation script
/home/vanba/miniconda3/envs/pivot/bin/python3 aggregate_seeds.py

echo "All benchmarks completed!"
