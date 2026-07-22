#!/bin/bash
# apply_restructure.sh — chạy TRƯỚC khi giải nén zip đè, từ thư mục gốc repo.
# Dùng git mv để GIỮ LỊCH SỬ file. Sau khi chạy xong: unzip -o PIVOT_pro_restructure.zip
set -e
mkdir -p experiments analysis docs reports/artifacts/logs
git mv run_learned_pruning_wn18rr.py run_learned_pruning_nell.py \
       budgeted_protocol.py make_perturbed_datasets_v2.py \
       sweep_alpha_wn18rr.py sweep_alpha_nell.py \
       run_grid_t78.sh run_robustness_t10.sh experiments/
git mv build_pareto.py build_robustness.py pareto_optimizer.py showResults.py analysis/
git mv PIVOT.pdf docs/
git mv grid_wn18rr_master.log reports/artifacts/logs/
git rm -q sweep_alpha_wn18rr_fact95.py   # đã gộp vào experiments/sweep_alpha_wn18rr.py (--fact_ratio)
echo "Moves xong. Bây giờ: unzip -o PIVOT_pro_restructure.zip && git add -A && git commit"
