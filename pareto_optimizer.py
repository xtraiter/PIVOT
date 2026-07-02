import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class BudgetController:
    def __init__(self, cache_path: str):
        """
        Initializes the BudgetController by loading the configuration cache.
        """
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache file not found at {self.cache_path}")
        
        with open(self.cache_path, "r") as f:
            self.configs = json.load(f)
            
        # Standardize field names just in case
        for cfg in self.configs:
            if "latency_per_query_ms" not in cfg and "latency" in cfg:
                cfg["latency_per_query_ms"] = cfg["latency"]
            if "MRR" not in cfg and "mrr" in cfg:
                cfg["MRR"] = cfg["mrr"]
            if "layer" not in cfg and "L" in cfg:
                cfg["layer"] = cfg["L"]

        self.frontier = self._compute_pareto_frontier()

    def _compute_pareto_frontier(self):
        """
        Computes the Pareto frontier (non-dominated points) from the loaded configurations.
        We want to maximize MRR and minimize latency.
        """
        frontier = []
        for i, c1 in enumerate(self.configs):
            dominated = False
            for j, c2 in enumerate(self.configs):
                if i == j:
                    continue
                # c2 dominates c1 if:
                # - c2.MRR >= c1.MRR and c2.latency <= c1.latency
                # - and at least one inequality is strict
                if (c2["MRR"] >= c1["MRR"] and c2["latency_per_query_ms"] <= c1["latency_per_query_ms"]) and \
                   (c2["MRR"] > c1["MRR"] or c2["latency_per_query_ms"] < c1["latency_per_query_ms"]):
                    dominated = True
                    break
            if not dominated:
                frontier.append(c1)
        
        # Sort frontier by latency
        frontier = sorted(frontier, key=lambda x: x["latency_per_query_ms"])
        return frontier

    def get_best_accuracy_under_latency(self, max_latency: float):
        """
        Returns the configuration that maximizes MRR subject to latency <= max_latency.
        """
        eligible = [c for c in self.configs if c["latency_per_query_ms"] <= max_latency]
        if not eligible:
            # If no config is under max_latency, return the one with the absolute minimum latency
            best_cfg = min(self.configs, key=lambda x: x["latency_per_query_ms"])
            print(f"[Warning] No configuration satisfies latency <= {max_latency}ms. "
                  f"Returning minimum latency configuration.")
            return best_cfg, False
        
        best_cfg = max(eligible, key=lambda x: x["MRR"])
        return best_cfg, True

    def get_min_latency_under_mrr(self, min_mrr: float):
        """
        Returns the configuration that minimizes latency subject to MRR >= min_mrr.
        """
        eligible = [c for c in self.configs if c["MRR"] >= min_mrr]
        if not eligible:
            # If no config is above min_mrr, return the one with the absolute maximum MRR
            best_cfg = max(self.configs, key=lambda x: x["MRR"])
            print(f"[Warning] No configuration satisfies MRR >= {min_mrr}. "
                  f"Returning maximum MRR configuration.")
            return best_cfg, False
        
        best_cfg = min(eligible, key=lambda x: x["latency_per_query_ms"])
        return best_cfg, True

    def plot_frontier(self, out_png_path: str, dataset_name: str = "WN18RR"):
        """
        Generates a beautiful Pareto frontier visualization.
        """
        out_png_path = Path(out_png_path)
        out_png_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        # Plot all configurations in light grey
        all_latencies = [c["latency_per_query_ms"] for c in self.configs]
        all_mrrs = [c["MRR"] for c in self.configs]
        ax.scatter(all_latencies, all_mrrs, color="lightgray", alpha=0.6, label="All Configurations (Grid Search)")

        # Plot the Pareto frontier points and connect them
        front_latencies = [c["latency_per_query_ms"] for c in self.frontier]
        front_mrrs = [c["MRR"] for c in self.frontier]
        ax.plot(front_latencies, front_mrrs, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.scatter(front_latencies, front_mrrs, color="#e74c3c", s=60, edgecolor="black", zorder=5, label="Pareto Frontier (PIVOT)")

        # Annotate Pareto frontier points
        for i, cfg in enumerate(self.frontier):
            label = f"({cfg['alpha']:.2f}, {cfg['beta']:.2f}, L={cfg['layer']}, b={cfg['budget']:.2f})"
            # Alternate annotation offsets to prevent overlaps
            xytext = (10, -5) if i % 2 == 0 else (10, 5)
            ax.annotate(
                label,
                (cfg["latency_per_query_ms"], cfg["MRR"]),
                textcoords="offset points",
                xytext=xytext,
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.5, alpha=0.5)
            )

        ax.set_xlabel("Latency per Query (ms)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Accuracy (MRR)", fontsize=11, fontweight="bold")
        ax.set_title(f"PIVOT Accuracy-Latency Pareto Frontier ({dataset_name})", fontsize=13, fontweight="bold", pad=15)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(out_png_path)
        plt.close()
        print(f"Pareto frontier plot saved successfully to {out_png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIVOT Budget Controller & Pareto Optimizer")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to pareto cache JSON file")
    parser.add_argument("--max_latency", type=float, default=None, help="Query: Best accuracy under latency <= max_latency")
    parser.add_argument("--min_mrr", type=float, default=None, help="Query: Min latency under MRR >= min_mrr")
    parser.add_argument("--plot_path", type=str, default=None, help="Path to save the Pareto frontier plot")
    parser.add_argument("--dataset_name", type=str, default="WN18RR", help="Dataset name to show in the plot title")
    args = parser.parse_args()

    controller = BudgetController(args.cache_path)

    print(f"\n=======================================================")
    print(f"      PIVOT PARETO CONTROLLER - DATASET: {args.dataset_name}")
    print(f"=======================================================")
    print(f"Total configurations analyzed: {len(controller.configs)}")
    print(f"Pareto frontier points: {len(controller.frontier)}")
    for i, pt in enumerate(controller.frontier):
        print(f"  Point {i+1}: MRR={pt['MRR']:.4f} | Latency={pt['latency_per_query_ms']:.2f}ms | Config: (alpha={pt['alpha']}, beta={pt['beta']}, layer={pt['layer']}, budget={pt['budget']})")
    
    if args.max_latency is not None:
        best_cfg, ok = controller.get_best_accuracy_under_latency(args.max_latency)
        print(f"\n--> Query: Best Accuracy under Latency <= {args.max_latency}ms")
        print(f"    Success: {ok}")
        print(f"    Selected config: alpha={best_cfg['alpha']}, beta={best_cfg['beta']}, layer={best_cfg['layer']}, budget={best_cfg['budget']}")
        print(f"    Metrics: MRR={best_cfg['MRR']:.4f} | Latency={best_cfg['latency_per_query_ms']:.2f}ms")

    if args.min_mrr is not None:
        best_cfg, ok = controller.get_min_latency_under_mrr(args.min_mrr)
        print(f"\n--> Query: Min Latency under MRR >= {args.min_mrr}")
        print(f"    Success: {ok}")
        print(f"    Selected config: alpha={best_cfg['alpha']}, beta={best_cfg['beta']}, layer={best_cfg['layer']}, budget={best_cfg['budget']}")
        print(f"    Metrics: MRR={best_cfg['MRR']:.4f} | Latency={best_cfg['latency_per_query_ms']:.2f}ms")

    if args.plot_path is not None:
        controller.plot_frontier(args.plot_path, args.dataset_name)
