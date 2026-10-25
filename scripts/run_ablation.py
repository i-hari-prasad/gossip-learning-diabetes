"""
run_ablation.py
---------------
Ablation study comparing:
  1. Energy-aware gossip learning (proposed)
  2. Conventional gossip (no energy threshold)

Also compares topologies: ring, fully_connected, random, small_world
"""

import os
import sys
import json
import pickle
import logging
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.train_lglm import train_gossip

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def run_ablation(processed: dict, config: dict):
    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts/plots", exist_ok=True)

    ablation_results = {}

    # -- Study 1: Energy-Aware vs Conventional --
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION 1: Energy-Aware vs Conventional Gossip")
    logger.info("=" * 60)

    for mode, energy_aware in [("energy_aware", True), ("conventional", False)]:
        logger.info(f"\nRunning mode: {mode}")
        r = train_gossip(processed, config, energy_aware=energy_aware, run_tag=f"ablation_{mode}")
        ablation_results[mode] = {
            'overall': r['overall'],
            'energy_stats': {pid: r['per_patient'].get(pid, {}).get('energy', {})
                             for pid in processed.keys()}
        }

    # -- Study 2: Topology comparison --
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION 2: Topology Comparison")
    logger.info("=" * 60)

    topologies = ['ring', 'fully_connected', 'random', 'small_world']
    topo_results = {}
    for topo in topologies:
        logger.info(f"\nTopology: {topo}")
        r = train_gossip(processed, config, topology_type=topo,
                         energy_aware=True, run_tag=f"ablation_topo_{topo}")
        topo_results[topo] = r['overall']

    # -- Save combined results --
    combined = {
        'energy_ablation': ablation_results,
        'topology_ablation': topo_results
    }
    with open("results/ablation_results.json", 'w') as f:
        json.dump(combined, f, indent=2, default=str)

    # -- Visualize ablation --
    _plot_energy_ablation(ablation_results)
    _plot_topology_comparison(topo_results)

    # -- Print summary table --
    print("\n" + "=" * 70)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"  {'Mode':<22} {'RMSE':>8} {'MAE':>8} {'MARD%':>8} {'R²':>8}")
    print("-" * 70)
    for mode, data in ablation_results.items():
        m = data['overall']
        print(f"  {mode:<22} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['MARD']:>8.2f} {m['R2']:>8.4f}")
    print()
    for topo, m in topo_results.items():
        print(f"  topo:{topo:<17} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['MARD']:>8.2f} {m['R2']:>8.4f}")
    print("=" * 70)
    return combined


def _plot_energy_ablation(ablation_results: dict):
    modes = list(ablation_results.keys())
    metrics = ['RMSE', 'MAE', 'R2']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#1a1a2e')
    colors = ['#00d2ff', '#f7971e']
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor('#16213e')
        vals = [ablation_results[m]['overall'].get(metric, 0) for m in modes]
        bars = ax.bar(modes, vals, color=colors[:len(modes)], edgecolor='#555577', width=0.5)
        ax.set_title(metric, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.set_ylabel(metric, color='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', color='white', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
    plt.suptitle("Energy-Aware vs Conventional Gossip", color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("artifacts/plots/ablation_energy.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_topology_comparison(topo_results: dict):
    topos = list(topo_results.keys())
    metrics = ['RMSE', 'MAE', 'R2']
    x = np.arange(len(topos))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#1a1a2e')
    palette = plt.cm.cool(np.linspace(0.2, 0.9, len(topos)))
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor('#16213e')
        vals = [topo_results[t].get(metric, 0) for t in topos]
        bars = ax.bar(x, vals, color=palette, edgecolor='#555577', width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(topos, rotation=20, color='white', fontsize=9)
        ax.set_title(metric, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', color='white', fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')
    plt.suptitle("Topology Comparison", color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("artifacts/plots/ablation_topology.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reproduction.yaml")
    parser.add_argument("--data_pkl", default="artifacts/processed_data.pkl")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.data_pkl, 'rb') as f:
        processed = pickle.load(f)

    run_ablation(processed, config)
