"""
reproduce_all.py
----------------
Master orchestration script — runs the full reproduction pipeline:

  Step 1: Prepare data (generate synthetic + preprocess)
  Step 2: Train gossip LGLM
  Step 3: Train centralized LSTM baseline
  Step 4: Evaluate LGLM + generate Clarke Error Grid
  Step 5: Evaluate baseline
  Step 6: Generate comparison report
  [Optional] Step 7: Run ablation studies

Usage:
  python scripts/reproduce_all.py
  python scripts/reproduce_all.py --skip_ablation
  python scripts/reproduce_all.py --config config/reproduction.yaml
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import yaml
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/reproduce_all.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


def banner(msg: str):
    border = "=" * (len(msg) + 4)
    print(f"\n{border}")
    print(f"  {msg}")
    print(f"{border}\n")


def main(args):
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ==============================================
    # STEP 1: DATA PREPARATION
    # ==============================================
    banner("STEP 1: Data Preparation")
    from scripts.prepare_data import main as prepare_main

    class PrepareArgs:
        config = args.config
        output = "artifacts/processed_data.pkl"

    prepare_main(PrepareArgs())

    with open("artifacts/processed_data.pkl", 'rb') as f:
        processed = pickle.load(f)
    logger.info(f"Loaded {len(processed)} patients.")

    # ==============================================
    # STEP 2: TRAIN GOSSIP LGLM (30-min horizon)
    # ==============================================
    banner("STEP 2: Gossip Learning - LGLM Training")
    from scripts.train_lglm import train_gossip
    gossip_results = train_gossip(processed, config, run_tag="lglm", energy_aware=True)

    # ==============================================
    # STEP 3: TRAIN CENTRALIZED LSTM
    # ==============================================
    banner("STEP 3: Centralized LSTM Baseline")
    from scripts.train_baselines import train_centralized
    baseline_results = train_centralized(processed, config, run_tag="centralized_lstm")

    # ==============================================
    # STEP 4: EVALUATE LGLM
    # ==============================================
    banner("STEP 4: Evaluate LGLM")
    from scripts.evaluate import evaluate_all
    lglm_eval = evaluate_all(processed, config, run_tag="lglm")

    # ==============================================
    # STEP 5: EVALUATE BASELINE (using centralized model artifacts)
    # ==============================================
    banner("STEP 5: Evaluate Centralized LSTM")
    # The baseline results already contain test metrics — format and display
    bl_overall = baseline_results.get('overall', {})
    print(f"  Baseline RMSE: {bl_overall.get('RMSE', 'N/A'):.4f} mg/dL")
    print(f"  Baseline MAE:  {bl_overall.get('MAE', 'N/A'):.4f} mg/dL")
    print(f"  Baseline MARD: {bl_overall.get('MARD', 'N/A'):.2f}%")
    print(f"  Baseline R²:   {bl_overall.get('R2', 'N/A'):.4f}")
    with open("results/centralized_lstm_evaluation.json", 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)

    # ==============================================
    # STEP 6: FINAL COMPARISON REPORT
    # ==============================================
    banner("STEP 6: Final Comparison Report")
    _generate_report(gossip_results, baseline_results, lglm_eval, config)

    # ==============================================
    # STEP 7: ABLATION (optional)
    # ==============================================
    if not args.skip_ablation:
        banner("STEP 7: Ablation Studies")
        from scripts.run_ablation import run_ablation
        run_ablation(processed, config)
    else:
        logger.info("Ablation skipped (--skip_ablation flag set)")

    elapsed = time.time() - t0
    banner(f"REPRODUCTION COMPLETE  ({elapsed:.0f}s)")
    print("  Artifacts saved in:  artifacts/plots/")
    print("  Metrics saved in:    artifacts/metrics/ and results/")
    print("  Models saved in:     artifacts/models/")
    print()
    print("  Run the dashboard:   streamlit run dashboard/app.py")


def _generate_report(gossip_results, baseline_results, lglm_eval, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    g_overall = gossip_results.get('overall', {})
    b_overall = baseline_results.get('overall', {})
    clarke = lglm_eval.get('clarke_zones', {})

    print("\n" + "=" * 75)
    print("  PAPER REPRODUCTION REPORT - Gossip Learning Diabetes Monitoring")
    print("=" * 75)
    print(f"  {'Model':<25} {'RMSE':>8} {'MAE':>8} {'MARD%':>8} {'R²':>8}")
    print("-" * 75)
    print(f"  {'LGLM (Gossip, EA)':<25} {g_overall.get('RMSE', 0):>8.4f} "
          f"{g_overall.get('MAE', 0):>8.4f} {g_overall.get('MARD', 0):>8.2f} "
          f"{g_overall.get('R2', 0):>8.4f}")
    print(f"  {'Centralized LSTM':<25} {b_overall.get('RMSE', 0):>8.4f} "
          f"{b_overall.get('MAE', 0):>8.4f} {b_overall.get('MARD', 0):>8.2f} "
          f"{b_overall.get('R2', 0):>8.4f}")
    print("-" * 75)
    print(f"\n  Clarke Error Grid (LGLM):")
    for zone, pct in clarke.items():
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"    {zone}: {bar} {pct:.1f}%")

    # Topology/Energy stats
    e_stats = {}
    for pid, pr in gossip_results.get('per_patient', {}).items():
        if 'energy' in pr:
            e_stats[pid] = pr['energy']
    if e_stats:
        avg_consumed = np.mean([v.get('total_consumed', 0) for v in e_stats.values()])
        avg_remaining = np.mean([v.get('remaining_battery', 0) for v in e_stats.values()])
        print(f"\n  Energy Summary (avg across patients):")
        print(f"    Total energy consumed: {avg_consumed:.1f} mJ")
        print(f"    Remaining battery:     {avg_remaining:.1f} mJ")
    print("=" * 75)

    # -- Combined bar chart for report --
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    metrics = ['RMSE', 'MAE', 'R2']
    x = np.arange(len(metrics))
    w = 0.35
    g_vals = [g_overall.get(m, 0) for m in metrics]
    b_vals = [b_overall.get(m, 0) for m in metrics]
    b1 = ax.bar(x - w / 2, g_vals, w, label='LGLM (Gossip)', color='#00d2ff', edgecolor='#555577')
    b2 = ax.bar(x + w / 2, b_vals, w, label='Centralized LSTM', color='#f7971e', edgecolor='#555577')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_title("LGLM vs Centralized LSTM — Performance Comparison",
                 color='white', fontsize=13, fontweight='bold')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{bar.get_height():.3f}', ha='center', color='white', fontsize=9)
    ax.legend(facecolor='#2d2d4e', edgecolor='#555577', labelcolor='white', fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    plt.tight_layout()
    plt.savefig("artifacts/plots/final_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Final comparison chart saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full reproduction pipeline")
    parser.add_argument("--config", default="config/reproduction.yaml")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip ablation studies (faster run)")
    args = parser.parse_args()
    main(args)
