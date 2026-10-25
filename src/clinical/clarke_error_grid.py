import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _clarke_zone(ref: float, pred: float) -> int:
    """Assign a Clarke Error Grid zone (0=A … 4=E)."""
    # Zone A: within 20% of reference or both <= 70
    if (abs(pred - ref) <= 0.20 * ref) or (ref <= 70 and pred <= 70):
        return 0
    # Zone E (dangerous errors)
    if (ref <= 70 and pred >= 180) or (ref >= 180 and pred <= 70):
        return 4
    # Zone D (clinically dangerous)
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 70 and 70 < pred <= 180):
        return 3
    # Zone C
    if (ref >= 130 and pred <= 70) or (ref <= 70 and pred >= 130):
        return 2
    # Zone B: remaining outside Zone A
    return 1


def clarke_error_grid(ref_values, pred_values,
                      title: str = "Clarke Error Grid",
                      save_path: str = None) -> dict:
    """
    Generate Clarke Error Grid and compute zone percentages.
    Returns dict with Zone A–E percentages.
    """
    ref = np.asarray(ref_values, dtype=float)
    pred = np.asarray(pred_values, dtype=float)

    zone_counts = [0, 0, 0, 0, 0]
    for r, p in zip(ref, pred):
        zone_counts[_clarke_zone(r, p)] += 1

    total = len(ref)
    pcts = {f'Zone {z}': zone_counts[i] / total * 100 if total > 0 else 0.0
            for i, z in enumerate('ABCDE')}

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.scatter(ref, pred, s=6, alpha=0.5, color='#00d2ff', label='Predictions')

    # Reference line (perfect prediction)
    lim = max(max(ref, default=400), max(pred, default=400), 400)
    ax.plot([0, lim], [0, lim], '--', color='#a8ff78', linewidth=1.5, label='Perfect')

    # ±20% lines
    ax.plot([0, lim], [0, lim * 1.20], color='#f7971e', linewidth=0.8, alpha=0.6)
    ax.plot([0, lim], [0, lim * 0.80], color='#f7971e', linewidth=0.8, alpha=0.6)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)', color='white', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', color='white', fontsize=12)
    ax.set_title(title, color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    # Zone labels
    for txt, pos in [('A', (30, 20)), ('B', (350, 280)), ('B', (280, 370)),
                     ('C', (180, 370)), ('C', (180, 20)), ('D', (30, 140)), ('D', (370, 120))]:
        ax.text(*pos, txt, color='#aaaadd', fontsize=14, fontweight='bold', alpha=0.7)

    legend = ax.legend(facecolor='#2d2d4e', edgecolor='#555577', labelcolor='white')

    # Add zone pct annotation
    zone_text = "\n".join(f"{z}: {pcts[z]:.1f}%" for z in ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E'])
    ax.text(0.02, 0.98, zone_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='#2d2d4e', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return pcts
