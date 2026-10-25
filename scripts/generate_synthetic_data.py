"""
generate_synthetic_data.py
--------------------------
Generates realistic synthetic OhioT1DM-compatible XML data for testing the
full pipeline without requiring the real (access-restricted) dataset.

The physiological glucose dynamics follow a simplified Bergman minimal model
with meal and insulin disturbances, plus realistic CGM noise.
"""

import os
import sys
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def bergman_glucose(n_steps: int, dt_min: int = 5, seed: int = 42) -> np.ndarray:
    """Simplified Bergman minimal model glucose trajectory."""
    rng = np.random.default_rng(seed)
    glucose = np.zeros(n_steps)
    glucose[0] = rng.uniform(90, 120)
    insulin_action = 0.0
    for t in range(1, n_steps):
        meal_impulse = 0.0
        # Random meals at approximately 6am, 12pm, 6pm
        step_hour = (t * dt_min / 60) % 24
        if abs(step_hour - 7) < 0.15 or abs(step_hour - 12.5) < 0.15 or abs(step_hour - 18.5) < 0.15:
            meal_impulse = rng.uniform(20, 60)
        insulin_dose = 0.0
        if meal_impulse > 0:
            insulin_dose = meal_impulse * 0.08
        insulin_action += insulin_dose * 0.3
        insulin_action *= 0.92
        g_change = 0.05 * meal_impulse - 0.3 * insulin_action - 0.01 * (glucose[t - 1] - 90)
        noise = rng.normal(0, 3)
        glucose[t] = np.clip(glucose[t - 1] + g_change + noise, 40, 400)
    return glucose


def create_patient_xml(patient_id: str, start_date: datetime,
                       n_days: int = 14, dt_min: int = 5, seed: int = 42) -> str:
    """Generate an OhioT1DM-compatible XML string for one patient."""
    rng = np.random.default_rng(seed)
    n_steps = n_days * 24 * (60 // dt_min)
    glucose_vals = bergman_glucose(n_steps, dt_min, seed=seed)

    root = ET.Element("patient", id=patient_id, weight="75", insulin_type="Humalog",
                      training_start=start_date.strftime("%d-%m-%Y %H:%M:%S"),
                      training_end=(start_date + timedelta(days=n_days - 1)).strftime("%d-%m-%Y %H:%M:%S"))

    # Glucose level events
    gl_el = ET.SubElement(root, "glucose_level")
    for i in range(n_steps):
        ts = start_date + timedelta(minutes=i * dt_min)
        # Randomly drop ~5% of readings (realistic CGM gaps)
        if rng.random() < 0.95:
            ET.SubElement(gl_el, "event",
                          ts=ts.strftime("%d-%m-%Y %H:%M:%S"),
                          value=f"{glucose_vals[i]:.1f}")

    # Bolus events (insulin)
    bolus_el = ET.SubElement(root, "bolus")
    meal_hours = [7.0, 12.5, 18.5]
    for day in range(n_days):
        for h in meal_hours:
            ts = start_date + timedelta(days=day, hours=h)
            dose = rng.uniform(2, 8)
            ET.SubElement(bolus_el, "event",
                          ts=ts.strftime("%d-%m-%Y %H:%M:%S"),
                          dose=f"{dose:.2f}",
                          bwz_carb_input=str(int(rng.uniform(30, 80))))

    # Meal events
    meal_el = ET.SubElement(root, "meal")
    for day in range(n_days):
        for h in meal_hours:
            ts = start_date + timedelta(days=day, hours=h + rng.uniform(-0.2, 0.2))
            carbs = rng.uniform(30, 80)
            ET.SubElement(meal_el, "event",
                          ts=ts.strftime("%d-%m-%Y %H:%M:%S"),
                          carbs=f"{carbs:.0f}")

    # Pretty-print XML
    raw = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    return pretty


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    patient_ids = [f"5{5 + i}" for i in range(args.n_patients)]
    base_date = datetime(2020, 1, 1, 0, 0, 0)

    for i, pid in enumerate(patient_ids):
        xml_str = create_patient_xml(
            patient_id=pid,
            start_date=base_date + timedelta(days=i),
            n_days=args.n_days,
            dt_min=5,
            seed=args.seed + i * 13
        )
        out_path = os.path.join(args.output_dir, f"patient_{pid}_training.xml")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        print(f"  Generated: {out_path}")

    print(f"\n[OK] {args.n_patients} synthetic patients written to '{args.output_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OhioT1DM-compatible data")
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--n_patients", type=int, default=6)
    parser.add_argument("--n_days", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
