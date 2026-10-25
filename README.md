# Lightweight Gossip Learning for Diabetes Monitoring
## Paper Reproduction Codebase

This is a complete, working implementation of:

> "Lightweight Gossip Learning Model for Energy-Constrained Continuous Diabetes Monitoring Systems"

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full reproduction pipeline
```bash
python -X utf8 scripts/reproduce_all.py --skip_ablation
```

### 3. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
gossip-learning-diabetes/
├── config/
│   └── reproduction.yaml        # All hyperparameters & settings
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── scripts/
│   ├── reproduce_all.py         # Master pipeline (run this)
│   ├── prepare_data.py          # Data ingestion & preprocessing
│   ├── generate_synthetic_data.py  # Synthetic OhioT1DM-compatible data
│   ├── train_lglm.py            # Gossip learning training loop
│   ├── train_baselines.py       # Centralized LSTM training
│   ├── evaluate.py              # Metrics, Clarke grid, clinical analysis
│   └── run_ablation.py          # Ablation: energy-aware vs topology
├── src/
│   ├── data/                    # Data pipeline (loader, sync, interpolation, windowing)
│   ├── models/                  # LightweightLGLM + CentralizedLSTM
│   ├── training/                # Local training loop
│   ├── communication/           # Gossip aggregation + topology builder
│   ├── energy/                  # Battery simulator
│   ├── evaluation/              # RMSE/MAE/MARD/R² metrics
│   └── clinical/                # Clarke Error Grid + Decision Support
├── artifacts/                   # Generated: models, plots, metrics, scalers
├── results/                     # Final evaluation JSON reports
└── data/                        # Place OhioT1DM XML files here, or use synthetic
```

---

## Using Real OhioT1DM Data

1. Request dataset access at: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
2. Place XML files under `data/` (any subdirectory)
3. Set `use_synthetic: false` in `config/reproduction.yaml`
4. Run the pipeline

---

## Key Equations

**Gossip Aggregation:**
```
W_i^(t+1) = α · W_i^t + (1-α)/|N| · Σ_{j∈N} W_j^t
```

**Energy-Aware Communication Gate:**
```
Communicate iff: E_available > E_threshold AND E_available ≥ E_comm
```

**Linear Interpolation (missing CGM):**
```
G_t = G_t1 + ((t-t1)/(t2-t1)) · (G_t2 - G_t1)
```

---

## Clinical Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error (mg/dL) |
| MAE  | Mean Absolute Error (mg/dL) |
| MARD | Mean Absolute Relative Difference (%) |
| R²   | Coefficient of Determination |
| Zone A (Clarke) | Clinically accurate predictions |
