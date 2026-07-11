# PC-NDT: Physics-Constrained Neural Digital Twin for CNC Machining

**Physics-Constrained Neural Ordinary Differential Equations and Spatio-Temporal Graph Networks for Real-Time Fault Propagation Modeling in CNC Machining Digital Twins**

> Author: Hiral Kumar | B.Tech CSE, Gautam Buddha University | AWS Cloud Club GBU

---

## What This Is

A production-grade ML system that predicts **when a CNC machine bearing will fail** — not just detecting faults, but forecasting Remaining Useful Life (RUL) with physical interpretability.

Three things make this different from standard predictive maintenance:

1. **Learned sensor graph (AGCRN)** — discovers which sensors physically influence each other, without being told in advance
2. **Continuous-time dynamics (Neural ODE)** — models degradation as a differential equation process, not discrete timesteps
3. **Physics constraints (Archard + Paris + Fourier)** — three physical laws embedded in training, so the model cannot make physically impossible predictions

---

## Project Structure

```
cnc-digital-twin/
├── config/config.yaml          ← ALL hyperparameters live here
├── src/
│   ├── data/
│   │   ├── ims_loader.py       ← NASA IMS dataset loader
│   │   ├── pronostia_loader.py ← PRONOSTIA dataset loader
│   │   ├── preprocessing.py    ← Normalization, windowing, PyTorch Dataset
│   │   └── graph_utils.py      ← Graph Laplacian, proximity adjacency
│   ├── models/
│   │   ├── agcrn.py            ← Adaptive Graph Conv Recurrent Network
│   │   ├── neural_ode.py       ← Neural ODE (torchdiffeq, adjoint method)
│   │   └── pc_ndt.py           ← Unified model (AGCRN + ODE + readout)
│   ├── physics/
│   │   └── constraints.py      ← Archard, Paris, Fourier loss terms
│   ├── training/
│   │   └── trainer.py          ← Training loop, warm-up schedule, early stopping
│   └── evaluation/
│       └── metrics.py          ← RMSE, PHM2012 Score, ΔRMSEdrop, Pearson ρ
├── scripts/
│   └── train.py                ← Entry point
├── tests/
│   ├── test_smoke.py           ← Data pipeline tests (no dataset needed)
│   └── test_model_smoke.py     ← Model architecture tests (no dataset needed)
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Hiral-Kumar/cnc-digital-twin.git
cd cnc-digital-twin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
#    NASA IMS: https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
#    Unzip to: data/raw/IMS/

# 4. Update paths in config/config.yaml → data.ims.test1_dir etc.
```

---

## Run Tests (no dataset needed)

```bash
pytest tests/ -v
```

Expected output: **all tests passing** using synthetic data.

---

## Train

```bash
# Debug mode (5 epochs, IMS Test 2 — fast sanity check)
python scripts/train.py --debug

# Full training (IMS Test 1, 200 epochs)
python scripts/train.py
```

---

## Architecture

```
Input [B, T, N, F]
  ↓
AGCRN (learns sensor graph A, produces H_T)
  ↓
Neural ODE (integrates dh/dt = f_θ(h,t) from H_T)
  ↓  ← Physics loss enforced here (Archard + Paris + Fourier)
Linear Readout → RUL [B, N]
```

---

## Datasets

| Dataset | Role | Access |
|---|---|---|
| NASA IMS Bearing | Primary training | NASA PCoE (public) |
| PRONOSTIA PHM 2012 | Cross-condition test | IEEE PHM (public) |
| NASA Milling | Archard validation | NASA PCoE (public) |

---

## Research Questions

- **RQ1:** Can adaptive graph learning recover physically meaningful sensor topology without human specification?
- **RQ2:** Do physics-constrained Neural ODEs produce more physically plausible degradation trajectories than unconstrained baselines?
- **RQ3:** Do physics constraints improve generalisation to unseen operating conditions?

---

*Part of AWS Cloud Club GBU AI/ML Mentorship Program*
