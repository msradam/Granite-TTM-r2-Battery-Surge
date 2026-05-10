# How this model was trained

Source: `riprap-nyc/experiments/20_ttm_battery_surge/finetune_ttm_battery.py`.
The script + reproducible config below produced the published checkpoint.

## Hardware + software

| | |
|---|---|
| GPU | 1× AMD Instinct MI300X (192 GB HBM3) |
| Cloud | AMD Developer Cloud (DigitalOcean droplet) |
| ROCm | 4.0.0+1a5c7ec |
| Container | `rocm:latest` (sibling of `terramind` container) |
| Python | 3.12 |
| granite-tsfm | 0.3.x |
| transformers | 4.55–4.x |
| Precision | fp32 (TTM is tiny; mixed-precision unnecessary) |

Total wall-clock to train: **~5 minutes** on MI300X. TTM is 1.5M
params; this is practically free GPU time.

## Data

| Component | Source | License |
|---|---|---|
| Observed water level | NOAA CO-OPS API, station 8518750 | US Government public domain |
| Astronomical tide prediction | NOAA CO-OPS `predictions` endpoint | US Government public domain |
| Surge residual | computed = observed − predicted | derived |

Range: 2015-01-01 to 2024-12-31 (10 years). 6-min observations
resampled to **hourly mean** to match TTM r2's training cadence.
Total: 87,672 hourly observations.

Surge residual range: −1.109 m to +1.591 m (Hurricane Sandy 2012
falls outside this 2015-2024 window — Sandy is in the historical
record but not in our training distribution).

## Splits

Chronological (no leakage):

- Train: 2015-2022 (70%, 60,251 sliding windows)
- Val: 2022-2023 (15%, 12,031 windows)
- Test: 2023-2024 (15%, 12,033 windows)

Each window is **1024 hours of context → 96 hours horizon** (43 days
in, 4 days out). Sliding stride 1 hour. Splits live in
`riprap-nyc/experiments/20_ttm_battery_surge/`.

## Hyperparameters

| | |
|---|---|
| Base model | `ibm-granite/granite-timeseries-ttm-r2` |
| Context length | 1024 (hourly) |
| Prediction length | 96 (hourly) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | Cosine with warmup (10% of total steps) |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Epochs | 10 |
| Loss | MSE on standardized residuals |
| Seed | 42 |

## Training command

```bash
# Inside the rocm container
cd /workspace/experiments/20_ttm_battery_surge

# 1. Pull NOAA data
python3 fetch_noaa_battery.py \
    --station 8518750 \
    --start 2015-01-01 \
    --end 2024-12-31 \
    --out /root/ttm_battery/battery_2015_2024.parquet

# 2. Fine-tune
python3 finetune_ttm_battery.py \
    --data /root/ttm_battery/battery_2015_2024.parquet \
    --out  /root/ttm_battery/output_phase20
```

## Eval methodology

Test windows: 12,033 chronologically-disjoint sliding windows from
2023-2024. Three baselines reported alongside the fine-tune:

- **Persistence**: forecast = last observed value, held flat for 96h.
- **Zero-shot TTM r2**: same architecture, IBM-pretrained weights only.
- **Fine-tuned TTM r2**: this checkpoint.

Metric: MAE in metres on the surge residual.

| | MAE (m) | RMSE (m) |
|---|---:|---:|
| Persistence | 0.1861 | 0.2417 |
| Zero-shot TTM r2 | 0.1467 | 0.1903 |
| **Fine-tuned (this work)** | **0.1091** | **0.1568** |

## Why not include exogenous variables

The original Phase 16 plan included wind / pressure / rainfall as
exogenous TTM channels (TTM r2 supports them). Phase 20 stripped that
back to pure univariate residual forecasting because:

1. The univariate model already beats persistence by 41% — exogenous
   features would be the next iteration, not the publishable v1.
2. Multivariate TTM input requires consistent METAR availability across
   the full 10-year window; KNYC has gaps that complicate clean splits.
3. Keeping the input shape minimal makes the model a clean drop-in for
   any other coastal tide-gauge fine-tune (Sandy Hook, Kings Point,
   Boston, Providence, Norfolk).

## Where to extend

- Add Sandy Hook (8531680) and Kings Point (8516945) fine-tunes (see
  the sniff-test results — current model is OOD at those stations).
- Add exogenous METAR / NWS HRRR features.
- Extend training to incorporate Hurricane Sandy 2012 once an extended
  data fetch is wired (NOAA CO-OPS goes back further than 2015).

The reproduction harness at
[github.com/msradam/riprap-models](https://github.com/msradam/riprap-models)
already includes the eval + bench infrastructure for these extensions.
