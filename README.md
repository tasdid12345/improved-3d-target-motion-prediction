# Simulated 3D Target Tracking: LightGBM vs Linear Autoregressive Motion Prediction

Motivation: Real-time target tracking during radiotherapy must compensate for respiratory motion, and the prediction of the target position ahead of time is central to that compensation. The paper "Real-time target tracking using image registration and motion prediction" (2026, https://link.springer.com/article/10.1007/s11548-026-03634-y) motivates this experiment, which asks whether a tree-based model can forecast the target more accurately than a standard linear motion predictor.

Question: Can a LightGBM model on lagged 3D positions predict an anatomical target's through-plane and full 3D position ahead of time more accurately than a per-axis linear autoregressive motion predictor?

Dataset: Simulated 3D respiratory-style target trajectories (sum of sinusoids with drift, varying period and amplitude, superior-inferior axis dominant, plus noise), generated in code with a fixed seed. This is not the paper's own data, which is not public (`same_as_paper: false`). Each patient is split in time: first 70% of frames for training, last 30% for testing.

Baseline vs improved:
- Baseline: per-axis linear autoregressive predictor fitted by least squares on the last 5 positions to predict the position 10 frames ahead.
- Improved: per-axis LightGBM regressor on the same lagged inputs plus velocity and summary features, predicting displacement from the current position. Hyper-parameters are chosen by cross-validation on the training set only.

Metrics: 3D RMSE (main), with superior-inferior RMSE and 95th-percentile 3D error as secondary metrics.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (figures are saved to figures/).

## Results (latest run)

On a simulated 3D respiratory-style trajectory dataset, a standard per-axis linear autoregressive predictor reached 5.38 mm 3D RMSE at +10 frames on the held-out test set, while my per-axis LightGBM model reached 2.47 mm on the same split.

| Metric | Value |
|---|---|
| baseline_score | 5.376 |
| improved_score | 2.467 |
| improvement | 2.909 |
| baseline_score_std | 0 |
| improved_score_std | 0.002741 |
| baseline_si_rmse | 4.442 |
| improved_si_rmse | 2.311 |
| baseline_p95_3d_error | 9.221 |
| improved_p95_3d_error | 5.259 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data only; the paper's MR data is not public, so absolute errors are not comparable to the paper. Baseline has no randomness, so its std is 0 across seeds. CV folds are contiguous blocks of training windows without a gap (mild optimism possible). LightGBM uses all-axis inputs whereas the baseline is per-axis, as specified. Selected params: {'num_leaves': 15, 'min_child_samples': 100}. Single simulated cohort of 12 patients: conclusions depend on the simulation design.

## Reference

This project is motivated by the following paper:

> V Buchanan, E Lugez (2026). *Real-time target tracking using image registration and motion prediction*. … Journal of Computer Assisted Radiology and. https://link.springer.com/article/10.1007/s11548-026-03634-y

Paper author: Prof. Elodie Lugez, Toronto Metropolitan University (elodie.lugez@torontomu.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
