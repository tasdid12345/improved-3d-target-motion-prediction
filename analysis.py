"""Simulated 3D respiratory-style target motion: per-axis linear AR baseline vs LightGBM."""
import json, sys, time
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

SEED, SEEDS = 42, [42, 43, 44]
FS = 10                   # simulated frame rate (Hz)
N_PAT, T = 12, 3000       # simulated patients and frames per patient (5 min)
LAGS, H = 5, 10           # number of lagged inputs; forecast horizon (frames)
TRAIN_FRAC = 0.7          # time-ordered split inside each patient
AXES = ["SI", "AP", "LR"] # axis 0 (SI) is dominant
rng = np.random.default_rng(SEED)
t_start = time.time()


def simulate(rng, n_frames):
    """Respiratory-like 3D trajectory (mm): sum of sinusoids, drift, varying period/amplitude, noise."""
    t = np.arange(n_frames) / FS
    period = rng.uniform(3.0, 5.0)                       # breathing period (s)
    phase = 2 * np.pi * t / period + 0.3 * np.sin(2 * np.pi * t / rng.uniform(60, 120))
    amp = rng.uniform(0.8, 1.2) * (1 + 0.15 * np.sin(2 * np.pi * t / rng.uniform(40, 90)))
    base_amp = np.array([10.0, 3.0, 2.0])                # SI dominant, then AP, LR
    traj = np.zeros((n_frames, 3))
    for ax in range(3):
        harmonic = np.sin(phase + 0.5 * ax) + 0.2 * np.sin(2 * phase + ax)
        drift = rng.uniform(-0.01, 0.01) * t            # slow baseline drift
        traj[:, ax] = base_amp[ax] * amp * harmonic + drift + rng.normal(0, 0.2, n_frames)
    return traj


def build(trajs):
    """One row per anchor frame t (last observed frame). Features use only frames <= t;
    target is the position at t+H. Windows straddling the split are dropped."""
    parts = {"tr": [], "te": []}
    for traj in trajs:
        split = int(TRAIN_FRAC * len(traj))
        a = np.arange(LAGS - 1, len(traj) - H)
        lag = np.stack([traj[a - k] for k in range(LAGS)], axis=1)   # (m, LAGS, 3), lag 0 = now
        cur, vel = traj[a], traj[a] - traj[a - 1]
        summ = np.concatenate([lag.mean(1), lag.std(1)], axis=1)     # simple summary features
        X = np.concatenate([lag.reshape(len(a), -1), vel, summ], axis=1)
        Y = traj[a + H]
        for key, m in (("tr", a + H < split), ("te", a >= split)):
            parts[key].append((X[m], Y[m], cur[m], lag[m]))
    return {k: [np.concatenate(z) for z in zip(*v)] for k, v in parts.items()}  # [X, Y, cur, lag]


def evaluate(pred, y):
    """3D RMSE (main), SI RMSE and 95th-percentile 3D error (secondary), in mm."""
    err = pred - y
    e3 = np.linalg.norm(err, axis=1)
    return {"rmse3d": float(np.sqrt(np.mean(e3 ** 2))),
            "rmse_si": float(np.sqrt(np.mean(err[:, 0] ** 2))),
            "p95_3d": float(np.percentile(e3, 95))}


def fit_baseline(Ltr, Ytr, Lte):
    """BASELINE: per-axis least-squares AR on the last LAGS positions of that axis -> position at t+H."""
    pred = np.zeros((len(Lte), 3))
    for ax in range(3):
        pred[:, ax] = LinearRegression().fit(Ltr[:, :, ax], Ytr[:, ax]).predict(Lte[:, :, ax])
    return pred


def fit_lgb(Xtr, Dtr, Xte, params, seed):
    """IMPROVED: one LightGBM regressor per axis predicting displacement (t+H minus t)."""
    out = np.zeros((len(Xte), 3))
    for ax in range(3):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1, **params)
        out[:, ax] = m.fit(Xtr, Dtr[:, ax]).predict(Xte)
    return out


def tune(Xtr, Ytr, Ctr):
    """Small grid chosen ONLY by 3-fold contiguous CV on the training windows (test set untouched)."""
    grid = [dict(num_leaves=nl, min_child_samples=mc) for nl in (15, 31) for mc in (20, 100)]
    best, best_s = None, np.inf
    for p in grid:
        errs = []
        for i_tr, i_va in KFold(3, shuffle=False).split(Xtr):
            pd = fit_lgb(Xtr[i_tr], Ytr[i_tr] - Ctr[i_tr], Xtr[i_va], p, SEED)
            errs.append(evaluate(pd + Ctr[i_va], Ytr[i_va])["rmse3d"])
        s = float(np.mean(errs))
        print(f"  CV {p}: 3D RMSE {s:.4f} mm")
        if s < best_s:
            best, best_s = p, s
    return best


def main():
    print("simulating patients")
    d = build([simulate(rng, T) for _ in range(N_PAT)])
    Xtr, Ytr, Ctr, Ltr = d["tr"]
    Xte, Yte, Cte, Lte = d["te"]
    if len(Xte) == 0 or len(Xtr) == 0:
        sys.exit("empty train or test set")
    print(f"train windows {len(Xtr)}, test windows {len(Xte)}")

    print("tuning LightGBM with CV on training set")
    best = tune(Xtr, Ytr, Ctr)
    print("selected", best)

    base_runs, imp_runs = [], []
    for s in SEEDS:  # baseline is deterministic (identical each seed); improved is seeded
        print(f"seed {s}: fitting both models")
        base_runs.append(evaluate(fit_baseline(Ltr, Ytr, Lte), Yte))
        imp_runs.append(evaluate(fit_lgb(Xtr, Ytr - Ctr, Xte, best, s) + Cte, Yte))

    get = lambda runs, k: [r[k] for r in runs]
    b_main, i_main = get(base_runs, "rmse3d"), get(imp_runs, "rmse3d")
    b_mean, i_mean = float(np.mean(b_main)), float(np.mean(i_main))
    results = {
        "question": "Can a LightGBM model on lagged 3D positions predict an anatomical target's through-plane and full 3D position ahead of time more accurately than a per-axis linear autoregressive motion predictor?",
        "method": "Simulated 3D respiratory-style trajectories; time-ordered 70/30 split per patient; baseline per-axis least-squares AR(5) predicting 10 frames ahead vs per-axis LightGBM predicting displacement from lagged/velocity/summary features; hyper-parameters chosen by 3-fold CV on training data; test set used once.",
        "dataset": {
            "name": "Simulated 3D respiratory-style target trajectories (numpy, seed 42)",
            "source_url": "https://lightgbm.readthedocs.io and https://scikit-learn.org (libraries; data generated in code, no download)",
            "n_rows": int(len(Xtr) + len(Xte)),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "3D RMSE of predicted target position at +10 frames (mm)",
        "higher_is_better": False,
        "baseline": "Per-axis linear autoregressive predictor (least squares on last 5 positions of each axis) predicting position 10 frames ahead.",
        "improved": "Per-axis LightGBM regressor on all-axis lagged positions, velocity and summary statistics, predicting displacement from the current position; hyper-parameters tuned by training-set CV.",
        "metrics": {
            "baseline_score": b_mean,
            "improved_score": i_mean,
            "improvement": b_mean - i_mean,
            "baseline_score_std": float(np.std(b_main)),
            "improved_score_std": float(np.std(i_main)),
            "baseline_si_rmse": float(np.mean(get(base_runs, "rmse_si"))),
            "improved_si_rmse": float(np.mean(get(imp_runs, "rmse_si"))),
            "baseline_p95_3d_error": float(np.mean(get(base_runs, "p95_3d"))),
            "improved_p95_3d_error": float(np.mean(get(imp_runs, "p95_3d"))),
        },
        "notes": ("Simulated data only; the paper's MR data is not public, so absolute errors are not comparable to the paper. "
                  "Baseline has no randomness, so its std is 0 across seeds. CV folds are contiguous blocks of training windows "
                  "without a gap (mild optimism possible). LightGBM uses all-axis inputs whereas the baseline is per-axis, as specified. "
                  f"Selected params: {best}. Single simulated cohort of {N_PAT} patients: conclusions depend on the simulation design."),
    }
    if not all(np.isfinite([b_mean, i_mean])):
        sys.exit("non-finite metrics")

    import os
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["baseline (AR)", "improved (LightGBM)"], [b_mean, i_mean], color=["gray", "tab:blue"])
    plt.ylabel("3D RMSE at +10 frames (mm)")
    plt.title("Test-set prediction error")
    plt.tight_layout()
    plt.savefig("figures/rmse_bar.png", dpi=120)
    plt.close()

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"baseline 3D RMSE {b_mean:.4f} mm, improved {i_mean:.4f} mm")
    print(f"done in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
