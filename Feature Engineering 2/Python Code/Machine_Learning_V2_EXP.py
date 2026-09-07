"""
=========================
30-Minute (v2) REGRESSION MODEL [CHRONOLOGICAL SPLIT]: Random Forest + XGBoost + SVR + Ridge
LIVE OPTUNA HYPERPARAMETER TUNING -- run fresh on this dataset every run,
with k-fold CV both as the tuning objective (TUNE_CV_FOLDS) and as a
post-fit generalisation check (CV_VARIABILITY_FOLDS). 8 models per target:
Tuned + Baseline for each of Random Forest / XGBoost / SVR / Ridge. Uses
the chronological (event-order, NOT shuffled) 80/20 train/test split --
see 30min_v2_phase1.py for the shuffled sibling.

FEATURE SELECTION ENGINE (replaces the old corr+RF-importance composite):
  1. Multicollinearity pre-filter: Spearman correlation matrix on the
     candidate feature pool -> hierarchical agglomerative clustering
     (threshold = 0.85) -> keep 1 representative per cluster, preferring
     Hydrodynamic terms > Spectral Peak/PSD terms > RMS terms > everything
     else, with Min/Max-style metrics deprioritised last.
  2. rank_features(): iterative greedy selection scored as
     0.45*avg_spearman_norm + 0.45*perm_importance_norm - 0.10*redundancy,
     where permutation importance comes from a screening RandomForest
     fit on an internal train/val carve-out of the TRAIN fold only.

Both steps run on the TRAIN fold only (df_train_full) -- never on the
pooled train+test set -- matching the leakage fix applied across the rest
of this project's pipeline.
=========================
"""

import warnings
warnings.filterwarnings("ignore")

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import re
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    learning_curve, validation_curve, cross_val_score, train_test_split,
)

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_OK = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("Warning: optuna not found. Run: pip install optuna")
    print("Falling back to generic default hyperparameters for 'Tuned' models.\n")


# =========================
# CONFIGURATION
# =========================


INPUT_TRAIN_DIR = r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Phase 2 Experiment\TrainTestSplit_30min_v2\Chronological\train_dataset_30min_v2_80pct_chronological.csv"
INPUT_TEST_DIR  = r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Phase 2 Experiment\TrainTestSplit_30min_v2\Chronological\test_dataset_30min_v2_20pct_chronological.csv"
BASE_OUTPUT_DIR = r"C:\Users\N O\Desktop\MASTER\Methodology & Result\Phase 2 Experiment"

DATASETS = [(INPUT_TRAIN_DIR, INPUT_TEST_DIR, "30MinuteV2_Chronological")]

MAX_SVR_ROWS  = 5_000
MAX_TREE_ROWS = 50_000

DIAG_MAX_ROWS         = 3_000
DIAG_CV_FOLDS         = 3
CV_VARIABILITY_FOLDS  = 5
LEARNING_CURVE_TRAIN_SIZES = np.linspace(0.2, 1.0, 5)

VALIDATION_CURVE_SPECS = {
    "Random Forest": dict(step="rf",  param="n_estimators",
                           range=[50, 100, 200, 300, 400, 600, 800, 1000]),
    "XGBoost":        dict(step="xgb", param="max_depth",
                           range=[2, 3, 4, 5, 6, 7, 8, 10, 12]),
    "SVR":            dict(step="svr", param="C",
                           range=np.logspace(-3, 2, 8)),
    "Ridge":          dict(step="ridge", param="alpha",
                           range=np.logspace(-3, 3, 8)),
}

EXCLUDE_COLS = ["Event_No", "Source_File"]   # identifiers, never features
TOP_K_FEATURES = 10
MULTICOLLINEARITY_THRESHOLD = 0.85
FEATURE_SELECTION_SEED = 42

# Each model family gets its OWN top-K feature set: the multicollinearity
# pre-filter is shared (it's a data-hygiene step, not a model property),
# but permutation importance is computed with a screening model matched to
# that family, so e.g. Ridge's features are ranked by what actually helps
# a linear model, not by RandomForest's own inductive bias.
FAMILY_NAMES  = ["rf", "xgb", "svr", "ridge"]
FAMILY_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "svr": "SVR", "ridge": "Ridge"}

FEATURE_COLS_BY_FAMILY   = {}
FEATURE_LABELS_BY_FAMILY = {}

NATF_EW_COL = "X Frequency"
NATF_NS_COL = "Y Frequency"
TARGET_COLS = [NATF_EW_COL, NATF_NS_COL]
SHOW_PLOTS  = False

# Colour scheme
TUNED_KEYWORDS = ["(Tuned)"]
COLOR_TUNED    = "#1565C0"   # dark blue  -- tuned models
COLOR_BASELINE = "#90CAF9"   # light blue -- baseline models

# Distinct per-model colours for the combined/segmented pred-vs-measured
# overlays (cycled by index, not matched by exact model name, since a
# fallback name like "Gradient Boosting (Tuned)" can replace "XGBoost
# (Tuned)" when xgboost isn't installed).
MODEL_PALETTE = ["#1565C0", "#2E7D32", "#EF6C00", "#8E24AA", "#C62828", "#00838F"]

# Each Event_No step is one 30-minute chronological window (matches Phase
# 2's WINDOW_MINUTES) -- used to build an "elapsed time since first test
# event" x-axis, since no real per-event clock time survives into the
# engineered feature set.
EVENT_WINDOW_MINUTES = 30

# =========================
# LIVE OPTUNA TUNING CONFIG
# Each "Tuned" model is optimised fresh, per target, on this dataset's
# actual chronological train fold -- the objective score itself is a
# k-fold CV score (TUNE_CV_FOLDS), so cross-validation drives which
# hyperparameters get picked, not just the single train/test split.
# =========================

N_TRIALS        = 50
TUNE_CV_FOLDS   = 3
TUNE_METRIC     = "neg_root_mean_squared_error"
OPTUNA_TIMEOUT  = 300          # seconds, safety cap per model's study
OPTUNA_SEED     = 42

# Used only if Optuna is unavailable / a study errors out.
FALLBACK_TUNED_PARAMS = {
    "rf":    dict(n_estimators=400, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0),
    "xgb":   dict(n_estimators=600, max_depth=4, learning_rate=0.05,
                 subsample=0.9, colsample_bytree=0.9),
    "svr":   dict(kernel="rbf", C=50, epsilon=0.001, gamma="scale"),
    "ridge": dict(alpha=1.0, solver="auto"),
}


# =========================
# HELPERS
# =========================

def safe_col(name):
    return re.sub(r"[^\w]", "_", name)

def fmt_elapsed(seconds):
    """H:MM:SS (or M:SS under an hour) for printing timing info."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def is_tuned(name):
    return any(kw in name for kw in TUNED_KEYWORDS)

def elapsed_time_axis(event_no):
    """Minutes elapsed since the earliest event in `event_no`, treating each
    Event_No step as one EVENT_WINDOW_MINUTES-long chronological window
    (holds even across any skipped/missing event IDs, since Event_No
    itself was assigned in real chronological order)."""
    event_no = np.asarray(event_no)
    return (event_no - event_no.min()) * EVENT_WINDOW_MINUTES

def fmt_elapsed_dhm(minutes, pos=None):
    """'Xd Yh' (or just 'Yh' under a day) tick-label formatter for an
    elapsed-minutes x-axis built by elapsed_time_axis()."""
    minutes = max(0, int(round(minutes)))
    d, rem = divmod(minutes, 24 * 60)
    h, _   = divmod(rem, 60)
    return f"{d}d {h}h" if d else f"{h}h"

def save_plot(path):
    try:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"    Saved plot : {path}")
    except Exception as e:
        print(f"    Warning: could not save plot {path}: {e}")
    finally:
        plt.close("all")

def calc_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps    = 1e-9
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    mape   = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100
    r2     = r2_score(y_true, y_pred)
    acc    = np.mean(
        np.clip(1.0 - np.abs(y_pred - y_true) / np.clip(np.abs(y_true), eps, None), 0.0, 1.0)
    ) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2, "Accuracy_%": acc}

def _subsample(X, y, max_rows, label, seed=42):
    n = len(X)
    if n <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_rows, replace=False)
    X_s = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    y_s = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    print(f"      ({label} subsample: {n:,} rows -> {max_rows:,} rows for speed)")
    return X_s, y_s

def subsample_for_svr(X, y, max_rows, seed=42):
    return _subsample(X, y, max_rows, "SVR", seed)

def subsample_for_trees(X, y, max_rows, seed=42):
    return _subsample(X, y, max_rows, "Tree", seed)

def load_csv_robust(filepath):
    df = pd.read_csv(filepath, header=0)
    df.columns = [str(c).strip().lstrip("\ufeff").strip() for c in df.columns]
    return df


# =========================
# FEATURE SELECTION ENGINE
# =========================

def multicollinearity_filter(X_train, threshold=MULTICOLLINEARITY_THRESHOLD, out_dir=None):
    """
    Spearman correlation matrix on the candidate feature pool -> hierarchical
    agglomerative clustering (average linkage, correlation threshold) -> 1
    representative feature kept per cluster. Preference order when a
    cluster has to pick one representative: Hydrodynamic terms > Spectral
    Peak/PSD terms > RMS terms > everything else > Min/Max-style metrics
    (most redundant / least preferred).
    """
    corr = X_train.corr(method="spearman").abs().fillna(0.0)
    # pandas' copy-on-write can back .values with a read-only array --
    # fill_diagonal needs a genuinely writable buffer.
    corr_vals = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 1.0)
    corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
    dist = 1.0 - corr
    dist_condensed = squareform(dist.values, checks=False)
    Z = linkage(dist_condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - threshold, criterion="distance")

    hydro_keys   = ["Hs (m)", "Hmax (m)", "Tz (s)", "Wave_Steepness", "Hydro_Interaction"]
    spectral_keys = ["PeakFreq", "PeakPower", "SpectralCentroid", "SpectralBandwidth", "SpectralEntropy"]
    rms_keys      = ["_RMS"]

    def priority_score(feat):
        if any(k in feat for k in hydro_keys):
            return 0
        if any(k in feat for k in spectral_keys):
            return 1
        if any(k in feat for k in rms_keys):
            return 2
        if "_Min" in feat or "_Max" in feat:
            return 4
        return 3

    clusters = {}
    for feat, cid in zip(X_train.columns, cluster_ids):
        clusters.setdefault(cid, []).append(feat)

    representatives, cluster_rows = [], []
    for cid, feats in clusters.items():
        feats_sorted = sorted(feats, key=priority_score)
        rep = feats_sorted[0]
        representatives.append(rep)
        for f in feats:
            cluster_rows.append({"Cluster": cid, "Feature": f, "Representative": rep,
                                  "Is_Representative": f == rep})

    cluster_df = pd.DataFrame(cluster_rows).sort_values(["Cluster", "Is_Representative"],
                                                          ascending=[True, False])
    print(f"\n  Multicollinearity filter: {len(X_train.columns)} candidates -> "
          f"{len(clusters)} clusters -> {len(representatives)} representatives "
          f"(Spearman threshold={threshold})")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "multicollinearity_clusters.csv")
        cluster_df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    return representatives, cluster_df


def _build_screening_pipeline(model_family, seed):
    """A fixed-config (not yet tuned) pipeline for the given family, used
    only to screen permutation importance during feature selection -- not
    the final model. Matches each family's own inductive bias: tree
    ensembles get no scaling, SVR/Ridge get StandardScaler since they're
    scale-sensitive."""
    if model_family == "rf":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)),
        ])
    elif model_family == "xgb":
        if XGB_OK:
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                                       random_state=seed, verbosity=0,
                                       eval_metric="rmse", n_jobs=-1)),
            ])
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(n_estimators=300, random_state=seed)),
        ])
    elif model_family == "svr":
        return Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model",  SVR(kernel="rbf", C=50, gamma="scale")),
        ])
    elif model_family == "ridge":
        return Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0, random_state=seed)),
        ])
    raise ValueError(f"Unknown model_family: {model_family}")


def rank_features(X_train, y_train_df, target_cols, model_family,
                  top_k=TOP_K_FEATURES, seed=FEATURE_SELECTION_SEED):
    """
    Iterative greedy selection:
        score(feat) = 0.45*avg_spearman_norm + 0.45*avg_perm_importance_norm
                      - 0.10*redundancy_vs_already_selected

    Spearman |r| is model-agnostic (shared across families). Permutation
    importance is family-specific: the screening model matches model_family
    (RF/XGBoost/SVR/Ridge), so each family's top-K reflects what actually
    helps THAT kind of model, not always RandomForest's own inductive bias.
    A screening model is fit per target on an internal 80/20 train/val
    split carved out of X_train/y_train_df (never touching the outer test
    fold); the two targets' normalised importances -- like their normalised
    |Spearman r| -- are averaged into one shared per-family score, so a
    single top-K feature set serves both targets for that family.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_df, test_size=0.2, random_state=seed
    )

    # SVR screening is O(n^2) -- subsample for speed, same safeguard as
    # SVR's own tuning/CV steps elsewhere in this script.
    if model_family == "svr" and len(X_tr) > MAX_SVR_ROWS:
        X_tr_fit, y_tr_fit = subsample_for_svr(X_tr, y_tr, MAX_SVR_ROWS)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    perm_norms, spearman_norms = [], []
    for tgt in target_cols:
        screen = _build_screening_pipeline(model_family, seed)
        screen.fit(X_tr_fit, y_tr_fit[tgt])
        perm_result = permutation_importance(
            screen, X_val, y_val[tgt],
            n_repeats=10, scoring="r2", random_state=seed, n_jobs=-1,
        )
        imp = pd.Series(perm_result.importances_mean, index=X_train.columns).clip(lower=0.0)
        perm_norms.append(imp / (imp.sum() + 1e-12))

        sp = X_train.apply(lambda col: abs(col.corr(y_train_df[tgt], method="spearman"))).fillna(0.0)
        spearman_norms.append(sp / (sp.sum() + 1e-12))

    perm_norm     = pd.concat(perm_norms, axis=1).mean(axis=1)
    spearman_norm = pd.concat(spearman_norms, axis=1).mean(axis=1)

    selected_features, feature_scores = [], {}
    remaining = list(X_train.columns)
    for _ in range(min(top_k, len(remaining))):
        best_feat, best_score = None, -np.inf
        for feat in remaining:
            redundancy = 0.0
            if selected_features:
                redundancy = X_train[selected_features].corrwith(
                    X_train[feat], method="spearman"
                ).abs().mean()
                if pd.isna(redundancy):
                    redundancy = 0.0
            score = 0.45 * spearman_norm[feat] + 0.45 * perm_norm[feat] - 0.10 * redundancy
            if score > best_score:
                best_score, best_feat = score, feat
        selected_features.append(best_feat)
        feature_scores[best_feat] = best_score
        remaining.remove(best_feat)

    ranking = pd.DataFrame({
        "Feature":               selected_features,
        "Composite_Score":       [feature_scores[f] for f in selected_features],
        "Avg_Spearman_norm":     [spearman_norm[f] for f in selected_features],
        "Avg_PermImportance_norm": [perm_norm[f] for f in selected_features],
    })
    return selected_features, ranking


def select_features_all_families(df_train_full, candidate_cols, target_cols, top_k, out_dir=None):
    """Full v2 selection pipeline, run once per model family:
    multicollinearity pre-filter (shared) -> greedy permutation-importance
    + Spearman ranking (family-specific screening model). TRAIN fold only.

    Returns (feature_sets, rankings), both dicts keyed by family name
    ('rf', 'xgb', 'svr', 'ridge')."""
    work = df_train_full[candidate_cols + target_cols].dropna()
    if work.empty:
        raise RuntimeError("No complete rows available for feature selection.")

    X_train = work[candidate_cols]
    y_train_df = work[target_cols]

    # Multicollinearity filtering is a property of the DATA (redundant
    # columns), not of any one model, so it only needs to run once and its
    # representatives are shared as the candidate pool for every family.
    representatives, cluster_df = multicollinearity_filter(X_train, MULTICOLLINEARITY_THRESHOLD, out_dir)
    X_train_repr = X_train[representatives]

    feature_sets, rankings = {}, {}
    for family in FAMILY_NAMES:
        # "xgb" always gets its own feature set, even without xgboost
        # installed -- _build_screening_pipeline falls back to
        # GradientBoostingRegressor for screening, and build_models_tuned's
        # fallback model is still tagged family="xgb", so family_X_train
        # must always have an "xgb" entry to key against.
        selected, ranking = rank_features(X_train_repr, y_train_df, target_cols, family, top_k)
        feature_sets[family] = selected
        rankings[family] = ranking

        print(f"\n  Feature selection [{FAMILY_LABELS[family]}] -- top {len(selected)} of "
              f"{len(representatives)} representatives "
              f"(0.45*Spearman + 0.45*PermImportance[{FAMILY_LABELS[family]}] - 0.10*redundancy, "
              f"vs {', '.join(target_cols)}):")
        print(ranking.to_string(index=False))

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"feature_selection_ranking_{family}.csv")
            ranking.to_csv(path, index=False)
            print(f"  Saved: {path}")

    if out_dir:
        comparison = build_feature_selection_comparison(rankings)
        path = os.path.join(out_dir, "feature_selection_comparison.csv")
        comparison.to_csv(path, index=False)
        print(f"\n  Feature selection comparison across all {len(rankings)} model families:")
        print(comparison.to_string(index=False))
        print(f"  Saved: {path}")

    return feature_sets, rankings


def build_feature_selection_comparison(rankings):
    """Wide comparison table -- one row per feature selected by AT LEAST
    one family, one Selected_{family}/Rank_{family}/Score_{family} column
    trio per family, plus a summary count of how many families picked it.
    Lets you see at a glance which features are universally useful vs.
    specific to one model's own inductive bias."""
    all_features = sorted(set().union(*(set(r["Feature"]) for r in rankings.values())))
    comp = pd.DataFrame({"Feature": all_features})

    family_cols = []
    for family in FAMILY_NAMES:
        if family not in rankings:
            continue
        r = rankings[family].reset_index(drop=True)
        r["Rank"] = r.index + 1  # 1 = top-ranked for this family
        r = r.set_index("Feature")

        sel_col = f"Selected_{family}"
        comp[sel_col] = comp["Feature"].isin(r.index)
        comp[f"Rank_{family}"]  = comp["Feature"].map(r["Rank"])
        comp[f"Score_{family}"] = comp["Feature"].map(r["Composite_Score"]).fillna(0.0).round(6)
        family_cols.append(sel_col)

    comp["Num_Families_Selected"] = comp[family_cols].sum(axis=1)
    comp = comp.sort_values(
        ["Num_Families_Selected", "Feature"], ascending=[False, True]
    ).reset_index(drop=True)
    return comp


def plot_feature_selection_ranking(ranking, filepath, resample_tag, family_label):
    try:
        fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(ranking))))
        ordered = ranking.sort_values("Composite_Score", ascending=True)
        ax.barh(ordered["Feature"], ordered["Composite_Score"], color=COLOR_TUNED, edgecolor="black")
        ax.set_xlabel("Composite Score (0.45*Spearman + 0.45*PermImportance - 0.10*redundancy)")
        ax.set_title(
            f"Feature Selection Ranking [{family_label}] (multicollinearity-filtered, "
            f"permutation-importance-ranked)\n"
            f"Resample: {resample_tag}"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_feature_selection_ranking failed: {e}")
        plt.close("all")


# =========================
# LIVE OPTUNA TUNING
# One study per model family, run fresh on this dataset's actual
# chronological train fold. Objective = mean k-fold CV score (neg RMSE,
# maximised = RMSE minimised) -- TUNE_CV_FOLDS controls the fold count.
# =========================

def tune_random_forest(X_train, y_train, n_trials):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 800),
            max_depth         = trial.suggest_int("max_depth", 3, 30),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
            max_features      = trial.suggest_float("max_features", 0.3, 1.0),
        )
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf",  RandomForestRegressor(random_state=42, n_jobs=1, **params)),
        ])
        return cross_val_score(pipe, X_train, y_train,
                               cv=TUNE_CV_FOLDS, scoring=TUNE_METRIC, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials,
                   timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def tune_xgboost(X_train, y_train, n_trials):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 1000),
            max_depth        = trial.suggest_int("max_depth", 2, 10),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # gamma is a minimum-loss-reduction threshold in absolute
            # squared-error units. A linear [0, 5] range puts ~99% of its
            # mass above the largest split gain XGBoost ever sees on this
            # target's scale, collapsing most trials to a constant
            # predictor -- log-scaled and rescaled to the gains actually
            # observed on this project's data (see 30m_phase1.py's gamma
            # fix for the full measurement).
            gamma            = trial.suggest_float("gamma", 1e-6, 0.05, log=True),
        )
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("xgb", XGBRegressor(random_state=42, verbosity=0,
                                 eval_metric="rmse", n_jobs=1, **params)),
        ])
        return cross_val_score(pipe, X_train, y_train,
                               cv=TUNE_CV_FOLDS, scoring=TUNE_METRIC, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials,
                   timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def tune_svr(X_train, y_train, n_trials):
    X_s, y_s = subsample_for_svr(X_train, y_train, MAX_SVR_ROWS)

    # epsilon is an absolute width in the TARGET's own units, not a
    # normalised fraction -- a fixed [1e-5, 1.0] search range is fine for a
    # target that spans O(1) units, but these frequency targets have a
    # std of only ~0.016 Hz (full range ~0.04 Hz). Any epsilon near or
    # above that spread lets SVR's epsilon-insensitive loss satisfy every
    # training point with a flat/constant predictor (w=0 trivially
    # minimises ||w||^2), which is exactly the "flat prediction" pattern
    # observed in the plots. Scale the upper bound to the actual target
    # spread so the search can't wander into that trivial-flat regime.
    y_std = float(np.std(y_s))
    eps_hi = max(y_std * 0.5, 1e-4)
    eps_lo = min(1e-5, eps_hi * 0.1)

    def objective(trial):
        kernel  = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
        C       = trial.suggest_float("C", 0.1, 500.0, log=True)
        epsilon = trial.suggest_float("epsilon", eps_lo, eps_hi, log=True)
        gamma   = trial.suggest_categorical("gamma", ["scale", "auto"])
        degree  = trial.suggest_int("degree", 2, 4) if kernel == "poly" else 3
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel=kernel, C=C, epsilon=epsilon,
                           gamma=gamma, degree=degree, max_iter=2000)),
        ])
        return cross_val_score(pipe, X_s, y_s,
                               cv=TUNE_CV_FOLDS, scoring=TUNE_METRIC, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials,
                   timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def tune_ridge(X_train, y_train, n_trials):
    def objective(trial):
        params = dict(
            alpha  = trial.suggest_float("alpha", 1e-3, 1e3, log=True),
            solver = trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "sag"]
            ),
        )
        pipe = Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(random_state=42, **params)),
        ])
        return cross_val_score(pipe, X_train, y_train,
                               cv=TUNE_CV_FOLDS, scoring=TUNE_METRIC, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=n_trials,
                   timeout=OPTUNA_TIMEOUT, show_progress_bar=False)
    return study.best_params


def optuna_tune_all(target_label, family_X_train, y_train, n_trials):
    """Run the 4 Optuna studies for this target's train fold -- each family
    tuned on its OWN selected features (family_X_train[family]), not a
    shared X. Returns ({'rf': {...}, 'xgb': {...}, 'svr': {...},
    'ridge': {...}} of freshly-discovered params, {'rf': secs, 'xgb': secs,
    'svr': secs, 'ridge': secs} of wall-clock tuning time per family).

    This tuning search (n_trials trials x k-fold CV, per family) is the
    dominant cost of a run -- the final run_target() fit afterwards uses
    only the single best param set it found, so it's comparatively instant.
    Timed separately here so the two numbers can be reported apart instead
    of the fast final-fit time misleadingly looking like the model's whole
    cost.
    """
    tuned        = {}
    tuning_times = {}

    t_start = time.time()
    if OPTUNA_OK:
        print(f"    [Optuna] Tuning Random Forest for {target_label} ({n_trials} trials) ...")
        try:
            tuned["rf"] = tune_random_forest(family_X_train["rf"], y_train, n_trials)
            print(f"      Best RF params : {tuned['rf']}")
        except Exception as e:
            print(f"      Warning: RF tuning failed ({e}), using fallback defaults.")
            tuned["rf"] = dict(FALLBACK_TUNED_PARAMS["rf"])
    else:
        tuned["rf"] = dict(FALLBACK_TUNED_PARAMS["rf"])
    tuning_times["rf"] = time.time() - t_start
    print(f"      RF tuning time: {fmt_elapsed(tuning_times['rf'])}")

    t_start = time.time()
    if XGB_OK:
        if OPTUNA_OK:
            print(f"    [Optuna] Tuning XGBoost for {target_label} ({n_trials} trials) ...")
            try:
                tuned["xgb"] = tune_xgboost(family_X_train["xgb"], y_train, n_trials)
                print(f"      Best XGB params: {tuned['xgb']}")
            except Exception as e:
                print(f"      Warning: XGB tuning failed ({e}), using fallback defaults.")
                tuned["xgb"] = dict(FALLBACK_TUNED_PARAMS["xgb"])
        else:
            tuned["xgb"] = dict(FALLBACK_TUNED_PARAMS["xgb"])
    else:
        tuned["xgb"] = dict(FALLBACK_TUNED_PARAMS["xgb"])
    tuning_times["xgb"] = time.time() - t_start
    print(f"      XGB tuning time: {fmt_elapsed(tuning_times['xgb'])}")

    t_start = time.time()
    if OPTUNA_OK:
        print(f"    [Optuna] Tuning SVR for {target_label} ({n_trials} trials) ...")
        try:
            tuned["svr"] = tune_svr(family_X_train["svr"], y_train, n_trials)
            print(f"      Best SVR params: {tuned['svr']}")
        except Exception as e:
            print(f"      Warning: SVR tuning failed ({e}), using fallback defaults.")
            tuned["svr"] = dict(FALLBACK_TUNED_PARAMS["svr"])
    else:
        tuned["svr"] = dict(FALLBACK_TUNED_PARAMS["svr"])
    tuning_times["svr"] = time.time() - t_start
    print(f"      SVR tuning time: {fmt_elapsed(tuning_times['svr'])}")

    t_start = time.time()
    if OPTUNA_OK:
        print(f"    [Optuna] Tuning Ridge for {target_label} ({n_trials} trials) ...")
        try:
            tuned["ridge"] = tune_ridge(family_X_train["ridge"], y_train, n_trials)
            print(f"      Best Ridge params: {tuned['ridge']}")
        except Exception as e:
            print(f"      Warning: Ridge tuning failed ({e}), using fallback defaults.")
            tuned["ridge"] = dict(FALLBACK_TUNED_PARAMS["ridge"])
    else:
        tuned["ridge"] = dict(FALLBACK_TUNED_PARAMS["ridge"])
    tuning_times["ridge"] = time.time() - t_start
    print(f"      Ridge tuning time: {fmt_elapsed(tuning_times['ridge'])}")

    return tuned, tuning_times


# =========================
# BUILD ALL MODELS (TUNED + BASELINE)
# TUNED   (4): RF, XGB, SVR, Ridge -- params discovered live via Optuna
# BASELINE (4): RF, XGB, SVR, Ridge -- fixed default params
# TOTAL   : 8 models => 8 bars per chart
# =========================

RF_PARAM_KEYS    = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"]
XGB_PARAM_KEYS   = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                    "min_child_weight", "reg_alpha", "reg_lambda", "gamma"]
SVR_PARAM_KEYS   = ["kernel", "C", "epsilon", "gamma"]
RIDGE_PARAM_KEYS = ["alpha", "solver"]

def _logged_params(estimator, keys):
    p = estimator.get_params()
    return {k: p[k] for k in keys if k in p}


def build_models_tuned(tuned_params):
    """Returns a list of (name, pipeline, params, family) 4-tuples -- the
    family tag tells callers which of family_X's per-family feature set
    this model was selected/tuned on, since each family now has its own
    features (see select_features_all_families)."""
    rf_p    = tuned_params["rf"]
    xgb_p   = tuned_params["xgb"]
    svr_p   = tuned_params["svr"]
    ridge_p = tuned_params["ridge"]
    mdls    = []

    # ── TUNED: Random Forest ──────────────────────────────────────────────
    mdls.append(("Random Forest (Tuned)", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf",  RandomForestRegressor(random_state=42, n_jobs=1, **rf_p)),
    ]), rf_p, "rf"))

    # ── TUNED: XGBoost ───────────────────────────────────────────────────
    if XGB_OK:
        mdls.append(("XGBoost (Tuned)", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("xgb", XGBRegressor(random_state=42, verbosity=0,
                                 eval_metric="rmse", n_jobs=1, **xgb_p)),
        ]), xgb_p, "xgb"))
    else:
        mdls.append(("Gradient Boosting (Tuned)", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ]), {}, "xgb"))

    # ── TUNED: SVR ───────────────────────────────────────────────────────
    svr_kwargs = dict(max_iter=2000, **svr_p)
    svr_kwargs.setdefault("kernel", "rbf")
    mdls.append(("SVR (Tuned)", Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr",    SVR(**svr_kwargs)),
    ]), svr_p, "svr"))

    # ── TUNED: Ridge ─────────────────────────────────────────────────────
    mdls.append(("Ridge (Tuned)", Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(random_state=42, **ridge_p)),
    ]), ridge_p, "ridge"))

    # ── BASELINE: Random Forest ───────────────────────────────────────────
    rf_baseline = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    mdls.append(("Random Forest (baseline)", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf",  rf_baseline),
    ]), _logged_params(rf_baseline, RF_PARAM_KEYS), "rf"))

    # ── BASELINE: XGBoost ────────────────────────────────────────────────
    if XGB_OK:
        xgb_baseline = XGBRegressor(n_estimators=600, max_depth=4, learning_rate=0.05,
                                    subsample=0.9, colsample_bytree=0.9,
                                    random_state=42, verbosity=0,
                                    eval_metric="rmse", n_jobs=1)
        mdls.append(("XGBoost (baseline)", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("xgb", xgb_baseline),
        ]), _logged_params(xgb_baseline, XGB_PARAM_KEYS), "xgb"))
    else:
        mdls.append(("Gradient Boosting (baseline)", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ]), {}, "xgb"))

    # ── BASELINE: SVR ────────────────────────────────────────────────────
    svr_baseline = SVR(kernel="rbf", C=50, gamma="scale", epsilon=0.001, max_iter=2000)
    mdls.append(("SVR (baseline)", Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr",    svr_baseline),
    ]), _logged_params(svr_baseline, SVR_PARAM_KEYS), "svr"))

    # ── BASELINE: Ridge ──────────────────────────────────────────────────
    ridge_baseline = Ridge(alpha=1.0, random_state=42)
    mdls.append(("Ridge (baseline)", Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge",  ridge_baseline),
    ]), _logged_params(ridge_baseline, RIDGE_PARAM_KEYS), "ridge"))

    return mdls   # 8 entries total


# =========================
# TRAIN + EVALUATE
# =========================

def run_target(target_label, family_X_train, family_X_test, y_train, y_test, tuned_params,
               tuning_times=None):
    """family_X_train / family_X_test: {'rf': DataFrame, 'xgb': ..., 'svr': ...,
    'ridge': ...} -- each model fits/predicts on its OWN family's feature set.
    y_train / y_test are shared (same event rows for every family).

    tuning_times: optional {'rf': secs, 'xgb': secs, 'svr': secs, 'ridge':
    secs} from optuna_tune_all() -- the Optuna hyperparameter search time
    for each family, which happened before this function was even called
    and otherwise wouldn't show up anywhere. Folded into the matching
    TUNED model's own Fit_Time_Sec below (never the baseline's, which
    wasn't searched) so that model's reported time reflects its real total
    cost, not just the comparatively instant final fit."""
    print(f"\n  {'─'*70}")
    print(f"  TARGET : {target_label}")
    print(f"  {'─'*70}")

    model_triples   = build_models_tuned(tuned_params)
    rows, preds     = {}, {}
    best_params_log = {}
    fitted_models   = []

    for name, model, bparams, family in model_triples:
        print(f"    Fitting : {name} [{family}] ...")
        model_start_time = time.time()
        X_train = family_X_train[family]
        X_test  = family_X_test[family]
        try:
            if family == "svr":
                X_fit, y_fit = subsample_for_svr(X_train, y_train, MAX_SVR_ROWS)
            elif family in ("rf", "xgb"):
                X_fit, y_fit = subsample_for_trees(X_train, y_train, MAX_TREE_ROWS)
            else:
                X_fit, y_fit = X_train, y_train
            model.fit(X_fit, y_fit)
            yp = model.predict(X_test)
        except Exception as e:
            print(f"      Warning: {name} failed: {e}")
            continue

        preds[name] = yp
        rows[name]  = calc_metrics(y_test, yp)

        try:
            cv_scores = cross_val_score(model, X_fit, y_fit,
                                        cv=CV_VARIABILITY_FOLDS, scoring="r2", n_jobs=1)
            rows[name]["CV_R2_Mean"] = cv_scores.mean()
            rows[name]["CV_R2_Std"]  = cv_scores.std()
        except Exception as e:
            rows[name]["CV_R2_Mean"] = np.nan
            rows[name]["CV_R2_Std"]  = np.nan
            print(f"      Warning: CV scoring failed for {name}: {e}")

        rows[name]["Fit_Time_Sec"] = time.time() - model_start_time
        if tuning_times and is_tuned(name) and family in tuning_times:
            rows[name]["Fit_Time_Sec"] += tuning_times[family]

        best_params_log[name] = bparams
        fitted_models.append((name, model))
        m = rows[name]
        print(f"      RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}  "
              f"MAPE={m['MAPE_%']:.4f}%  R2={m['R2']:.6f}  "
              f"Accuracy={m['Accuracy_%']:.2f}%  "
              f"CV_R2={m['CV_R2_Mean']:.4f}±{m['CV_R2_Std']:.4f}  "
              f"Time={fmt_elapsed(m['Fit_Time_Sec'])}")

    if not rows:
        raise RuntimeError("All models failed for this target.")

    results = (
        pd.DataFrame(rows).T
          .reset_index().rename(columns={"index": "Model"})
          .sort_values("RMSE").reset_index(drop=True)
    )
    print(f"\n  SUMMARY -- {target_label}")
    print(results.to_string(index=False))
    return results, preds, rows, fitted_models, best_params_log


# =========================
# PLOTS
# =========================

def plot_accuracy_bar(rows, target_label, filepath, resample_tag):
    """
    6-bar chart: 3 tuned (dark blue) + 3 baseline (light blue).
    Each bar labelled with accuracy % and [T]/[B] tag.
    """
    try:
        names  = list(rows.keys())
        accs   = [rows[n]["Accuracy_%"] for n in names]
        colors = [COLOR_TUNED if is_tuned(n) else COLOR_BASELINE for n in names]

        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos   = np.arange(len(names))
        bars    = ax.bar(x_pos, accs, color=colors, edgecolor="black", width=0.55)

        for bar, val, name in zip(bars, accs, names):
            tag = "[T]" if is_tuned(name) else "[B]"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.2f}%\n{tag}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [n.replace(" (", "\n(") for n in names],
            rotation=0, ha="center", fontsize=9,
        )
        ax.set_ylim(0, 125)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(
            f"Model Accuracy Comparison — {target_label}\n"
            f"Resample: {resample_tag} | Each family uses its own top-{TOP_K_FEATURES} features | "
            f"[T] = Optuna Tuned   [B] = Baseline (default params)",
            fontsize=11,
        )
        ax.axhline(y=100, color="red", linestyle="--", linewidth=1.2, label="100% line")
        ax.legend(handles=[
            Patch(color=COLOR_TUNED,    label="Optuna-Tuned [T]"),
            Patch(color=COLOR_BASELINE, label="Baseline [B]"),
            plt.Line2D([0], [0], color="red", linestyle="--", label="100% line"),
        ], fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_accuracy_bar failed: {e}")
        plt.close("all")


def plot_model_fit_time(rows, target_label, filepath, resample_tag):
    """
    8-bar chart of wall-clock time taken per model (rows[name]["Fit_Time_Sec"]):
    for a Tuned model this is its Optuna hyperparameter search time PLUS its
    final fit/predict/CV-scoring time (search dominates); for a Baseline
    model -- never searched -- it's just fit/predict/CV-scoring, so it will
    read far smaller than its Tuned counterpart almost by construction, not
    because it's a "faster model". Same layout/coloring as plot_accuracy_bar
    (tuned = dark blue, baseline = light blue), so the two figures read side
    by side: which model performs best vs. which model costs the most
    compute.
    """
    try:
        names = list(rows.keys())
        times = [rows[n]["Fit_Time_Sec"] for n in names]
        colors = [COLOR_TUNED if is_tuned(n) else COLOR_BASELINE for n in names]

        fig, ax = plt.subplots(figsize=(14, 6))
        x_pos   = np.arange(len(names))
        bars    = ax.bar(x_pos, times, color=colors, edgecolor="black", width=0.55)

        for bar, val, name in zip(bars, times, names):
            tag = "[T]" if is_tuned(name) else "[B]"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.01,
                f"{fmt_elapsed(val)}\n{tag}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [n.replace(" (", "\n(") for n in names],
            rotation=0, ha="center", fontsize=9,
        )
        ax.set_ylabel("Time taken (seconds)", fontsize=11)
        ax.set_title(
            f"Model Fit Time Comparison — {target_label}\n"
            f"Resample: {resample_tag} | [T] = Optuna search + final fit/predict/"
            f"{CV_VARIABILITY_FOLDS}-fold CV   [B] = fit/predict/CV only (never searched)",
            fontsize=11,
        )
        ax.legend(handles=[
            Patch(color=COLOR_TUNED,    label="Optuna-Tuned [T]"),
            Patch(color=COLOR_BASELINE, label="Baseline [B]"),
        ], fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_model_fit_time failed: {e}")
        plt.close("all")


def plot_best_scatter(results, preds, rows, y_test, target_label, filepath, resample_tag):
    try:
        best_name = results.iloc[0]["Model"]
        best_pred = preds[best_name]
        y_arr     = np.array(y_test)
        fig, ax   = plt.subplots(figsize=(8, 6))
        ax.scatter(y_arr, best_pred, s=60, alpha=0.75, color="steelblue", label="Predictions")
        lo = min(y_arr.min(), best_pred.min())
        hi = max(y_arr.max(), best_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=2, label="Perfect fit (y=x)")
        ax.set_xlabel("Actual Frequency (Hz)")
        ax.set_ylabel("Predicted Frequency (Hz)")
        ax.set_title(
            f"Best Model: {best_name}\n{target_label} | Resample: {resample_tag}\n"
            f"Accuracy={rows[best_name]['Accuracy_%']:.2f}%  R2={rows[best_name]['R2']:.4f}"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_best_scatter failed: {e}")
        plt.close("all")


def plot_scatter_grid(fitted_models, preds, rows, y_test, target_label, filepath, resample_tag):
    try:
        y_arr = np.array(y_test)
        n     = len(fitted_models)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.array(axes).flatten()

        for i, (name, _) in enumerate(fitted_models):
            ax     = axes[i]
            y_pred = preds[name]
            color  = COLOR_TUNED if is_tuned(name) else COLOR_BASELINE
            ax.scatter(y_arr, y_pred, s=30, alpha=0.60, color=color)
            lo = min(y_arr.min(), y_pred.min())
            hi = max(y_arr.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
            tag = "[T]" if is_tuned(name) else "[B]"
            ax.set_title(
                f"{name} {tag}\nR2={rows[name]['R2']:.4f}  RMSE={rows[name]['RMSE']:.6f}\n"
                f"Acc={rows[name]['Accuracy_%']:.2f}%",
                fontsize=8,
            )
            ax.set_xlabel("Actual (Hz)", fontsize=8)
            ax.set_ylabel("Predicted (Hz)", fontsize=8)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Actual vs Predicted — All Models\n{target_label} | Resample: {resample_tag}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_scatter_grid failed: {e}")
        plt.close("all")


def plot_pred_vs_measured_best_line(results, preds, rows, y_test, target_label, filepath, resample_tag):
    try:
        best_name = results.iloc[0]["Model"]
        best_pred = preds[best_name]
        y_arr     = np.array(y_test)
        idx       = np.arange(len(y_arr))
        color     = COLOR_TUNED if is_tuned(best_name) else COLOR_BASELINE

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(idx, y_arr, color="#6b7a8f", linewidth=1.4, label="Measured")
        ax.plot(idx, best_pred, color=color, linewidth=1.4, label=f"Predicted ({best_name})")
        ax.set_xlabel("Test Event Index")
        ax.set_ylabel("Natural Frequency (Hz)")
        ax.set_title(
            f"Predicted vs. Measured Natural Frequency — {target_label}\n"
            f"Resample: {resample_tag} | Best Model: {best_name}  "
            f"(R2={rows[best_name]['R2']:.4f}, Accuracy={rows[best_name]['Accuracy_%']:.2f}%)",
            fontsize=12, fontweight="bold",
        )
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_pred_vs_measured_best_line failed: {e}")
        plt.close("all")


def plot_pred_vs_measured_grid(fitted_models, preds, rows, y_test, target_label, filepath, resample_tag):
    try:
        y_arr = np.array(y_test)
        idx   = np.arange(len(y_arr))
        n     = len(fitted_models)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
        axes = np.array(axes).flatten()

        for i, (name, _) in enumerate(fitted_models):
            ax     = axes[i]
            y_pred = preds[name]
            color  = COLOR_TUNED if is_tuned(name) else COLOR_BASELINE
            ax.plot(idx, y_arr, color="#6b7a8f", linewidth=1.2, label="Measured")
            ax.plot(idx, y_pred, color=color, linewidth=1.2, label="Predicted")
            tag = "[T]" if is_tuned(name) else "[B]"
            ax.set_title(f"{name} {tag}\nR2={rows[name]['R2']:.4f}  RMSE={rows[name]['RMSE']:.6f}", fontsize=9)
            ax.set_xlabel("Test Event Index", fontsize=8)
            ax.set_ylabel("Natural Frequency (Hz)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Predicted vs. Measured Natural Frequency — All Models\n"
            f"{target_label} | Resample: {resample_tag}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_pred_vs_measured_grid failed: {e}")
        plt.close("all")


def plot_pred_vs_measured_grid_tuned(fitted_models, preds, rows, y_test, target_label, filepath, resample_tag):
    """Same line-plot grid as plot_pred_vs_measured_grid, but restricted to
    the 4 Optuna-Tuned models -- baseline models excluded."""
    try:
        tuned_models = [(n, m) for n, m in fitted_models if is_tuned(n)]
        if not tuned_models:
            return
        y_arr = np.array(y_test)
        idx   = np.arange(len(y_arr))
        n     = len(tuned_models)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5 * nrows))
        axes = np.array(axes).flatten()

        for i, (name, _) in enumerate(tuned_models):
            ax     = axes[i]
            y_pred = preds[name]
            ax.plot(idx, y_arr, color="#6b7a8f", linewidth=1.2, label="Measured")
            ax.plot(idx, y_pred, color=COLOR_TUNED, linewidth=1.2, label="Predicted")
            ax.set_title(f"{name}\nR2={rows[name]['R2']:.4f}  RMSE={rows[name]['RMSE']:.6f}", fontsize=10)
            ax.set_xlabel("Test Event Index", fontsize=9)
            ax.set_ylabel("Natural Frequency (Hz)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Predicted vs. Measured Natural Frequency — Tuned Models Only\n"
            f"{target_label} | Resample: {resample_tag}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_pred_vs_measured_grid_tuned failed: {e}")
        plt.close("all")


def plot_pred_vs_measured_combined_tuned(fitted_models, preds, rows, y_test, event_no,
                                         target_label, filepath, resample_tag):
    """All 4 Optuna-Tuned models overlaid on ONE panel (each its own colour
    from MODEL_PALETTE) against the shared Measured line, so the models can
    be compared directly instead of eyeballed across separate subplots.

    X-axis is elapsed time since the first test event (elapsed_time_axis()),
    not a bare row index -- each Event_No step is one EVENT_WINDOW_MINUTES
    chronological window, since no real per-event clock time survives into
    the engineered feature set."""
    try:
        tuned_models = [(n, m) for n, m in fitted_models if is_tuned(n)]
        if not tuned_models:
            return
        y_arr   = np.array(y_test)
        elapsed = elapsed_time_axis(event_no)

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(elapsed, y_arr, color="black", linewidth=1.8, label="Measured", zorder=10)

        for i, (name, _) in enumerate(tuned_models):
            color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
            ax.plot(elapsed, preds[name], color=color, linewidth=1.2, alpha=0.85,
                   label=f"{name}  (R2={rows[name]['R2']:.3f})")

        ax.xaxis.set_major_formatter(FuncFormatter(fmt_elapsed_dhm))
        ax.set_xlabel("Elapsed Time Since First Test Event (chronological, "
                     f"{EVENT_WINDOW_MINUTES}-min windows)", fontsize=10)
        ax.set_ylabel("Natural Frequency (Hz)", fontsize=10)
        ax.set_title(
            f"Predicted vs. Measured Natural Frequency — All Tuned Models Combined\n"
            f"{target_label} | Resample: {resample_tag}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_pred_vs_measured_combined_tuned failed: {e}")
        plt.close("all")


def plot_pred_vs_measured_segments_tuned(fitted_models, preds, rows, y_test, event_no,
                                         target_label, filepath, resample_tag, n_segments=5):
    """Same combined multi-model overlay as plot_pred_vs_measured_combined_tuned,
    split into n_segments contiguous chronological chunks (start -> end)
    stacked in one figure -- individual model lines are unreadable when
    ~100+ test events are compressed into one wide panel, so this trades
    "everything in one glance" for "each region actually legible"."""
    try:
        tuned_models = [(n, m) for n, m in fitted_models if is_tuned(n)]
        if not tuned_models:
            return
        y_arr   = np.array(y_test)
        elapsed = elapsed_time_axis(event_no)
        n_rows  = len(y_arr)
        seg_idx_groups = np.array_split(np.arange(n_rows), n_segments)

        default_labels = ["Start", "Early", "Middle", "Late", "End"]
        seg_labels = default_labels if n_segments == 5 else \
            [f"Segment {i + 1}" for i in range(n_segments)]

        fig, axes = plt.subplots(n_segments, 1, figsize=(14, 3.2 * n_segments), sharey=True)
        axes = np.atleast_1d(axes)

        for i, seg_idx in enumerate(seg_idx_groups):
            ax          = axes[i]
            seg_elapsed = elapsed[seg_idx]
            ax.plot(seg_elapsed, y_arr[seg_idx], color="black", linewidth=1.8,
                   label="Measured", zorder=10)
            for j, (name, _) in enumerate(tuned_models):
                color = MODEL_PALETTE[j % len(MODEL_PALETTE)]
                ax.plot(seg_elapsed, np.asarray(preds[name])[seg_idx], color=color,
                       linewidth=1.2, alpha=0.85, label=name)
            ax.xaxis.set_major_formatter(FuncFormatter(fmt_elapsed_dhm))
            ax.set_title(
                f"{seg_labels[i]}  ({fmt_elapsed_dhm(seg_elapsed.min())} "
                f"– {fmt_elapsed_dhm(seg_elapsed.max())})",
                fontsize=10,
            )
            ax.set_ylabel("Nat. Freq (Hz)", fontsize=9)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8, loc="upper right", ncol=len(tuned_models) + 1)

        axes[-1].set_xlabel("Elapsed Time Since First Test Event", fontsize=10)

        plt.suptitle(
            f"Predicted vs. Measured — Tuned Models by Time Segment "
            f"({n_segments}: Start to End)\n{target_label} | Resample: {resample_tag}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_pred_vs_measured_segments_tuned failed: {e}")
        plt.close("all")


def plot_feature_importance(fitted_models, rf_labels, target_label, filepath, resample_tag):
    """Uses Tuned RF feature importances; falls back to baseline RF if missing.
    rf_labels must be the RF family's own selected feature labels (FEATURE_LABELS_BY_FAMILY['rf'])."""
    try:
        rf_pipe = next(
            (m for n, m in fitted_models if "Random Forest (Tuned)" in n), None
        ) or next(
            (m for n, m in fitted_models if "Random Forest" in n), None
        )
        if rf_pipe is None:
            return
        importances = rf_pipe.named_steps["rf"].feature_importances_
        indices     = np.argsort(importances)[::-1]
        sorted_lbls = [rf_labels[i] for i in indices]
        sorted_vals = importances[indices]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sorted_lbls, sorted_vals, color="steelblue", edgecolor="black")
        ax.set_ylabel("Importance Score")
        ax.set_title(f"Random Forest (Tuned) — Feature Importance\n{target_label} | Resample: {resample_tag}")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_feature_importance failed: {e}")
        plt.close("all")


def plot_learning_curves(model_triples, family_X_train, y_train, target_label, filepath, resample_tag):
    """family_X_train: {'rf': DataFrame, 'xgb': ..., 'svr': ..., 'ridge': ...}
    -- each model's curve is computed on its OWN family's feature set."""
    try:
        n     = len(model_triples)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = np.array(axes).flatten()

        curve_data  = {}
        n_rows_used = {}
        y_lo, y_hi = np.inf, -np.inf
        for name, model, _, family in model_triples:
            X_lc, y_lc = _subsample(family_X_train[family], y_train, DIAG_MAX_ROWS, "LearningCurve")
            n_rows_used[family] = len(X_lc)
            try:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_lc, y_lc, cv=DIAG_CV_FOLDS, scoring="r2",
                    train_sizes=LEARNING_CURVE_TRAIN_SIZES,
                    n_jobs=1, shuffle=True, random_state=42,
                )
            except Exception as e:
                curve_data[name] = None
                print(f"      Warning: learning_curve failed for {name}: {e}")
                continue
            train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
            val_mean,   val_std   = val_scores.mean(axis=1),   val_scores.std(axis=1)
            curve_data[name] = (train_sizes, train_mean, train_std, val_mean, val_std)
            y_lo = min(y_lo, (train_mean - train_std).min(), (val_mean - val_std).min())
            y_hi = max(y_hi, (train_mean + train_std).max(), (val_mean + val_std).max())

        if not np.isfinite(y_lo) or not np.isfinite(y_hi):
            y_lo, y_hi = 0.0, 1.0
        pad = 0.05 * max(y_hi - y_lo, 1e-6)
        shared_ylim = (y_lo - pad, y_hi + pad)

        for i, (name, model, _, family) in enumerate(model_triples):
            ax   = axes[i]
            data = curve_data.get(name)
            tag  = "[T]" if is_tuned(name) else "[B]"
            if data is None:
                ax.set_title(f"{name} {tag}\n(failed)", fontsize=8)
                ax.axis("off")
                continue

            train_sizes, train_mean, train_std, val_mean, val_std = data
            color = COLOR_TUNED if is_tuned(name) else COLOR_BASELINE
            ax.plot(train_sizes, train_mean, "o-", color="dimgray", label="Training score")
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                            color="dimgray", alpha=0.15)
            ax.plot(train_sizes, val_mean, "o-", color=color, label="Validation score")
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                            color=color, alpha=0.25)

            ax.set_ylim(shared_ylim)
            ax.set_title(f"{name} {tag}", fontsize=9)
            ax.set_xlabel("Training examples", fontsize=8)
            ax.set_ylabel("R2 score", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Learning Curves — Training vs Validation Score\n"
            f"{target_label} | Resample: {resample_tag} | "
            f"(subsampled to {DIAG_MAX_ROWS:,} rows per family, {DIAG_CV_FOLDS}-fold CV)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_learning_curves failed: {e}")
        plt.close("all")


def plot_validation_curve(tuned_params, family_X_train, y_train, target_label, filepath, resample_tag):
    """
    For each model family (RF, XGBoost, SVR, Ridge), sweep one key hyperparameter
    while holding the rest at their tuned values, and plot train vs CV score
    across the sweep -- each family swept on its OWN feature set
    (family_X_train[family]). Reveals under-fitting / over-fitting regions for
    that hyperparameter.
    """
    try:
        families = ["Random Forest"] + (["XGBoost"] if XGB_OK else []) + ["SVR", "Ridge"]
        family_keys = {"Random Forest": "rf", "XGBoost": "xgb", "SVR": "svr", "Ridge": "ridge"}
        fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5))
        axes = np.atleast_1d(axes)

        for i, fam in enumerate(families):
            ax   = axes[i]
            spec = VALIDATION_CURVE_SPECS[fam]
            X_vc, y_vc = _subsample(family_X_train[family_keys[fam]], y_train,
                                    DIAG_MAX_ROWS, f"ValidationCurve[{fam}]")

            if fam == "Random Forest":
                base_p = {k: v for k, v in tuned_params["rf"].items() if k != spec["param"]}
                est = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("rf",  RandomForestRegressor(random_state=42, n_jobs=1, **base_p)),
                ])
            elif fam == "XGBoost":
                base_p = {k: v for k, v in tuned_params["xgb"].items() if k != spec["param"]}
                est = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("xgb", XGBRegressor(random_state=42, verbosity=0,
                                         eval_metric="rmse", n_jobs=1, **base_p)),
                ])
            elif fam == "SVR":
                base_p = {k: v for k, v in tuned_params["svr"].items() if k != spec["param"]}
                svr_kwargs = dict(max_iter=2000, **base_p)
                svr_kwargs.setdefault("kernel", "rbf")
                est = Pipeline([
                    ("imp",    SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("svr",    SVR(**svr_kwargs)),
                ])
            else:  # Ridge
                base_p = {k: v for k, v in tuned_params["ridge"].items() if k != spec["param"]}
                est = Pipeline([
                    ("imp",    SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge",  Ridge(random_state=42, **base_p)),
                ])

            param_name = f"{spec['step']}__{spec['param']}"
            try:
                train_scores, val_scores = validation_curve(
                    est, X_vc, y_vc, param_name=param_name, param_range=spec["range"],
                    cv=DIAG_CV_FOLDS, scoring="r2", n_jobs=1,
                )
            except Exception as e:
                ax.set_title(f"{fam}\n(failed: {e})", fontsize=8)
                ax.axis("off")
                continue

            train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
            val_mean,   val_std   = val_scores.mean(axis=1),   val_scores.std(axis=1)
            xr = spec["range"]

            ax.plot(xr, train_mean, "o-", color="dimgray", label="Training score")
            ax.fill_between(xr, train_mean - train_std, train_mean + train_std,
                            color="dimgray", alpha=0.15)
            ax.plot(xr, val_mean, "o-", color=COLOR_TUNED, label="CV score")
            ax.fill_between(xr, val_mean - val_std, val_mean + val_std,
                            color=COLOR_TUNED, alpha=0.25)

            if fam == "SVR":
                ax.set_xscale("log")
            ax.set_xlabel(spec["param"], fontsize=9)
            ax.set_ylabel("R2 score", fontsize=9)
            ax.set_title(fam, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle(
            f"Validation Curves — Hyperparameter Sensitivity\n"
            f"{target_label} | Resample: {resample_tag} | "
            f"(subsampled to {DIAG_MAX_ROWS:,} rows per family, {DIAG_CV_FOLDS}-fold CV)",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_validation_curve failed: {e}")
        plt.close("all")


def plot_cv_score_variability(model_triples, family_X_train, y_train, target_label, filepath, resample_tag):
    """
    k-fold CV R2 scores per model as a boxplot + jittered points, showing how
    stable/variable each model's performance is across folds (not just a
    single train/test split). Each model is scored on its OWN family's
    feature set (family_X_train[family]).
    """
    try:
        names, fold_scores, colors = [], [], []

        for name, model, _, family in model_triples:
            X_train = family_X_train[family]
            if family == "svr":
                X_cv, y_cv = subsample_for_svr(X_train, y_train, MAX_SVR_ROWS)
            elif family in ("rf", "xgb"):
                X_cv, y_cv = subsample_for_trees(X_train, y_train, MAX_TREE_ROWS)
            else:
                X_cv, y_cv = X_train, y_train
            try:
                scores = cross_val_score(model, X_cv, y_cv, cv=CV_VARIABILITY_FOLDS,
                                         scoring="r2", n_jobs=1)
            except Exception as e:
                print(f"      Warning: CV failed for {name}: {e}")
                continue
            names.append(name)
            fold_scores.append(scores)
            colors.append(COLOR_TUNED if is_tuned(name) else COLOR_BASELINE)

        if not names:
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        bp = ax.boxplot(fold_scores, patch_artist=True, showmeans=True,
                        labels=[n.replace(" (", "\n(") for n in names])
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")

        rng = np.random.default_rng(42)
        for i, scores in enumerate(fold_scores):
            jitter = rng.normal(0, 0.04, size=len(scores))
            ax.scatter(np.full(len(scores), i + 1) + jitter, scores,
                       color="black", alpha=0.6, s=18, zorder=3)

        ax.set_ylabel(f"R2 score ({CV_VARIABILITY_FOLDS}-fold CV)", fontsize=11)
        ax.set_title(
            f"Cross-Validation Score Variability — {target_label}\n"
            f"Resample: {resample_tag} | {CV_VARIABILITY_FOLDS}-fold CV",
            fontsize=11,
        )
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(fontsize=9)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_cv_score_variability failed: {e}")
        plt.close("all")


def plot_tuned_vs_baseline_paired(rows, target_label, filepath, resample_tag):
    """Side-by-side paired bar chart: Tuned vs Baseline for each model family."""
    try:
        families = ["Random Forest", "XGBoost" if XGB_OK else "Gradient Boosting", "SVR", "Ridge"]
        tuned_accs    = []
        baseline_accs = []
        labels        = []

        for fam in families:
            t_key = next((k for k in rows if fam in k and is_tuned(k)), None)
            b_key = next((k for k in rows if fam in k and not is_tuned(k)), None)
            if t_key and b_key:
                tuned_accs.append(rows[t_key]["Accuracy_%"])
                baseline_accs.append(rows[b_key]["Accuracy_%"])
                labels.append(fam)

        if not labels:
            return

        x     = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - width/2, tuned_accs,    width, color=COLOR_TUNED,
                    edgecolor="black", label="Tuned [T]")
        b2 = ax.bar(x + width/2, baseline_accs, width, color=COLOR_BASELINE,
                    edgecolor="black", label="Baseline [B]")

        for bar, val in zip(b1, tuned_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, val in zip(b2, baseline_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 120)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(
            f"Tuned vs Baseline Accuracy — {target_label}\n"
            f"Resample: {resample_tag}",
            fontsize=11,
        )
        ax.axhline(100, color="red", linestyle="--", linewidth=1)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_tuned_vs_baseline_paired failed: {e}")
        plt.close("all")


def plot_pred_vs_actual_tuned(fitted_models, preds, rows, y_test, target_label, filepath, resample_tag):
    """Predicted-vs-actual scatter grid for ONLY the 4 Optuna-Tuned models
    (Random Forest / XGBoost / SVR / Ridge) -- baseline models excluded."""
    try:
        tuned_models = [(n, m) for n, m in fitted_models if is_tuned(n)]
        if not tuned_models:
            return
        y_arr = np.array(y_test)
        n     = len(tuned_models)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes = np.array(axes).flatten()

        for i, (name, _) in enumerate(tuned_models):
            ax     = axes[i]
            y_pred = preds[name]
            ax.scatter(y_arr, y_pred, s=40, alpha=0.65, color=COLOR_TUNED,
                      edgecolor="black", linewidth=0.3, label="Predictions")
            lo = min(y_arr.min(), y_pred.min())
            hi = max(y_arr.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect fit (y=x)")
            ax.set_title(
                f"{name}\nR2={rows[name]['R2']:.4f}  RMSE={rows[name]['RMSE']:.6f}  "
                f"Acc={rows[name]['Accuracy_%']:.2f}%",
                fontsize=10,
            )
            ax.set_xlabel("Actual (Hz)", fontsize=9)
            ax.set_ylabel("Predicted (Hz)", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Predicted vs Actual — Tuned Models Only\n{target_label} | Resample: {resample_tag}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_pred_vs_actual_tuned failed: {e}")
        plt.close("all")


def plot_pred_vs_actual_combined_tuned(fitted_models, preds, rows, y_test, target_label,
                                       filepath, resample_tag):
    """All 4 Optuna-Tuned models' predicted-vs-actual scatter overlaid on
    ONE panel (each its own colour from MODEL_PALETTE) against a single
    shared y=x reference line -- lets you see directly whether one model's
    points sit systematically off the diagonal relative to the others,
    instead of eyeballing 4 separate axes with 4 separate diagonals."""
    try:
        tuned_models = [(n, m) for n, m in fitted_models if is_tuned(n)]
        if not tuned_models:
            return
        y_arr = np.array(y_test)

        fig, ax = plt.subplots(figsize=(8, 8))
        all_vals = [y_arr]
        for i, (name, _) in enumerate(tuned_models):
            y_pred = np.asarray(preds[name])
            all_vals.append(y_pred)
            color = MODEL_PALETTE[i % len(MODEL_PALETTE)]
            ax.scatter(y_arr, y_pred, s=35, alpha=0.6, color=color,
                      edgecolor="black", linewidth=0.3,
                      label=f"{name}  (R2={rows[name]['R2']:.3f})")

        lo = min(v.min() for v in all_vals)
        hi = max(v.max() for v in all_vals)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Perfect fit (y=x)")

        ax.set_xlabel("Actual (Hz)", fontsize=10)
        ax.set_ylabel("Predicted (Hz)", fontsize=10)
        ax.set_title(
            f"Predicted vs. Actual — All Tuned Models Combined\n"
            f"{target_label} | Resample: {resample_tag}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        save_plot(filepath)
    except Exception as e:
        print(f"    Warning: plot_pred_vs_actual_combined_tuned failed: {e}")
        plt.close("all")


# =========================
# MAIN LOOP
# =========================

all_results        = {}
summary_rows       = []
improvement_rows   = []
failed_files       = []
model_time_totals  = {}   # model name -> summed Fit_Time_Sec (tuning + final fit) across every target/dataset
tuning_time_totals = {}   # family ('rf'/'xgb'/'svr'/'ridge') -> summed Optuna search time alone

script_start_time = time.time()

for train_file, test_file, resample_tag in DATASETS:
    dataset_start_time = time.time()

    try:
        missing_input = [f for f in (train_file, test_file) if not os.path.exists(f)]
        if missing_input:
            print(f"\nWarning: File(s) not found, skipping dataset '{resample_tag}': {missing_input}")
            failed_files.append((f"{train_file}|{test_file}", "file not found"))
            continue

        print(f"\n{'='*80}")
        print(f"  TRAIN SOURCE: {train_file}")
        print(f"  TEST SOURCE : {test_file}")
        print(f"  Resample    : {resample_tag}")
        print(f"  Mode        : Live Optuna tuning ({N_TRIALS} trials, {TUNE_CV_FOLDS}-fold CV objective)")
        print(f"  SVR max rows: {MAX_SVR_ROWS}")
        print(f"{'='*80}")

        out_dir = os.path.join(BASE_OUTPUT_DIR, f"Phase_1_{resample_tag}_Machine_Learning_Analysis")
        os.makedirs(out_dir, exist_ok=True)

        # Pre-split train/test files loaded and used as-is -- no reshuffle,
        # no reslice. Only concatenated below for column/dtype bookkeeping;
        # feature selection runs on df_train_full alone (see below).
        df_train_full = load_csv_robust(train_file).reset_index(drop=True)
        df_test_full  = load_csv_robust(test_file).reset_index(drop=True)
        df_all        = pd.concat([df_train_full, df_test_full], ignore_index=True)
        print(f"  Columns: {list(df_all.columns)}")
        print(f"  Pool: {len(df_train_full)} train + {len(df_test_full)} test = {len(df_all)} rows")

        missing_targets = [c for c in TARGET_COLS if c not in df_all.columns]
        if missing_targets:
            print(f"  Warning: Missing target columns: {missing_targets} -- skipping.")
            failed_files.append((train_file, f"missing cols: {missing_targets}"))
            continue

        candidate_cols = [c for c in df_all.columns if c not in EXCLUDE_COLS + TARGET_COLS]
        for c in candidate_cols + TARGET_COLS:
            df_all[c]        = pd.to_numeric(df_all[c], errors="coerce")
            df_train_full[c] = pd.to_numeric(df_train_full[c], errors="coerce")
            df_test_full[c]  = pd.to_numeric(df_test_full[c], errors="coerce")

        # TRAIN ONLY -- multicollinearity clustering and permutation-
        # importance ranking both run on df_train_full alone. Running
        # either on the pooled train+test set would let the test set's
        # actual target values (and its correlation structure) influence
        # which features get selected -- target leakage.
        # Each model family gets its OWN top-K feature set (see
        # select_features_all_families / rank_features's model_family arg).
        feature_sets, rankings = select_features_all_families(
            df_train_full, candidate_cols, TARGET_COLS, TOP_K_FEATURES, out_dir
        )
        for family, ranking in rankings.items():
            plot_feature_selection_ranking(
                ranking, os.path.join(out_dir, f"feature_selection_ranking_{family}.png"),
                resample_tag, FAMILY_LABELS[family],
            )

        # Rows are dropped only for missing TARGETS -- never for a missing
        # feature value (each model's own SimpleImputer handles that). This
        # keeps the row set -- and therefore y_train/y_test -- identical
        # across every family, so only the FEATURE columns differ between
        # models, not the rows/targets being compared.
        train_data = df_train_full.dropna(subset=TARGET_COLS).reset_index(drop=True)
        test_data  = df_test_full.dropna(subset=TARGET_COLS).reset_index(drop=True)
        n_total = len(df_all)
        n_rows  = n_total
        print(f"  Total usable rows: {n_total} -> Train: {len(train_data)} | Test: {len(test_data)}")
        if len(train_data) < 10 or len(test_data) < 2:
            print(f"  Warning: Too few rows -- skipping.")
            failed_files.append((train_file, f"too few rows: train={len(train_data)} test={len(test_data)}"))
            continue

        family_X_train, family_X_test = {}, {}
        for family, feats in feature_sets.items():
            labels = [
                c.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                for c in feats
            ]
            FEATURE_COLS_BY_FAMILY[family]   = feats
            FEATURE_LABELS_BY_FAMILY[family] = labels
            X_tr = train_data[feats].copy(); X_tr.columns = labels
            X_te = test_data[feats].copy();  X_te.columns = labels
            family_X_train[family] = X_tr
            family_X_test[family]  = X_te

        yEW_train = train_data[NATF_EW_COL]
        yNS_train = train_data[NATF_NS_COL]
        yEW_test  = test_data[NATF_EW_COL]
        yNS_test  = test_data[NATF_NS_COL]

        tag_results = {}

        for target_label, y_train, y_test in [
            ("X Frequency (Hz)", yEW_train, yEW_test),
            ("Y Frequency (Hz)", yNS_train, yNS_test),
        ]:
            target_start_time = time.time()
            try:
                tparams, tuning_times = optuna_tune_all(target_label, family_X_train, y_train, N_TRIALS)
                results, preds, rows, fitted_models, best_params_log = run_target(
                    target_label, family_X_train, family_X_test, y_train, y_test, tparams,
                    tuning_times,
                )
            except Exception as e:
                print(f"\n  Error: run_target failed for {target_label}: {e}")
                traceback.print_exc()
                continue

            tag_results[target_label] = {
                "results": results, "preds": preds, "rows": rows,
                "fitted_models": fitted_models, "y_test": y_test,
            }

            for fam in tuning_times:
                tuning_time_totals[fam] = tuning_time_totals.get(fam, 0.0) + tuning_times[fam]

            for mname, m in rows.items():
                model_time_totals[mname] = model_time_totals.get(mname, 0.0) + m["Fit_Time_Sec"]

            tpfx = "XFreq" if "X" in target_label else "YFreq"
            pfx  = os.path.join(out_dir, tpfx)

            try:
                results.to_csv(f"{pfx}_metrics.csv", index=False)
                print(f"    Saved: {pfx}_metrics.csv")
            except Exception as e:
                print(f"    Warning: metrics CSV failed: {e}")

            try:
                params_rows = [{"Model": k, **v} for k, v in best_params_log.items()]
                pd.DataFrame(params_rows).to_csv(f"{pfx}_best_hyperparams.csv", index=False)
                print(f"    Saved: {pfx}_best_hyperparams.csv")
            except Exception as e:
                print(f"    Warning: hyperparams CSV failed: {e}")

            try:
                # No single shared X_test to dump feature values from any
                # more (each model used its own family's feature set) --
                # just the true/predicted/error/accuracy columns per model.
                pred_table = pd.DataFrame({"y_true": np.array(y_test)})
                eps = 1e-9
                for mname, yp in preds.items():
                    cn = safe_col(mname)
                    ae = np.abs(yp - np.array(y_test))
                    pred_table[f"{cn}_pred"]   = yp
                    pred_table[f"{cn}_abserr"] = ae
                    pred_table[f"{cn}_acc_%"]  = np.clip(
                        1.0 - ae / np.clip(np.abs(np.array(y_test)), eps, None), 0.0, 1.0,
                    ) * 100
                pred_table.to_csv(f"{pfx}_predictions.csv", index=False)
                print(f"    Saved: {pfx}_predictions.csv")
            except Exception as e:
                print(f"    Warning: predictions CSV failed: {e}")

            try:
                # Same true/predicted/error/accuracy layout as the CSV
                # above, but restricted to the 4 Optuna-Tuned models --
                # baseline models excluded.
                tuned_pred_table = pd.DataFrame({"y_true": np.array(y_test)})
                eps = 1e-9
                for mname, yp in preds.items():
                    if not is_tuned(mname):
                        continue
                    cn = safe_col(mname)
                    ae = np.abs(yp - np.array(y_test))
                    tuned_pred_table[f"{cn}_pred"]   = yp
                    tuned_pred_table[f"{cn}_abserr"] = ae
                    tuned_pred_table[f"{cn}_acc_%"]  = np.clip(
                        1.0 - ae / np.clip(np.abs(np.array(y_test)), eps, None), 0.0, 1.0,
                    ) * 100
                tuned_pred_table.to_csv(f"{pfx}_predictions_tuned_only.csv", index=False)
                print(f"    Saved: {pfx}_predictions_tuned_only.csv")
            except Exception as e:
                print(f"    Warning: tuned-only predictions CSV failed: {e}")

            print(f"\n    Generating plots for {target_label} ...")
            plot_accuracy_bar(rows, target_label, f"{pfx}_accuracy_bar.png", resample_tag)
            plot_model_fit_time(rows, target_label, f"{pfx}_fit_time.png", resample_tag)
            plot_tuned_vs_baseline_paired(rows, target_label,
                                          f"{pfx}_tuned_vs_baseline.png", resample_tag)
            plot_best_scatter(results, preds, rows, y_test, target_label,
                              f"{pfx}_best_scatter.png", resample_tag)
            plot_scatter_grid(fitted_models, preds, rows, y_test, target_label,
                              f"{pfx}_all_scatter.png", resample_tag)
            plot_pred_vs_actual_tuned(fitted_models, preds, rows, y_test, target_label,
                                      f"{pfx}_pred_vs_actual_tuned.png", resample_tag)
            plot_pred_vs_actual_combined_tuned(fitted_models, preds, rows, y_test, target_label,
                                               f"{pfx}_pred_vs_actual_combined.png", resample_tag)
            plot_pred_vs_measured_best_line(results, preds, rows, y_test, target_label,
                                            f"{pfx}_pred_vs_measured_best.png", resample_tag)
            plot_pred_vs_measured_grid(fitted_models, preds, rows, y_test, target_label,
                                       f"{pfx}_pred_vs_measured_all_models.png", resample_tag)
            plot_pred_vs_measured_grid_tuned(fitted_models, preds, rows, y_test, target_label,
                                             f"{pfx}_pred_vs_measured_tuned.png", resample_tag)
            plot_pred_vs_measured_combined_tuned(fitted_models, preds, rows, y_test,
                                                 test_data["Event_No"], target_label,
                                                 f"{pfx}_pred_vs_measured_combined.png", resample_tag)
            plot_pred_vs_measured_segments_tuned(fitted_models, preds, rows, y_test,
                                                 test_data["Event_No"], target_label,
                                                 f"{pfx}_pred_vs_measured_segments.png", resample_tag)
            plot_feature_importance(fitted_models, FEATURE_LABELS_BY_FAMILY["rf"], target_label,
                                    f"{pfx}_feature_importance.png", resample_tag)
            plot_learning_curves(build_models_tuned(tparams), family_X_train, y_train, target_label,
                                 f"{pfx}_learning_curves.png", resample_tag)
            plot_validation_curve(tparams, family_X_train, y_train, target_label,
                                  f"{pfx}_validation_curve.png", resample_tag)
            plot_cv_score_variability(build_models_tuned(tparams), family_X_train, y_train, target_label,
                                      f"{pfx}_cv_variability.png", resample_tag)

            best_name = results.iloc[0]["Model"]
            try:
                best_model_row = {
                    "Model":       best_name,
                    "RMSE":        rows[best_name]["RMSE"],
                    "MAE":         rows[best_name]["MAE"],
                    "MAPE_%":      rows[best_name]["MAPE_%"],
                    "R2":          rows[best_name]["R2"],
                    "Accuracy_%":  rows[best_name]["Accuracy_%"],
                    "CV_R2_Mean":  rows[best_name]["CV_R2_Mean"],
                    "CV_R2_Std":   rows[best_name]["CV_R2_Std"],
                    "Fit_Time_Sec": rows[best_name]["Fit_Time_Sec"],
                    **best_params_log[best_name],
                }
                pd.DataFrame([best_model_row]).to_csv(f"{pfx}_best_model_hyperparams.csv", index=False)
                print(f"    Saved: {pfx}_best_model_hyperparams.csv")
            except Exception as e:
                print(f"    Warning: best model hyperparams CSV failed: {e}")

            target_elapsed_sec = time.time() - target_start_time
            print(f"    Time taken ({target_label}): {fmt_elapsed(target_elapsed_sec)}")

            summary_rows.append({
                "Resample":        resample_tag,
                "Target":          target_label,
                "N_rows":          n_rows,
                "Train_rows":      len(train_data),
                "Test_rows":       len(test_data),
                "Best_Model":      best_name,
                "Best_RMSE":       rows[best_name]["RMSE"],
                "Best_MAE":        rows[best_name]["MAE"],
                "Best_MAPE_%":     rows[best_name]["MAPE_%"],
                "Best_R2":         rows[best_name]["R2"],
                "Best_Accuracy_%": rows[best_name]["Accuracy_%"],
                "Best_CV_R2_Mean": rows[best_name]["CV_R2_Mean"],
                "Best_CV_R2_Std":  rows[best_name]["CV_R2_Std"],
                "Elapsed_Sec":     round(target_elapsed_sec, 1),
            })

            # Tuning improvement: best tuned vs its baseline counterpart
            fam_map = {
                "Random Forest (Tuned)": "Random Forest (baseline)",
                "XGBoost (Tuned)":       "XGBoost (baseline)",
                "SVR (Tuned)":           "SVR (baseline)",
                "Ridge (Tuned)":         "Ridge (baseline)",
            }
            if best_name in fam_map:
                b_key = fam_map[best_name]
                if b_key in rows:
                    gain = rows[best_name]["Accuracy_%"] - rows[b_key]["Accuracy_%"]
                    improvement_rows.append({
                        "Resample":        resample_tag,
                        "Target":          target_label,
                        "Tuned_Model":     best_name,
                        "Baseline_Model":  b_key,
                        "Accuracy_Gain_%": gain,
                    })

        dataset_elapsed_sec = time.time() - dataset_start_time
        if tag_results:
            all_results[resample_tag] = tag_results
            print(f"\n  File complete: {resample_tag}  (time taken: {fmt_elapsed(dataset_elapsed_sec)})")
        else:
            failed_files.append((train_file, "all targets failed"))

    except Exception as e:
        print(f"\n  ERROR on {resample_tag}: {e}")
        traceback.print_exc()
        failed_files.append((train_file, str(e)))
        plt.close("all")
        continue


# =========================
# FINAL REPORT
# =========================

if summary_rows:
    combined_dir = os.path.join(BASE_OUTPUT_DIR, "Phase_1_30MinuteV2_Chronological_Master_Summary")
    os.makedirs(combined_dir, exist_ok=True)
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(combined_dir, "Phase_1_30MinuteV2_Chronological_Master_Summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Master summary saved: {summary_path}")
    print("\n" + summary_df.to_string(index=False))

total_elapsed_sec = time.time() - script_start_time
print(f"\n{'='*80}")
print(f"  COMPLETED : {len(all_results)} / {len(DATASETS)} datasets processed successfully")
print(f"  TOTAL TIME: {fmt_elapsed(total_elapsed_sec)}")
if model_time_totals:
    print(f"\n  TIME PER MODEL (Optuna search + final fit/predict/CV, "
          f"summed across every target/dataset):")
    for mname, secs in sorted(model_time_totals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {mname:30s} : {fmt_elapsed(secs)}")
    print(f"    {'SUM (approx. accounts for TOTAL TIME above)':30s} : "
          f"{fmt_elapsed(sum(model_time_totals.values()))}")
    if tuning_time_totals:
        print(f"\n  ...of which, Optuna hyperparameter search alone "
              f"(the rest is a comparatively instant final fit):")
        for fam, secs in sorted(tuning_time_totals.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {fam:30s} : {fmt_elapsed(secs)}")
if failed_files:
    print(f"  FAILED    : {len(failed_files)} file(s):")
    for fn, reason in failed_files:
        print(f"    x {fn}  ->  {reason}")
else:
    print("  All files completed with no failures.")
if not XGB_OK:
    print("  NOTE: pip install xgboost  to enable XGBoost")
print(f"{'='*80}")
