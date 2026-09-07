"""
Model Structure Visualization Add-on
=====================================
For the 30-Minute v2 chronological regression pipeline (RF / XGBoost / SVR / Ridge).

TREE-STYLE FIGURES (only possible for tree-based models):
  - Random Forest : one tree from the tuned forest, via sklearn's plot_tree.
                    Split rule, squared_error, sample count, value per node --
                    same layout as the reference RF figure.
  - XGBoost       : one boosting round's tree, via xgboost.to_graphviz.
                    Yes/No/missing branches -- same layout as the reference
                    XGBoost figure. Requires the `graphviz` python package
                    AND the Graphviz system binaries on PATH (pip install
                    graphviz; on Windows also install the Graphviz installer
                    from graphviz.org and add its bin/ folder to PATH).

NOT TREE-BASED -- Ridge (linear) and SVR (kernel) have no internal splits,
so there is no decision-tree figure that would honestly represent either.
What's shown instead is each model's own actual structure:
  - Ridge : learned coefficients + the full regression formula (converted
            back out of StandardScaler units into raw feature units), plus
            partial regression LINES -- straight, because Ridge is linear.
  - SVR   : the learned RBF kernel shape (using the model's own fitted
            gamma), partial dependence CURVES -- curved, because the RBF
            kernel makes SVR nonlinear -- and a residual-vs-epsilon
            histogram showing which test points actually fall inside the
            epsilon-insensitive tube.

USAGE
-----
Call `plot_all_model_structures(...)` once per target, right after your
existing per-target loop produces `fitted_models` from run_target(). It
runs every plot below for whichever tuned models are present in
fitted_models and writes them next to your other per-target outputs using
the same `pfx` prefix your main script already uses.

Individual functions can also be called on their own -- see the bottom of
this file for standalone call signatures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance


# =========================================================================
# RANDOM FOREST
# =========================================================================

def plot_rf_single_tree(fitted_models, feature_labels_rf, target_label, filepath,
                        tree_index=0, max_depth_display=3):
    """One tree from the tuned Random Forest, styled like the reference RF figure."""
    rf_pipe = next((m for n, m in fitted_models if "Random Forest (Tuned)" in n), None)
    if rf_pipe is None:
        print("  No tuned Random Forest model found in fitted_models.")
        return
    forest = rf_pipe.named_steps["rf"]
    single_tree = forest.estimators_[tree_index]

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        single_tree,
        feature_names=feature_labels_rf,
        filled=True,
        rounded=True,
        max_depth=max_depth_display,   # cap depth so the figure stays legible
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        f"Random Forest (Tuned) — Tree #{tree_index} (first {max_depth_display} levels)\n"
        f"{target_label}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# =========================================================================
# XGBOOST
# =========================================================================

def _ensure_graphviz_on_path():
    """
    Graphviz's `dot` binary must be on PATH for graph.render() to work.
    A fresh install (e.g. via winget) updates the registry's PATH, but a
    terminal/IDE process already running when that happened keeps its own
    stale, cached copy of PATH until it's fully restarted -- so `dot` can
    still be missing here even right after installing it. Fall back to the
    default Windows install location rather than requiring a restart.
    """
    import shutil
    if shutil.which("dot") is not None:
        return
    for candidate in (r"C:\Program Files\Graphviz\bin",
                      r"C:\Program Files (x86)\Graphviz\bin"):
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "dot.exe")):
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            return


def plot_xgb_single_tree(fitted_models, target_label, filepath, tree_index=0,
                         max_depth_display=3):
    """One boosting round's tree from the tuned XGBoost model, cropped to the
    first `max_depth_display` levels (styled like the reference XGBoost
    figure: Yes/No/missing branches). Needs graphviz.

    xgboost.to_graphviz() has no depth-limit option (unlike sklearn's
    plot_tree used for the RF figure), so a full boosted tree can run to
    dozens of levels and hundreds/thousands of nodes -- unreadable at any
    practical figure size. This crops it by walking the same node table
    xgboost itself builds internally (booster.trees_to_dataframe()),
    stopping at max_depth_display levels from the root, and rendering just
    that portion as a small graph in the same visual style. A split node
    sitting exactly at the depth cutoff is drawn gray with a "..." suffix
    (rather than the usual blue split color) to mark that the real tree
    continues below it, distinguishing it from an actual leaf.
    """
    import graphviz

    _ensure_graphviz_on_path()

    xgb_pipe = next((m for n, m in fitted_models if "XGBoost (Tuned)" in n), None)
    if xgb_pipe is None:
        print("  No tuned XGBoost model found in fitted_models.")
        return
    booster = xgb_pipe.named_steps["xgb"].get_booster()

    df = booster.trees_to_dataframe()
    df = df[df["Tree"] == tree_index].set_index("ID")
    if df.empty:
        print(f"  No tree at index {tree_index} found in the XGBoost booster.")
        return
    total_nodes = len(df)

    root_id = f"{tree_index}-0"
    depth = {root_id: 0}
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        row = df.loc[nid]
        if row["Feature"] == "Leaf" or depth[nid] >= max_depth_display:
            continue
        for child_id in (row["Yes"], row["No"]):
            if child_id not in depth:
                depth[child_id] = depth[nid] + 1
                queue.append(child_id)

    graph = graphviz.Digraph()
    for nid in depth:
        row = df.loc[nid]
        is_leaf = row["Feature"] == "Leaf"
        is_cropped_split = (not is_leaf) and depth[nid] == max_depth_display
        if is_leaf:
            label = f"leaf={row['Gain']:.6g}"
            graph.node(nid, label, shape="box", style="filled, rounded", fillcolor="#e0e0e0")
        elif is_cropped_split:
            label = f"{row['Feature']}<{row['Split']:.6g}\n..."
            graph.node(nid, label, shape="box", style="filled, rounded", fillcolor="#e0e0e0")
        else:
            label = f"{row['Feature']}<{row['Split']:.6g}"
            graph.node(nid, label, shape="box", style="filled, rounded", fillcolor="#bde0fe")
            if row["Yes"] in depth:
                graph.edge(nid, row["Yes"], label="yes", color="red")
            if row["No"] in depth:
                no_label = "no, missing" if row["No"] == row["Missing"] else "no"
                graph.edge(nid, row["No"], label=no_label, color="blue")

    graph.attr(
        label=f"XGBoost (Tuned) — Tree #{tree_index} "
              f"(first {max_depth_display} levels of {total_nodes}-node tree)\n{target_label}",
        labelloc="t", fontsize="16",
    )

    out_path_no_ext = filepath.rsplit(".", 1)[0]
    graph.render(out_path_no_ext, format="png", cleanup=True)
    print(f"  Saved: {out_path_no_ext}.png")


# =========================================================================
# RIDGE
# =========================================================================

def plot_ridge_coefficients(fitted_models, feature_labels_ridge, target_label, filepath):
    """Ridge has no tree structure. Its actual learned structure is its
    coefficients -- shown here sorted by magnitude (standardized units)."""
    ridge_pipe = next((m for n, m in fitted_models if "Ridge (Tuned)" in n), None)
    if ridge_pipe is None:
        print("  No tuned Ridge model found in fitted_models.")
        return
    coefs = ridge_pipe.named_steps["ridge"].coef_

    order = sorted(range(len(coefs)), key=lambda i: abs(coefs[i]), reverse=True)
    labels_sorted = [feature_labels_ridge[i] for i in order]
    coefs_sorted  = [coefs[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1565C0" if c >= 0 else "#C62828" for c in coefs_sorted]
    ax.barh(labels_sorted, coefs_sorted, color=colors, edgecolor="black")
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title(f"Ridge (Tuned) — Coefficients\n{target_label}\n"
                "(no decision-tree structure exists for a linear model)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def get_ridge_formula(fitted_models, feature_labels_ridge, target_label, decimals=6):
    """
    Ridge fits ONE hyperplane across all `len(feature_labels_ridge)` features
    -- y = intercept + coef_1*x_1 + ... + coef_10*x_10 -- not a single 2D
    line. This extracts that formula with coefficients converted OUT of the
    pipeline's internal StandardScaler units and back into your original
    feature units, since ridge_pipe.named_steps["ridge"].coef_ alone is in
    standardized (mean 0, std 1) units and isn't usable directly on raw
    feature values.

    Conversion (per feature i, with StandardScaler mean_i and scale_i):
        raw_coef_i    = std_coef_i / scale_i
        raw_intercept = std_intercept - sum_i(std_coef_i * mean_i / scale_i)

    Returns (raw_coefs: dict{feature -> coef}, raw_intercept: float,
    formula_str: str). Also prints the formula.
    """
    ridge_pipe = next((m for n, m in fitted_models if "Ridge (Tuned)" in n), None)
    if ridge_pipe is None:
        print("  No tuned Ridge model found in fitted_models.")
        return None, None, None

    scaler        = ridge_pipe.named_steps["scaler"]
    ridge_model   = ridge_pipe.named_steps["ridge"]
    std_coefs     = ridge_model.coef_
    std_intercept = ridge_model.intercept_
    means, scales = scaler.mean_, scaler.scale_

    raw_coefs     = std_coefs / scales
    raw_intercept = std_intercept - np.sum(std_coefs * means / scales)

    terms = " + ".join(
        f"({c:.{decimals}f} * {name})" for name, c in zip(feature_labels_ridge, raw_coefs)
    )
    formula_str = f"{target_label} = {raw_intercept:.{decimals}f} + {terms}"

    print(f"\n  Ridge (Tuned) regression formula -- {target_label} (raw feature units):")
    print(f"  {formula_str}\n")
    for name, c in zip(feature_labels_ridge, raw_coefs):
        print(f"    {name:30s} : {c:.{decimals}f}")
    print(f"    {'intercept':30s} : {raw_intercept:.{decimals}f}")

    return dict(zip(feature_labels_ridge, raw_coefs)), raw_intercept, formula_str


def plot_ridge_partial_regression_lines(fitted_models, X_train_ridge, y_train,
                                        feature_labels_ridge, target_label,
                                        filepath, top_n=4):
    """
    Draws an actual regression LINE per feature: holds every other feature at
    its training-set mean and sweeps the chosen feature across its observed
    range, plotting the resulting straight-line prediction against the real
    training scatter for that feature. This is a true 2D slice through the
    fitted hyperplane -- not an approximation -- because Ridge is linear, so
    holding the other 9 features fixed makes the model's response to the
    remaining feature exactly linear.

    top_n: how many of the largest-|coefficient| features to plot (each gets
    its own panel).
    """
    ridge_pipe = next((m for n, m in fitted_models if "Ridge (Tuned)" in n), None)
    if ridge_pipe is None:
        print("  No tuned Ridge model found in fitted_models.")
        return

    scaler      = ridge_pipe.named_steps["scaler"]
    ridge_model = ridge_pipe.named_steps["ridge"]
    imputer     = ridge_pipe.named_steps["imp"]

    means_orig = X_train_ridge.mean(axis=0).values
    order = np.argsort(-np.abs(ridge_model.coef_ / scaler.scale_))[:top_n]

    ncols = 2
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax_i, feat_i in enumerate(order):
        feat_name = feature_labels_ridge[feat_i]
        ax = axes[ax_i]

        x_vals = X_train_ridge.iloc[:, feat_i].dropna()
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

        synth = np.tile(means_orig, (100, 1))
        synth[:, feat_i] = x_line
        synth_scaled = scaler.transform(imputer.transform(synth))
        y_line = ridge_model.predict(synth_scaled)

        ax.scatter(x_vals, y_train.loc[x_vals.index], s=14, alpha=0.35,
                  color="#90CAF9", label="Training data")
        ax.plot(x_line, y_line, color="#1565C0", linewidth=2.2,
               label="Ridge partial regression line")
        ax.set_xlabel(feat_name)
        ax.set_ylabel(target_label)
        ax.set_title(f"{feat_name}\n(other 9 features held at their mean)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(top_n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Ridge (Tuned) — Partial Regression Lines, Top {top_n} Features\n{target_label}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# =========================================================================
# SVR
# =========================================================================

def plot_svr_support_vectors(fitted_models, target_label, filepath):
    """SVR has no tree structure. Its actual learned structure is which
    training points became support vectors -- shown here as a count."""
    svr_pipe = next((m for n, m in fitted_models if "SVR (Tuned)" in n), None)
    if svr_pipe is None:
        print("  No tuned SVR model found in fitted_models.")
        return
    svr_model = svr_pipe.named_steps["svr"]
    n_sv = len(svr_model.support_)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["Support Vectors"], [n_sv], color="#1565C0", edgecolor="black")
    ax.set_title(f"SVR (Tuned) — Support Vector Count: {n_sv}\n{target_label}\n"
                "(no decision-tree structure exists for a kernel model)")
    ax.set_ylabel("Number of Points")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def get_svr_landmark_table(fitted_models, feature_labels_svr, target_label,
                           filepath=None, top_n=5):
    """
    Table of every support vector ("landmark") the tuned SVR actually
    learned to anchor its predictions to: its feature values and the
    dual-coefficient weight sklearn assigned it. support_vectors_ is
    stored by sklearn in the pipeline's standardized (StandardScaler)
    space, not raw units, so it's converted back out
    (raw = scaled * scale_ + mean_) the same way get_ridge_formula/
    get_svr_formula convert their coefficients -- otherwise the feature
    values wouldn't be interpretable against the real data.

    Prints the total support-vector count and the top_n landmarks ranked
    by |SV_Weight| (the ones with the most influence on predictions), and
    -- if filepath is given -- saves the FULL table (all support vectors,
    sorted the same way) as a CSV, since printing every one isn't
    practical when there are dozens/hundreds of them.

    Returns the full sv_df (pandas.DataFrame, columns: SV_Train_Index,
    one column per feature in feature_labels_svr, SV_Weight), sorted by
    |SV_Weight| descending, or None if no tuned SVR model is present.
    """
    svr_pipe = next((m for n, m in fitted_models if "SVR (Tuned)" in n), None)
    if svr_pipe is None:
        print("  No tuned SVR model found in fitted_models.")
        return None
    svr_model = svr_pipe.named_steps["svr"]
    scaler    = svr_pipe.named_steps["scaler"]

    sv_raw = svr_model.support_vectors_ * scaler.scale_ + scaler.mean_
    sv_df = pd.DataFrame(sv_raw, columns=feature_labels_svr)
    sv_df.insert(0, "SV_Train_Index", svr_model.support_)
    sv_df["SV_Weight"] = svr_model.dual_coef_[0]
    sv_df = sv_df.sort_values(by="SV_Weight", key=np.abs, ascending=False).reset_index(drop=True)

    print(f"\n  SVR (Tuned) — Support Vector ('Landmark') Table -- {target_label}:")
    print(f"    Number of Support Vectors: {len(sv_df)}")
    print(f"    Top {min(top_n, len(sv_df))} most influential landmarks (by |SV_Weight|):")
    print(sv_df.head(top_n).to_string(index=False))
    print()

    if filepath:
        sv_df.to_csv(filepath, index=False)
        print(f"  Saved: {filepath}")

    return sv_df


def get_svr_formula(fitted_models, feature_labels_svr, target_label, decimals=6):
    """
    SVR's actual prediction function -- unlike Ridge, this isn't one fixed
    hyperplane; it's a sum of RBF-kernel similarities to every support
    vector, weighted by that vector's fitted dual coefficient, plus a bias
    term:

        y_pred(x) = intercept + sum_i dual_coef_i * K(x, sv_i)
        K(x, sv_i) = exp(-gamma * ||x_scaled - sv_i||^2)

    x_scaled is x standardized the same way the pipeline's StandardScaler
    standardizes training data (x_scaled_j = (x_j - mean_j) / scale_j),
    sv_i are the support vectors (already stored by sklearn in that scaled
    space), gamma is the model's own fitted RBF gamma, and dual_coef_i =
    alpha_i - alpha_i* is sklearn's already-combined dual coefficient per
    support vector.

    Prints the formula's structure and parameters (gamma, intercept,
    support-vector count, per-feature scaler mean/scale). With dozens or
    hundreds of support vectors, the full per-vector coefficient/coordinate
    table isn't practical to print in full, so only the first few terms are
    shown -- the complete arrays are returned for inspection or export.

    Returns (dual_coefs: np.ndarray shape (n_sv,), support_vectors_scaled:
    np.ndarray shape (n_sv, n_features), intercept: float, gamma: float,
    formula_str: str).
    """
    svr_pipe = next((m for n, m in fitted_models if "SVR (Tuned)" in n), None)
    if svr_pipe is None:
        print("  No tuned SVR model found in fitted_models.")
        return None, None, None, None, None
    svr_model = svr_pipe.named_steps["svr"]
    if svr_model.kernel != "rbf":
        print(f"  Tuned SVR uses kernel='{svr_model.kernel}', not 'rbf' -- "
              f"this formula is specific to the RBF kernel.")
        return None, None, None, None, None

    scaler = svr_pipe.named_steps["scaler"]

    dual_coefs   = svr_model.dual_coef_.ravel()
    support_vecs = svr_model.support_vectors_
    intercept    = float(svr_model.intercept_[0])
    gamma        = svr_model._gamma
    n_sv         = len(dual_coefs)
    n_preview    = min(3, n_sv)

    preview_terms = " + ".join(
        f"({c:.{decimals}f} * exp(-{gamma:.{decimals}f} * ||x_scaled - sv_{i}||^2))"
        for i, c in enumerate(dual_coefs[:n_preview])
    )
    ellipsis = f" + ... [{n_sv - n_preview} more support-vector terms]" if n_sv > n_preview else ""
    formula_str = f"{target_label} = {intercept:.{decimals}f} + {preview_terms}{ellipsis}"

    print(f"\n  SVR (Tuned) prediction function -- {target_label} (RBF kernel):")
    print("    y_pred(x) = intercept + sum_i dual_coef_i * exp(-gamma * ||x_scaled - sv_i||^2)")
    print(f"    intercept       : {intercept:.{decimals}f}")
    print(f"    gamma           : {gamma:.{decimals}f}")
    print(f"    support vectors : {n_sv}")
    print("    x_scaled_j = (x_j - mean_j) / scale_j, per feature (pipeline's StandardScaler):")
    for name, mean, scale in zip(feature_labels_svr, scaler.mean_, scaler.scale_):
        print(f"      {name:30s} : mean={mean:.{decimals}f}, scale={scale:.{decimals}f}")
    print(f"\n    First {n_preview} of {n_sv} support-vector terms:")
    print(f"    {formula_str}\n")

    return dual_coefs, support_vecs, intercept, gamma, formula_str


def plot_svr_kernel_shape(fitted_models, target_label, filepath, max_distance=None):
    """
    THE kernel plot: draws the actual RBF kernel function this model learned
    to use, K(d) = exp(-gamma * d^2), using the model's own fitted gamma
    (SVR resolves gamma='scale'/'auto' to a concrete float at fit time,
    stored internally as svr_model._gamma). Every SVR prediction is a
    gamma-weighted sum of kernel similarities to the support vectors, so
    this curve IS how much influence a point loses as it gets farther from
    a support vector -- the real mechanism, not a proxy for it.
    """
    svr_pipe = next((m for n, m in fitted_models if "SVR (Tuned)" in n), None)
    if svr_pipe is None:
        print("  No tuned SVR model found in fitted_models.")
        return
    svr_model = svr_pipe.named_steps["svr"]
    if svr_model.kernel != "rbf":
        print(f"  Tuned SVR uses kernel='{svr_model.kernel}', not 'rbf' -- "
              f"this plot is specific to the RBF kernel shape.")
        return
    gamma = svr_model._gamma

    if max_distance is None:
        # Half-max distance (K drops to 0.5) is sqrt(ln2/gamma) -- show a
        # bit past 3x that so the full decay is visible.
        max_distance = 3.0 * np.sqrt(np.log(2) / gamma)
    d = np.linspace(0, max_distance, 300)
    k = np.exp(-gamma * d**2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d, k, color="#1565C0", linewidth=2.2)
    ax.fill_between(d, k, alpha=0.12, color="#1565C0")
    half_life = np.sqrt(np.log(2) / gamma)
    ax.axvline(half_life, color="#D85A30", linestyle="--", linewidth=1.2)
    ax.text(half_life, 0.55, f"  K=0.5 at distance={half_life:.4f}",
           color="#D85A30", fontsize=9)
    ax.set_xlabel("Distance between two points (standardized feature space)")
    ax.set_ylabel("Kernel similarity, K(d) = exp(-gamma * d^2)")
    ax.set_title(f"SVR (Tuned) — Learned RBF Kernel Shape\n"
                f"{target_label} | gamma={gamma:.6f}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_svr_partial_dependence_lines(fitted_models, X_train_svr, y_train,
                                      feature_labels_svr, target_label,
                                      filepath, top_n=4):
    """
    SVR equivalent of the Ridge partial regression lines -- but CURVED,
    since the RBF kernel makes SVR nonlinear. Holds every other feature at
    its training mean, sweeps one feature across its observed range, and
    runs the swept rows through the actual fitted pipeline (imputer +
    scaler + SVR) to get the real predicted response shape. Feature ranking
    for top_n uses permutation importance since SVR has no coefficients.
    """
    svr_pipe = next((m for n, m in fitted_models if "SVR (Tuned)" in n), None)
    if svr_pipe is None:
        print("  No tuned SVR model found in fitted_models.")
        return

    means_orig = X_train_svr.mean(axis=0).values

    perm = permutation_importance(svr_pipe, X_train_svr, y_train,
                                  n_repeats=5, scoring="r2", random_state=42, n_jobs=-1)
    order = np.argsort(-perm.importances_mean)[:top_n]

    ncols = 2
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax_i, feat_i in enumerate(order):
        feat_name = feature_labels_svr[feat_i]
        ax = axes[ax_i]

        x_vals = X_train_svr.iloc[:, feat_i].dropna()
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

        synth = np.tile(means_orig, (100, 1))
        synth[:, feat_i] = x_line
        synth_df = pd.DataFrame(synth, columns=feature_labels_svr)
        y_line = svr_pipe.predict(synth_df)

        ax.scatter(x_vals, y_train.loc[x_vals.index], s=14, alpha=0.35,
                  color="#90CAF9", label="Training data")
        ax.plot(x_line, y_line, color="#1565C0", linewidth=2.2,
               label="SVR partial dependence")
        ax.set_xlabel(feat_name)
        ax.set_ylabel(target_label)
        ax.set_title(f"{feat_name}\n(other 9 features held at their mean)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(top_n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"SVR (Tuned) — Partial Dependence Curves, Top {top_n} Features\n{target_label}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_svr_residual_distribution(fitted_models, X_test_svr, y_test, target_label, filepath):
    """
    Histogram of |y_true - y_pred| on the TEST set, with the model's tuned
    epsilon marked -- shows directly, from real residuals, which points
    fall inside the epsilon-insensitive tube (no penalty during training)
    vs outside it (support vectors). More concrete than a bare
    support-vector count since it's tied to actual error magnitudes.
    """
    svr_pipe = next((m for n, m in fitted_models if "SVR (Tuned)" in n), None)
    if svr_pipe is None:
        print("  No tuned SVR model found in fitted_models.")
        return
    epsilon = svr_pipe.named_steps["svr"].epsilon
    y_pred = svr_pipe.predict(X_test_svr)
    abs_err = np.abs(np.array(y_test) - y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(abs_err, bins=30, color="#90CAF9", edgecolor="black", alpha=0.85)
    ax.axvline(epsilon, color="#D85A30", linestyle="--", linewidth=1.5,
              label=f"epsilon = {epsilon:.6f}")
    pct_inside = (abs_err <= epsilon).mean() * 100
    ax.set_xlabel("Absolute error, |y_true - y_pred| (test set)")
    ax.set_ylabel("Count")
    ax.set_title(f"SVR (Tuned) — Test Residuals vs. Epsilon Tube\n"
                f"{target_label} | {pct_inside:.1f}% of test points fall inside the tube")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# =========================================================================
# ORCHESTRATOR -- run everything for one target in one call
# =========================================================================

def plot_all_model_structures(fitted_models, family_X_train, family_X_test,
                              y_train, y_test, target_label, pfx,
                              feature_labels_by_family):
    """
    Runs every structure plot above for one target, using the exact
    variables your main script's per-target loop already has in scope.
    Call this immediately after:
        results, preds, rows, fitted_models, best_params_log = run_target(...)

    Parameters mirror what's already available at that point in your loop:
        family_X_train, family_X_test : dicts keyed 'rf'/'xgb'/'svr'/'ridge'
        y_train, y_test               : shared target series for this target
        target_label                  : e.g. "X Frequency (Hz)"
        pfx                           : your existing output path prefix
        feature_labels_by_family      : FEATURE_LABELS_BY_FAMILY from main script
    """
    # Each plot runs independently -- one model's failure (e.g. XGBoost's
    # tree render needing Graphviz binaries that may not be installed)
    # must not prevent the other models' plots from being generated.
    def _safe(label, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"  Warning: {label} failed: {e}")

    # Random Forest
    _safe("RF tree plot", plot_rf_single_tree, fitted_models,
          feature_labels_by_family["rf"], target_label, f"{pfx}_rf_tree_example.png")

    # XGBoost
    _safe("XGBoost tree plot", plot_xgb_single_tree, fitted_models,
          target_label, f"{pfx}_xgb_tree_example.png")

    # Ridge
    _safe("Ridge coefficients plot", plot_ridge_coefficients, fitted_models,
          feature_labels_by_family["ridge"], target_label, f"{pfx}_ridge_coefficients.png")
    _safe("Ridge formula", get_ridge_formula, fitted_models,
          feature_labels_by_family["ridge"], target_label)
    _safe("Ridge partial regression lines", plot_ridge_partial_regression_lines,
          fitted_models, family_X_train["ridge"], y_train,
          feature_labels_by_family["ridge"], target_label,
          f"{pfx}_ridge_partial_lines.png")

    # SVR
    _safe("SVR support vector plot", plot_svr_support_vectors, fitted_models,
          target_label, f"{pfx}_svr_support_vectors.png")
    _safe("SVR landmark table", get_svr_landmark_table, fitted_models,
          feature_labels_by_family["svr"], target_label,
          f"{pfx}_svr_landmark_table.csv")
    _safe("SVR prediction function", get_svr_formula, fitted_models,
          feature_labels_by_family["svr"], target_label)
    _safe("SVR kernel shape plot", plot_svr_kernel_shape, fitted_models,
          target_label, f"{pfx}_svr_kernel_shape.png")
    _safe("SVR partial dependence plot", plot_svr_partial_dependence_lines,
          fitted_models, family_X_train["svr"], y_train,
          feature_labels_by_family["svr"], target_label,
          f"{pfx}_svr_partial_dependence.png")
    _safe("SVR residual distribution plot", plot_svr_residual_distribution,
          fitted_models, family_X_test["svr"], y_test, target_label,
          f"{pfx}_svr_residual_distribution.png")


# =========================================================================
# Standalone usage (if you'd rather call things individually instead of
# through plot_all_model_structures)
# =========================================================================
#
# plot_rf_single_tree(fitted_models, FEATURE_LABELS_BY_FAMILY["rf"],
#                     target_label, f"{pfx}_rf_tree_example.png")
# plot_xgb_single_tree(fitted_models, target_label, f"{pfx}_xgb_tree_example.png")
#
# plot_ridge_coefficients(fitted_models, FEATURE_LABELS_BY_FAMILY["ridge"],
#                         target_label, f"{pfx}_ridge_coefficients.png")
# get_ridge_formula(fitted_models, FEATURE_LABELS_BY_FAMILY["ridge"], target_label)
# plot_ridge_partial_regression_lines(fitted_models, family_X_train["ridge"], y_train,
#                                     FEATURE_LABELS_BY_FAMILY["ridge"], target_label,
#                                     f"{pfx}_ridge_partial_lines.png")
#
# plot_svr_support_vectors(fitted_models, target_label, f"{pfx}_svr_support_vectors.png")
# get_svr_landmark_table(fitted_models, FEATURE_LABELS_BY_FAMILY["svr"], target_label,
#                        f"{pfx}_svr_landmark_table.csv")
# get_svr_formula(fitted_models, FEATURE_LABELS_BY_FAMILY["svr"], target_label)
# plot_svr_kernel_shape(fitted_models, target_label, f"{pfx}_svr_kernel_shape.png")
# plot_svr_partial_dependence_lines(fitted_models, family_X_train["svr"], y_train,
#                                   FEATURE_LABELS_BY_FAMILY["svr"], target_label,
#                                   f"{pfx}_svr_partial_dependence.png")
# plot_svr_residual_distribution(fitted_models, family_X_test["svr"], y_test, target_label,
#                                f"{pfx}_svr_residual_distribution.png")
