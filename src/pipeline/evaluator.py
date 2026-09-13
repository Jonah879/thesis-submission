"""
evaluator.py — Computes evaluation metrics on the predictions CSV.

Public API:
    evaluate(results_csv)  -> dict   (also prints a formatted report)
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from . import config

logger = logging.getLogger(__name__)

LABEL_ORDER = ["Yea", "Nay"]


def _fmt_confusion_matrix(cm, labels: list) -> str:
    """Return a readable text confusion matrix."""
    col_w = max(len(l) for l in labels) + 2
    header = " " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels)
    lines  = [header]
    for i, true_label in enumerate(labels):
        row = f"{true_label:<{col_w}}  " + "  ".join(f"{cm[i][j]:>{col_w}}" for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


def _restrict_to_reference(
    df: pd.DataFrame,
    restrict_to: Union[str, Path, None],
) -> tuple:
    """
    If restrict_to is set, inner-join df against the (vote_id, bioguide)
    pairs that have a non-ERROR prediction in the reference results CSV.

    Returns (filtered_df, n_before). n_before is None if no restriction
    was applied (so callers can distinguish "not restricted" from
    "restricted, 0 rows dropped").
    """
    if restrict_to is None:
        return df, None

    ref_path = Path(restrict_to)
    if not ref_path.exists():
        raise FileNotFoundError(f"Restriction reference file not found: {ref_path}")

    ref_df = pd.read_csv(ref_path, usecols=["vote_id", "bioguide", "predicted_label"])
    valid_pairs = (
        ref_df[ref_df["predicted_label"] != "ERROR"][["vote_id", "bioguide"]]
        .drop_duplicates()
    )

    n_before = len(df)
    df = df.merge(valid_pairs, on=["vote_id", "bioguide"], how="inner")
    return df, n_before


def evaluate(
    results_csv: Union[str, Path] = config.PREDICTIONS_CSV,
    min_samples: int = 10,
    restrict_to: Union[str, Path, None] = None,
) -> dict:
    """
    Load predictions CSV, compute metrics, print a report, and return a metrics dict.

    Rows where predicted_label == "ERROR" are excluded from metric computation
    but their count is reported.

    If restrict_to is given, evaluation is limited to (vote_id, bioguide)
    pairs that have a non-ERROR prediction in that reference results CSV —
    for a fair comparison against a baseline that couldn't score every row
    (e.g. results/baseline_dw_nominate.csv).

    Returns a dict with keys:
        n_total, n_errors, n_evaluated,
        accuracy, macro_f1, weighted_f1,
        per_class_f1, confusion_matrix (as list-of-lists),
        classification_report (str)
    """
    path = Path(results_csv)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    df, n_before_restrict = _restrict_to_reference(df, restrict_to)
    n_total  = len(df)
    errors   = df[df["predicted_label"] == "ERROR"]
    n_errors = len(errors)

    df_eval = df[df["predicted_label"] != "ERROR"].copy()
    n_eval  = len(df_eval)

    print("=" * 60)
    print("VOTE PREDICTION — EVALUATION REPORT")
    print("=" * 60)
    if n_before_restrict is not None:
        print(f"Restricted to     : {n_total}/{n_before_restrict} rows (via {Path(restrict_to).name})")
    print(f"Total predictions : {n_total}")
    print(f"Errors (skipped)  : {n_errors}")
    print(f"Evaluated rows    : {n_eval}")

    if n_eval < min_samples:
        msg = f"Not enough valid predictions to evaluate (need >= {min_samples}, got {n_eval})."
        print(msg)
        return {
            "n_total": n_total, "n_errors": n_errors, "n_evaluated": n_eval,
            "accuracy": None, "macro_f1": None, "weighted_f1": None,
            "per_class_f1": {}, "confusion_matrix": [], "classification_report": msg,
        }

    y_true = df_eval["true_label"].tolist()
    y_pred = df_eval["predicted_label"].tolist()

    # Keep only labels that actually appear
    present_labels = [l for l in LABEL_ORDER if l in set(y_true) | set(y_pred)]

    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    labels=present_labels, zero_division=0)
    weighted_f1= f1_score(y_true, y_pred, average="weighted", labels=present_labels, zero_division=0)
    cm         = confusion_matrix(y_true, y_pred, labels=present_labels)
    report     = classification_report(y_true, y_pred, labels=present_labels, zero_division=0)

    per_class_f1 = {
        label: round(f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0), 4)
        for label in present_labels
    }

    print()
    print(f"Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print()
    print("Per-class F1:")
    for label, score in per_class_f1.items():
        print(f"  {label:<12}: {score:.4f}")
    print()
    print("Classification Report:")
    print(report)
    print("Confusion Matrix  (rows = true, cols = predicted):")
    print(_fmt_confusion_matrix(cm.tolist(), present_labels))
    print()

    # -----------------------------------------------------------------------
    # Optional breakdown by party and chamber
    # -----------------------------------------------------------------------
    for groupby_col in ["party", "chamber"]:
        if groupby_col in df_eval.columns:
            print(f"Accuracy by {groupby_col}:")
            for val, grp in df_eval.groupby(groupby_col):
                grp_acc = accuracy_score(grp["true_label"], grp["predicted_label"])
                print(f"  {str(val):<12}: {grp_acc:.4f}  (n={len(grp)})")
            print()

    return {
        "n_total":               n_total,
        "n_errors":              n_errors,
        "n_evaluated":           n_eval,
        "accuracy":              round(acc, 4),
        "macro_f1":              round(macro_f1, 4),
        "weighted_f1":           round(weighted_f1, 4),
        "per_class_f1":          per_class_f1,
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_evaluate(
    results_csv: Union[str, Path] = config.PREDICTIONS_CSV,
    n_bootstrap: int = config.N_BOOTSTRAP,
    ci: float = config.BOOTSTRAP_CI,
    restrict_to: Union[str, Path, None] = None,
) -> dict:
    """
    Bootstrap confidence intervals for accuracy, macro F1, and weighted F1.

    Resamples the prediction set with replacement B times and computes
    metrics on each bootstrap sample. Reports the mean and percentile CI.

    Also computes per-group breakdowns (chamber, party, competitiveness)
    if those columns are present.

    If restrict_to is given, evaluation is limited to (vote_id, bioguide)
    pairs that have a non-ERROR prediction in that reference results CSV.

    Returns a dict with overall and per-group bootstrap results.
    """
    path = Path(results_csv)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    df, n_before_restrict = _restrict_to_reference(df, restrict_to)
    df_eval = df[df["predicted_label"] != "ERROR"].reset_index(drop=True)
    n = len(df_eval)

    print("=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    if n_before_restrict is not None:
        print(f"Restricted        : {len(df)}/{n_before_restrict} rows (via {Path(restrict_to).name})")
    print(f"(B={n_bootstrap}, CI={ci*100:.0f}%, N={n} predictions)")
    print("=" * 60)

    if n < 30:
        print(f"Not enough predictions for bootstrap (need >= 30, got {n}).")
        return {}

    alpha = (1 - ci) / 2
    rng = np.random.default_rng(42)

    # ---- Overall bootstrap ------------------------------------------------
    y_true = df_eval["true_label"].values
    y_pred = df_eval["predicted_label"].values

    present_labels = [l for l in LABEL_ORDER if l in set(y_true) | set(y_pred)]

    boot_acc = np.empty(n_bootstrap)
    boot_macro_f1 = np.empty(n_bootstrap)
    boot_weighted_f1 = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        boot_acc[b] = accuracy_score(yt, yp)
        boot_macro_f1[b] = f1_score(yt, yp, average="macro", labels=present_labels, zero_division=0)
        boot_weighted_f1[b] = f1_score(yt, yp, average="weighted", labels=present_labels, zero_division=0)

    print()
    print(f"Accuracy          {boot_acc.mean():.4f}  [{np.percentile(boot_acc, alpha*100):.4f}, {np.percentile(boot_acc, (1-alpha)*100):.4f}]")
    print(f"Macro F1          {boot_macro_f1.mean():.4f}  [{np.percentile(boot_macro_f1, alpha*100):.4f}, {np.percentile(boot_macro_f1, (1-alpha)*100):.4f}]")
    print(f"Weighted F1       {boot_weighted_f1.mean():.4f}  [{np.percentile(boot_weighted_f1, alpha*100):.4f}, {np.percentile(boot_weighted_f1, (1-alpha)*100):.4f}]")

    # ---- Per-group breakdowns ---------------------------------------------
    for groupby_col in ["chamber", "party", "competitiveness"]:
        if groupby_col not in df_eval.columns:
            continue

        print(f"\nBy {groupby_col}:")
        for val, grp in df_eval.groupby(groupby_col):
            n_grp = len(grp)
            if n_grp < 10:
                print(f"  {str(val):<12}: n={n_grp} (too few to bootstrap)")
                continue

            yt = grp["true_label"].values
            yp = grp["predicted_label"].values
            grp_labels = [l for l in present_labels if l in set(yt) | set(yp)]
            boot_grp_acc = np.empty(n_bootstrap)

            for b in range(n_bootstrap):
                idx = rng.integers(0, n_grp, size=n_grp)
                boot_grp_acc[b] = accuracy_score(yt[idx], yp[idx])

            ci_lo = np.percentile(boot_grp_acc, alpha * 100)
            ci_hi = np.percentile(boot_grp_acc, (1 - alpha) * 100)
            print(f"  {str(val):<12}: acc={boot_grp_acc.mean():.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  (n={n_grp})")

    print()
    return {
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
        "n_predictions": n,
        "accuracy": {
            "mean": float(boot_acc.mean()),
            "ci_low": float(np.percentile(boot_acc, alpha * 100)),
            "ci_high": float(np.percentile(boot_acc, (1 - alpha) * 100)),
        },
        "macro_f1": {
            "mean": float(boot_macro_f1.mean()),
            "ci_low": float(np.percentile(boot_macro_f1, alpha * 100)),
            "ci_high": float(np.percentile(boot_macro_f1, (1 - alpha) * 100)),
        },
        "weighted_f1": {
            "mean": float(boot_weighted_f1.mean()),
            "ci_low": float(np.percentile(boot_weighted_f1, alpha * 100)),
            "ci_high": float(np.percentile(boot_weighted_f1, (1 - alpha) * 100)),
        },
    }


# ---------------------------------------------------------------------------
# Per-stratum / groupby evaluation
# ---------------------------------------------------------------------------

def groupby_accuracy(
    results_csv: Union[str, Path] = config.PREDICTIONS_CSV,
    groupby_cols: list = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    min_samples: int = 10,
) -> dict:
    """
    Compute accuracy with bootstrap CI for each group defined by groupby_cols.

    If groupby_cols includes "congress_era", it is derived from the "congress"
    column using the same era mapping as the stratified sampler.

    Args:
        results_csv: Path to predictions CSV.
        groupby_cols: List of column names to group by (e.g. ["congress_era", "chamber"]).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence interval level (default 0.95 = 95%).
        min_samples: Skip groups with fewer than this many predictions.

    Returns:
        Dict mapping group label -> {accuracy, ci_low, ci_high, n}.
    """
    from pipeline.pipeline import VotePredictionPipeline

    if groupby_cols is None:
        groupby_cols = ["chamber", "party"]

    path = Path(results_csv)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    df_eval = df[df["predicted_label"] != "ERROR"].copy()

    # Derive congress_era if needed
    if "congress_era" in groupby_cols and "congress_era" not in df_eval.columns:
        if "congress" in df_eval.columns:
            df_eval["congress_era"] = df_eval["congress"].apply(
                VotePredictionPipeline._congress_to_era
            )
        else:
            logger.warning("No 'congress' column — cannot derive congress_era.")
            groupby_cols = [c for c in groupby_cols if c != "congress_era"]

    alpha = (1 - ci) / 2
    rng = np.random.default_rng(42)

    results = {}
    for name, grp in df_eval.groupby(groupby_cols):
        n_grp = len(grp)
        if n_grp < min_samples:
            continue

        yt = grp["true_label"].values
        yp = grp["predicted_label"].values

        boot_acc = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_grp, size=n_grp)
            boot_acc[b] = accuracy_score(yt[idx], yp[idx])

        results[name] = {
            "accuracy": round(float(boot_acc.mean()), 4),
            "ci_low": round(float(np.percentile(boot_acc, alpha * 100)), 4),
            "ci_high": round(float(np.percentile(boot_acc, (1 - alpha) * 100)), 4),
            "n": n_grp,
        }

    # Pretty print
    print(f"\nAccuracy by {' × '.join(groupby_cols)}:")
    print(f"  (bootstrap B={n_bootstrap}, CI={ci*100:.0f}%, min_samples={min_samples})")
    print("-" * 60)
    for name, metrics in sorted(results.items()):
        label = str(name) if isinstance(name, tuple) else name
        print(
            f"  {label:<40} acc={metrics['accuracy']:.4f}  "
            f"[{metrics['ci_low']:.4f}, {metrics['ci_high']:.4f}]  "
            f"(n={metrics['n']})"
        )

    return results


# ---------------------------------------------------------------------------
# ROC / AUC threshold diagnostics
# ---------------------------------------------------------------------------

def roc_auc_analysis(
    results_csv: Union[str, Path],
    score_col: str = "p_yea_hat",
    label_col: str = "true_label",
    positive_label: str = "Yea",
    groupby_col: str = "chamber",
    plot_path: Union[str, Path, None] = None,
) -> dict:
    """
    Compute ROC AUC for a continuous score column against the binary label,
    and pick the threshold that maximizes Youden's J (tpr - fpr).

    Rows where predicted_label == "ERROR" or score_col is NaN are excluded.

    Returns a dict with keys:
        n_total, n_excluded, n_scored, auc, best_threshold,
        fpr, tpr, thresholds, by_group (dict of group -> auc)
    """
    path = Path(results_csv)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)
    n_total = len(df)

    if score_col not in df.columns:
        raise ValueError(
            f"'{score_col}' not found in {path} — regenerate the baseline "
            "with a version that writes this column."
        )

    df_scored = df[(df["predicted_label"] != "ERROR") & df[score_col].notna()].copy()
    n_scored = len(df_scored)
    n_excluded = n_total - n_scored

    print("=" * 60)
    print("ROC / AUC ANALYSIS")
    print("=" * 60)
    print(f"Total rows        : {n_total}")
    print(f"Excluded          : {n_excluded}")
    print(f"Scored rows       : {n_scored}")

    y_true = (df_scored[label_col] == positive_label).astype(int).values
    y_score = df_scored[score_col].values

    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])

    print()
    print(f"AUC               : {auc:.4f}")
    print(f"Best threshold    : {best_threshold:.4f}  (Youden's J = {tpr[best_idx] - fpr[best_idx]:.4f})")

    by_group = {}
    if groupby_col in df_scored.columns:
        print()
        print(f"AUC by {groupby_col}:")
        for val, grp in df_scored.groupby(groupby_col):
            yt = (grp[label_col] == positive_label).astype(int).values
            if len(set(yt)) < 2:
                continue
            grp_auc = float(roc_auc_score(yt, grp[score_col].values))
            by_group[val] = grp_auc
            print(f"  {str(val):<12}: {grp_auc:.4f}  (n={len(grp)})")
    print()

    if plot_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"overall (AUC={auc:.3f})")
        if groupby_col in df_scored.columns:
            for val, grp in df_scored.groupby(groupby_col):
                yt = (grp[label_col] == positive_label).astype(int).values
                if len(set(yt)) < 2:
                    continue
                g_fpr, g_tpr, _ = roc_curve(yt, grp[score_col].values)
                ax.plot(g_fpr, g_tpr, linestyle="--",
                        label=f"{val} (AUC={by_group[val]:.3f})")
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {score_col} vs {label_col}")
        ax.legend(loc="lower right")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ROC plot -> {plot_path}")
        print()

    return {
        "n_total": n_total,
        "n_excluded": n_excluded,
        "n_scored": n_scored,
        "auc": round(auc, 4),
        "best_threshold": round(best_threshold, 4),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "by_group": {str(k): round(v, 4) for k, v in by_group.items()},
    }
