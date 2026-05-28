"""
Joint human + chimp TPM regression using AlphaGenome scalars + MPRA features.

Trains two XGBoost models on the SAME train/test split:
  - XGBoost_human: predicts log10(ExpLBM_TPM_human_allele + 1)
  - XGBoost_chimp: predicts log10(ExpLBM_TPM_chimp_allele  + 1)

Main output: scatter of predicted log10-ratio (human/chimp) vs actual log10-ratio,
             Section 5.2-style (mean-centred, KDE density, red identity line).

Outputs saved to --output-dir:
  - scatter_ratio.png                     ← the key new plot
  - scatter_XGBoost_human.png / _chimp.png
  - scatter_LinearRegression_human.png / _chimp.png
  - feature_importance_human.png / _chimp.png
  - metrics.json
  - feature_names.json
  - scaler.pkl
  - xgboost_human.json / xgboost_chimp.json
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import gaussian_kde, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

MPRA_DIR = os.path.join(os.path.dirname(__file__), "..", "XGBoost_mpra_to_gene_expression")
sys.path.insert(0, os.path.abspath(MPRA_DIR))
import dataMaker as dm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scatter_plot(y_true, y_pred, title, output_path, config_label=""):
    r, _ = pearsonr(y_true, y_pred)
    density = gaussian_kde(np.vstack([y_true, y_pred]))(np.vstack([y_true, y_pred]))
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sc = ax.scatter(y_true, y_pred, c=density, cmap="viridis", alpha=0.6, s=15, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Density")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xlabel("True log₁₀(TPM + 1)")
    ax.set_ylabel("Predicted log₁₀(TPM + 1)")
    ax.set_title(f"{title}  (r={r:.3f})")
    if config_label:
        fig.text(0.5, 0.01, config_label, ha="center", va="bottom",
                 fontsize=7, color="grey", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150)
    plt.close()


def scatter_ratio_plot(actual_ratio, pred_ratio, output_path,
                       title="XGBoost Predicted vs Actual TPM Ratio",
                       config_label=""):
    """Section 5.2-style: mean-centred, KDE density, red identity line."""
    x = np.asarray(actual_ratio, dtype=float)
    y = np.asarray(pred_ratio,   dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    x -= np.mean(x)
    y -= np.mean(y)

    z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    r, p = pearsonr(x, y)
    r2 = r ** 2

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=z, cmap="viridis", s=20, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Point Density")

    lims = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax.plot(lims, lims, "r--", linewidth=2, label="x = y")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.axhline(0, linestyle="--", linewidth=1, color="grey")
    ax.axvline(0, linestyle="--", linewidth=1, color="grey")
    ax.set_xlabel("Actual log₁₀(TPM human / TPM chimp)  [mean-centred]")
    ax.set_ylabel("Predicted log₁₀(TPM human / TPM chimp)  [mean-centred]")
    ax.set_title(title)
    ax.text(
        0.05, 0.95,
        f"N = {len(x)}\nr = {r:.3f}\nR² = {r2:.3f}\np = {p:.2e}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    if config_label:
        fig.text(0.5, 0.01, config_label, ha="center", va="bottom",
                 fontsize=7, color="grey", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")
    return {"pearson_r": float(r), "pearson_pval": float(p), "r2": float(r2)}


def evaluate(y_true, y_pred, model_name):
    r, pval = pearsonr(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    rmse    = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {model_name:<30}  Pearson r={r:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"pearson_r": float(r), "pearson_pval": float(pval),
            "r2": float(r2), "rmse": float(rmse)}


def feature_importance_plot(model, feat_names, output_path, title, config_label=""):
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50]
    fnames, fvals = zip(*sorted_imp)
    fig, ax = plt.subplots(figsize=(7, max(4, len(fnames) * 0.25)))
    ax.barh(range(len(fnames)), fvals[::-1])
    ax.set_yticks(range(len(fnames)))
    ax.set_yticklabels(list(fnames[::-1]), fontsize=7)
    ax.set_xlabel("Gain")
    ax.set_title(title)
    if config_label:
        fig.text(0.5, 0.01, config_label, ha="center", va="bottom",
                 fontsize=7, color="grey", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MPRA gene features (copied from predict_tpm_regression_with_ag_preds.py)
# ---------------------------------------------------------------------------

def build_mpra_gene_features(
    active_oligos_only=True,
    active_thresh=0,
    atacseq_oligos_distance=1,
    elite_lineage_only=False,
    min_barcode_counts=50,
    tissue="ExpLBM",
):
    (mpra_raw, R2GTool_data, human_chimp_hybrids,
     _derived, _hk, _dmr, _tf, _tff,
     _hg, _prna, total_number_of_screen_elements) = dm.load_data()

    filtered_mpra = dm.filter_mpra_data(
        mpra_raw, False, active_oligos_only, active_thresh,
        atacseq_oligos_distance, min_barcode_counts, False,
    )
    filtered_mpra = filtered_mpra.filter(dm.active_only(filtered_mpra, both_active=True))
    R2GTool_filtered = dm.filter_R2GTool(R2GTool_data, elite_lineage_only, False, False)

    filtered_mpra_with_sign = filtered_mpra.with_columns(
        (pl.col("logFC_derived_vs_ancestral") > 0).alias("logFC_sign")
    )
    joined = (
        filtered_mpra_with_sign
        .join(R2GTool_filtered, on=["chromosome", "start", "end"], how="inner")
        .rename({"Gene_symbol": "Gene"})
    )

    tissue_cols = human_chimp_hybrids.select([
        "Gene",
        f"{tissue}_gene_ase_type",
        f"{tissue}_LFC_human_ref",
        f"{tissue}_LFC_padj_human_ref",
        f"{tissue}_TPM_total",
    ])
    final_data = tissue_cols.join(joined, on="Gene", how="inner")

    data_sign = (
        final_data
        .select("Gene", "logFC_sign", f"{tissue}_gene_ase_type",
                f"{tissue}_LFC_human_ref", f"{tissue}_LFC_padj_human_ref",
                f"{tissue}_TPM_total")
        .with_columns((pl.col(f"{tissue}_LFC_human_ref") > 0).alias("gene_logFC_positive_sign"))
    )

    oligo_counts = (
        data_sign
        .group_by(["Gene", f"{tissue}_gene_ase_type", f"{tissue}_LFC_human_ref",
                   "gene_logFC_positive_sign", f"{tissue}_LFC_padj_human_ref",
                   f"{tissue}_TPM_total"])
        .agg([
            pl.len().alias("n_oligos_passed_filters"),
            pl.col("logFC_sign").cast(pl.Int64).sum().alias("n_oligos_positive"),
        ])
        .with_columns(
            (pl.col("n_oligos_positive") / pl.col("n_oligos_passed_filters"))
            .alias("frac_positive_oligos")
        )
    )

    final_with_counts = final_data.join(oligo_counts, on="Gene", how="inner")

    gene_data = (
        final_with_counts
        .with_columns((pl.col(f"{tissue}_LFC_human_ref") > 0).alias("gene_logFC_positive_sign"))
        .filter(pl.col("n_oligos_passed_filters") >= 1)
        .join(total_number_of_screen_elements, on="Gene", how="inner")
    )

    n_compared = (
        mpra_raw
        .join(R2GTool_filtered, on=["chromosome", "start", "end"], how="inner")
        .with_columns(pl.col("Gene_symbol").str.split(";"))
        .explode("Gene_symbol")
        .group_by("Gene_symbol").len()
        .rename({"Gene_symbol": "Gene", "len": "total_number_of_oligos_in_mpra"})
    )
    n_active = (
        mpra_raw.filter(dm.active_only(mpra_raw))
        .with_columns(pl.col("Gene_symbol").str.split(";"))
        .explode("Gene_symbol")
        .group_by("Gene_symbol").len()
        .rename({"Gene_symbol": "Gene", "len": "number_of_active_oligos"})
    )

    gene_data = (
        gene_data
        .join(n_compared, on="Gene", how="inner")
        .join(n_active, on="Gene", how="inner")
        .with_columns(
            (pl.col("number_of_active_oligos") / pl.col("total_number_of_oligos_in_mpra"))
            .alias("frac_active_oligos")
        )
        .with_columns(pl.col(pl.Boolean).cast(pl.Float64))
    )

    oligo_features = [
        "Distance_to_gene(TSS)",
        "DNA_counts_raw_ancestral",
        "barcode_count_ancestral",
        "barcode_count_derived",
        "normalized_activity_estimate_ancestral",
        "normalized_activity_estimate_derived",
        "differential_activity_fdr",
        "variants_count",
        "within_promoter",
        "num_sources_linking_gene",
    ]
    oligo_features = [c for c in oligo_features if c in gene_data.columns]

    agg_exprs = []
    for col in oligo_features:
        agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))

    gene_level_cols = [
        "Gene", "chromosome",
        f"{tissue}_TPM_total",
        "n_oligos_passed_filters", "n_oligos_positive", "frac_positive_oligos",
        "number_of_screen", "total_number_of_oligos_in_mpra",
        "number_of_active_oligos", "frac_active_oligos",
    ]
    gene_level_cols = [c for c in gene_level_cols if c in gene_data.columns]

    oligo_agg   = gene_data.group_by("Gene").agg(agg_exprs)
    gene_static = gene_data.select(gene_level_cols).unique(subset=["Gene"])
    return oligo_agg.join(gene_static, on="Gene", how="inner")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ag_preds_glob: str, output_dir: str, use_gpu: bool, ase_only: bool,
         active_oligos_only: bool = True, elite_lineage_only: bool = False,
         min_barcode_counts: int = 50):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load AG predictions
    # ------------------------------------------------------------------
    print("Loading AlphaGenome predictions...")
    ag = pl.scan_csv(ag_preds_glob, separator="\t").collect()
    if "GeneSymbol" in ag.columns and "Gene" not in ag.columns:
        ag = ag.rename({"GeneSymbol": "Gene"})
    print(f"  {ag.height} genes loaded")

    lfc_arr = ag["LFC"].to_numpy()
    signed_log10_lfc = np.sign(lfc_arr) * np.log10(np.abs(lfc_arr) + 1)
    ag = ag.with_columns(pl.Series("ag_signed_log10_lfc", signed_log10_lfc))

    # ------------------------------------------------------------------
    # Load hybrid labels — need BOTH TPM columns
    # ------------------------------------------------------------------
    hybrids = pl.read_csv(
        config.HUMAN_CHIMP_HYBRIDS_DATA_PATH_WEXAC, separator="\t"
    ).select([
        "Gene",
        "ExpLBM_TPM_human_allele",
        "ExpLBM_TPM_chimp_allele",
        "ExpLBM_gene_ase_type",
    ])

    joined = ag.join(hybrids, on="Gene", how="inner")
    print(f"  {joined.height} genes after join with hybrid labels")

    if ase_only:
        joined = joined.filter(pl.col("ExpLBM_gene_ase_type") == "ASE")
        print(f"  {joined.height} genes after --ase-only filter")

    # ------------------------------------------------------------------
    # Config label for figures
    # ------------------------------------------------------------------
    config_label = (
        f"ASE only: {'yes' if ase_only else 'no'} | "
        f"active oligos (both): {'yes' if active_oligos_only else 'no'} | "
        f"elite lineage: {'yes' if elite_lineage_only else 'no'} | "
        f"min barcodes: {min_barcode_counts}"
    )

    # ------------------------------------------------------------------
    # Build MPRA gene features
    # ------------------------------------------------------------------
    print("Building MPRA gene features...")
    mpra_gene = build_mpra_gene_features(
        active_oligos_only=active_oligos_only,
        elite_lineage_only=elite_lineage_only,
        min_barcode_counts=min_barcode_counts,
    )
    print(f"  {mpra_gene.height} genes, {mpra_gene.width} columns")

    mpra_feat_cols = [c for c in mpra_gene.columns if c not in ("Gene", "chromosome")]
    mpra_lookup = {
        row["Gene"]: [float(row[c]) if row[c] is not None else 0.0 for c in mpra_feat_cols]
        for row in mpra_gene.iter_rows(named=True)
    }

    # ------------------------------------------------------------------
    # Build feature matrix (shared X, two y targets)
    # ------------------------------------------------------------------
    ag_feat_cols = ["HumanGeneExpression", "ChimpGeneExpression", "ag_signed_log10_lfc"]
    all_feat_names = ag_feat_cols + mpra_feat_cols

    rows_X        = []
    y_human_list  = []
    y_chimp_list  = []

    for row in joined.iter_rows(named=True):
        gene = row["Gene"]
        if gene not in mpra_lookup:
            continue
        try:
            ag_feats = [float(row[c]) for c in ag_feat_cols]
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ag_feats).all():
            continue
        try:
            tpm_h = float(row["ExpLBM_TPM_human_allele"])
            tpm_c = float(row["ExpLBM_TPM_chimp_allele"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(tpm_h) or tpm_h < 0:
            continue
        if not np.isfinite(tpm_c) or tpm_c < 0:
            continue
        rows_X.append(ag_feats + mpra_lookup[gene])
        y_human_list.append(tpm_h)
        y_chimp_list.append(tpm_c)

    if not rows_X:
        raise ValueError("No samples remain after filters.")

    ag_X    = np.array([r[:len(ag_feat_cols)] for r in rows_X], dtype=np.float32)
    mpra_X  = np.nan_to_num(
        np.array([r[len(ag_feat_cols):] for r in rows_X], dtype=np.float32), nan=0.0
    )
    X = np.hstack([ag_X, mpra_X])

    y_human = np.log10(np.array(y_human_list, dtype=np.float32) + 1)
    y_chimp = np.log10(np.array(y_chimp_list, dtype=np.float32) + 1)

    print(f"\n  Final dataset: {X.shape[0]} genes × {X.shape[1]} features")
    print(f"  ({len(ag_feat_cols)} AG + {len(mpra_feat_cols)} MPRA)")
    print(f"  y_human range: [{y_human.min():.3f}, {y_human.max():.3f}]")
    print(f"  y_chimp range: [{y_chimp.min():.3f}, {y_chimp.max():.3f}]")

    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(all_feat_names, f, indent=2)

    # ------------------------------------------------------------------
    # Shared train / test split + scaler
    # ------------------------------------------------------------------
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X[idx_train], X[idx_test]
    yh_train, yh_test = y_human[idx_train], y_human[idx_test]
    yc_train, yc_test = y_chimp[idx_train], y_chimp[idx_test]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    all_metrics = {}
    print(f"\n{'Model':<30}  {'Pearson r':>10}  {'R²':>8}  {'RMSE':>8}")
    print("-" * 63)

    # ------------------------------------------------------------------
    # 1. Linear Regression (both species)
    # ------------------------------------------------------------------
    for species, y_tr, y_te in [("human", yh_train, yh_test),
                                 ("chimp", yc_train, yc_test)]:
        lr = LinearRegression().fit(X_train, y_tr)
        y_pred = lr.predict(X_test)
        all_metrics[f"LinearRegression_{species}"] = evaluate(y_te, y_pred, f"LinearRegression_{species}")
        scatter_plot(y_te, y_pred,
                     f"LinearRegression — {species}",
                     os.path.join(output_dir, f"scatter_LinearRegression_{species}.png"),
                     config_label=config_label)

    # ------------------------------------------------------------------
    # 2. XGBoost — human
    # ------------------------------------------------------------------
    def make_xgb(use_gpu):
        return xgb.XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.3,
            eval_metric="rmse",
            early_stopping_rounds=30,
            device="cuda" if use_gpu else "cpu",
            random_state=42,
            n_jobs=-1,
            feature_names=all_feat_names,
        )

    print("\nTraining XGBoost — human...")
    xgb_human = make_xgb(use_gpu)
    xgb_human.fit(X_train, yh_train, eval_set=[(X_test, yh_test)], verbose=50)
    yh_pred = xgb_human.predict(X_test)
    all_metrics["XGBoost_human"] = evaluate(yh_test, yh_pred, "XGBoost_human")
    scatter_plot(yh_test, yh_pred,
                 "XGBoost — human",
                 os.path.join(output_dir, "scatter_XGBoost_human.png"),
                 config_label=config_label)
    xgb_human.save_model(os.path.join(output_dir, "xgboost_human.json"))
    feature_importance_plot(xgb_human, all_feat_names,
                            os.path.join(output_dir, "feature_importance_human.png"),
                            "XGBoost Top-50 Feature Importance — human",
                            config_label=config_label)

    # ------------------------------------------------------------------
    # 3. XGBoost — chimp
    # ------------------------------------------------------------------
    print("\nTraining XGBoost — chimp...")
    xgb_chimp = make_xgb(use_gpu)
    xgb_chimp.fit(X_train, yc_train, eval_set=[(X_test, yc_test)], verbose=50)
    yc_pred = xgb_chimp.predict(X_test)
    all_metrics["XGBoost_chimp"] = evaluate(yc_test, yc_pred, "XGBoost_chimp")
    scatter_plot(yc_test, yc_pred,
                 "XGBoost — chimp",
                 os.path.join(output_dir, "scatter_XGBoost_chimp.png"),
                 config_label=config_label)
    xgb_chimp.save_model(os.path.join(output_dir, "xgboost_chimp.json"))
    feature_importance_plot(xgb_chimp, all_feat_names,
                            os.path.join(output_dir, "feature_importance_chimp.png"),
                            "XGBoost Top-50 Feature Importance — chimp",
                            config_label=config_label)

    # ------------------------------------------------------------------
    # 4. Ratio scatter: predicted log10(human/chimp) vs actual
    # ------------------------------------------------------------------
    print("\nGenerating TPM ratio scatter...")
    actual_log_ratio = yh_test - yc_test          # log10(actual_human / actual_chimp)
    pred_log_ratio   = yh_pred - yc_pred           # log10(pred_human   / pred_chimp)

    ratio_metrics = scatter_ratio_plot(
        actual_ratio=actual_log_ratio,
        pred_ratio=pred_log_ratio,
        output_path=os.path.join(output_dir, "scatter_ratio.png"),
        config_label=config_label,
    )
    all_metrics["XGBoost_ratio"] = ratio_metrics

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics["_config"] = {
        "ag_preds_glob": ag_preds_glob,
        "ase_only": ase_only,
        "active_oligos_only": active_oligos_only,
        "elite_lineage_only": elite_lineage_only,
        "min_barcode_counts": min_barcode_counts,
        "target": "log10(TPM + 1)",
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "n_ag_features": len(ag_feat_cols),
        "n_mpra_features": len(mpra_feat_cols),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n=== Summary ===")
    for name, m in all_metrics.items():
        if name.startswith("_"):
            continue
        rmse_str = f"  RMSE={m['rmse']:.4f}" if "rmse" in m else ""
        print(f"  {name:<28}  r={m['pearson_r']:.4f}  R²={m['r2']:.4f}{rmse_str}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint human+chimp TPM XGBoost regression + predicted ratio scatter"
    )
    parser.add_argument("--ag-preds-glob", type=str, default="results/all_genes/*.tsv")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ase-only", action="store_true",
                        help="Restrict to genes classified as ASE in ExpLBM")
    parser.add_argument("--no-active-oligos-only", action="store_true",
                        help="Include oligos that are not active in either allele (default: active in both)")
    parser.add_argument("--elite-lineage-only", action="store_true",
                        help="Restrict to elite R2GTool links (promoter or ≥2 sources)")
    parser.add_argument("--min-barcode-counts", type=int, default=50,
                        help="Minimum barcode count per oligo (default: 50)")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    active_oligos_only = not args.no_active_oligos_only

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ase_tag = "_ase" if args.ase_only else ""
    output_dir = args.output_dir or f"results/ag_preds_tpm_ratio{ase_tag}_{run_ts}"

    main(
        ag_preds_glob=args.ag_preds_glob,
        output_dir=output_dir,
        use_gpu=args.gpu,
        ase_only=args.ase_only,
        active_oligos_only=active_oligos_only,
        elite_lineage_only=args.elite_lineage_only,
        min_barcode_counts=args.min_barcode_counts,
    )
