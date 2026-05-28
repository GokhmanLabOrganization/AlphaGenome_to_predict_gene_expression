"""
Oligo-level XGBoost LFC regression using AlphaGenome scalar predictions + MPRA.

Each MPRA oligo linked to a gene becomes a separate training sample:
  features = AG gene-level scalars + individual oligo MPRA measurements
  target   = experimental LFC (gene-level, repeated for all oligos of that gene)

Train/test split is done at the GENE level to prevent leakage.
At inference, per-oligo predictions are averaged to obtain a gene-level prediction.

Models:
  - LinearRegression  (OLS baseline)
  - XGBoostRegressor  (gradient-boosted trees)

Default: ASE genes only (--no-ase-only to include all genes).

Outputs saved to --output-dir:
  - metrics.json
  - scatter_LinearRegression.png
  - scatter_XGBoost.png
  - feature_importance_xgboost.png
  - feature_names.json
  - scaler.pkl
  - xgboost_regressor.json
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
from sklearn.preprocessing import StandardScaler

import config

MPRA_DIR = os.path.join(os.path.dirname(__file__), "..", "XGBoost_mpra_to_gene_expression")
sys.path.insert(0, os.path.abspath(MPRA_DIR))
import dataMaker as dm


# ---------------------------------------------------------------------------
# Oligo feature columns (all available after MPRA × R2GTool join)
# ---------------------------------------------------------------------------

OLIGO_BASE_FEATURES = [
    "logFC_derived_vs_ancestral",
    "normalized_activity_estimate_ancestral",
    "normalized_activity_estimate_derived",
    "Distance_to_gene(TSS)",
    "differential_activity_fdr",
    "within_promoter",
    "num_sources_linking_gene",
    "barcode_count_ancestral",
    "barcode_count_derived",
    "DNA_counts_raw_ancestral",
    "variants_count",
]


def add_oligo_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add engineered columns at the oligo level."""
    df = df.with_columns([
        pl.col("normalized_activity_estimate_derived").log().alias("log_activity_derived"),
        (pl.col("logFC_derived_vs_ancestral") > 0).cast(pl.Float64).alias("logFC_sign"),
        pl.col("Distance_to_gene(TSS)").abs().alias("distance_to_TSS_abs"),
        (pl.col("differential_activity_fdr") <= 0.05).cast(pl.Float64).alias("sig_diff_activity"),
    ])
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scatter_plot(y_true, y_pred, model_name, output_dir):
    r, _ = pearsonr(y_true, y_pred)
    density = gaussian_kde(np.vstack([y_true, y_pred]))(np.vstack([y_true, y_pred]))
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sc = ax.scatter(y_true, y_pred, c=density, cmap="viridis", alpha=0.6, s=20, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Density")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xlabel("True LFC  (gene-level)")
    ax.set_ylabel("Predicted LFC  (mean over oligos)")
    ax.set_title(f"{model_name}  (Pearson r={r:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_{model_name.replace(' ', '_')}.png"), dpi=150)
    plt.close()


def evaluate(y_true, y_pred, model_name):
    r, pval = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {model_name:<25}  Pearson r={r:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"pearson_r": float(r), "pearson_pval": float(pval),
            "r2": float(r2), "rmse": float(rmse)}


def gene_level_predict(model, X_oligos, gene_ids, scaler=None):
    """Predict per oligo, then average per gene. Returns (genes, y_pred_gene)."""
    X = scaler.transform(X_oligos) if scaler else X_oligos
    preds = model.predict(X)
    gene_arr = np.array(gene_ids)
    genes = np.unique(gene_arr)
    y_pred = np.array([preds[gene_arr == g].mean() for g in genes])
    return genes, y_pred


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------

def build_oligo_level_data(
    hybrids_joined: pl.DataFrame,
    active_oligos_only: bool = True,
    min_barcode_counts: int = 50,
    elite_lineage_only: bool = False,
    tissue: str = "ExpLBM",
) -> pl.DataFrame:
    """
    Returns a DataFrame with one row per (Gene, oligo) pair.

    Columns: Gene, ExpLBM_LFC_human_ref, HumanGeneExpression, ChimpGeneExpression,
             ag_signed_log10_lfc, <oligo feature cols>
    """
    print("Loading MPRA data...")
    (mpra_raw, R2GTool_data, _human_chimp_hybrids,
     *_rest) = dm.load_data()

    filtered_mpra = dm.filter_mpra_data(
        mpra_raw, False, active_oligos_only, 0,
        1, min_barcode_counts, False,
    )
    filtered_mpra = filtered_mpra.filter(dm.active_only(filtered_mpra, both_active=True))
    R2GTool_filtered = dm.filter_R2GTool(R2GTool_data, elite_lineage_only, False, False)

    print(f"  {filtered_mpra.height} oligos after MPRA filters")
    print(f"  {R2GTool_filtered.height} R2GTool links after filter")

    oligo_gene = (
        filtered_mpra
        .join(R2GTool_filtered, on=["chromosome", "start", "end"], how="inner")
        .rename({"Gene_symbol": "Gene"})
    )
    print(f"  {oligo_gene.height} (gene, oligo) pairs after join")

    oligo_gene = add_oligo_derived_features(oligo_gene)

    # Join AG predictions + hybrid labels (gene-level columns)
    gene_cols = ["Gene", "ExpLBM_LFC_human_ref",
                 "HumanGeneExpression", "ChimpGeneExpression", "ag_signed_log10_lfc"]
    available = [c for c in gene_cols if c in hybrids_joined.columns]
    oligo_with_labels = oligo_gene.join(
        hybrids_joined.select(available), on="Gene", how="inner"
    )
    print(f"  {oligo_with_labels.height} rows after joining with AG+hybrids ({oligo_with_labels['Gene'].n_unique()} genes)")
    return oligo_with_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ag_preds_glob: str, output_dir: str, lfc_threshold: float,
         use_gpu: bool, ase_only: bool, target: str = "lfc"):
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
    # Load hybrid labels
    # ------------------------------------------------------------------
    hybrids = pl.read_csv(
        config.HUMAN_CHIMP_HYBRIDS_DATA_PATH_WEXAC, separator="\t"
    ).select(["Gene", "ExpLBM_LFC_human_ref", "ExpLBM_gene_ase_type"])

    joined = ag.join(hybrids, on="Gene", how="inner")
    print(f"  {joined.height} genes after join with hybrid labels")

    if ase_only:
        joined = joined.filter(pl.col("ExpLBM_gene_ase_type") == "ASE")
        print(f"  {joined.height} genes after ASE filter")

    if lfc_threshold > 0:
        joined = joined.filter(pl.col("ExpLBM_LFC_human_ref").abs() >= lfc_threshold)
        print(f"  {joined.height} genes after |LFC| >= {lfc_threshold} filter")

    # ------------------------------------------------------------------
    # Build oligo-level data
    # ------------------------------------------------------------------
    oligo_df = build_oligo_level_data(joined)

    # Determine available oligo feature columns
    derived_features = ["log_activity_derived", "logFC_sign", "distance_to_TSS_abs", "sig_diff_activity"]
    oligo_feat_cols = [c for c in OLIGO_BASE_FEATURES + derived_features if c in oligo_df.columns]
    ag_feat_cols = ["HumanGeneExpression", "ChimpGeneExpression", "ag_signed_log10_lfc"]
    all_feat_names = ag_feat_cols + oligo_feat_cols

    print(f"\n  Oligo features: {len(oligo_feat_cols)}")
    print(f"  AG features:    {len(ag_feat_cols)}")

    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(all_feat_names, f, indent=2)

    # ------------------------------------------------------------------
    # Build feature matrix (oligo level)
    # ------------------------------------------------------------------
    genes_all = oligo_df["Gene"].to_list()
    y_gene_map = {
        row["Gene"]: float(row["ExpLBM_LFC_human_ref"])
        for row in joined.iter_rows(named=True)
        if row["Gene"] in set(genes_all)
    }

    rows_X, rows_y, rows_gene = [], [], []
    for row in oligo_df.iter_rows(named=True):
        gene = row["Gene"]
        try:
            ag_feats = [float(row[c]) for c in ag_feat_cols]
            oligo_feats = [float(row[c]) if row[c] is not None else 0.0 for c in oligo_feat_cols]
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ag_feats).all():
            continue
        y_val = y_gene_map.get(gene)
        if y_val is None or not np.isfinite(y_val):
            continue
        rows_X.append(ag_feats + oligo_feats)
        rows_y.append(y_val)
        rows_gene.append(gene)

    X_all = np.nan_to_num(np.array(rows_X, dtype=np.float32), nan=0.0)
    y_all = np.array(rows_y, dtype=np.float32)
    genes_all = np.array(rows_gene)

    if target == "log10_lfc":
        y_all = (np.sign(y_all) * np.log10(np.abs(y_all) + 1)).astype(np.float32)
        target_label = "sign(LFC)×log10(|LFC|+1)"
    else:
        target_label = "LFC (raw)"

    unique_genes = np.unique(genes_all)
    print(f"\n  {len(X_all)} oligo rows  ({len(unique_genes)} unique genes)")
    print(f"  {X_all.shape[1]} total features")
    print(f"  Target: {target_label}")

    # ------------------------------------------------------------------
    # Gene-level train / test split (prevents leakage)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(unique_genes)
    n_test = max(1, int(len(shuffled) * 0.2))
    test_genes  = set(shuffled[:n_test])
    train_genes = set(shuffled[n_test:])

    train_mask = np.array([g in train_genes for g in genes_all])
    test_mask  = np.array([g in test_genes  for g in genes_all])

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
    genes_test = genes_all[test_mask]

    # Gene-level ground truth for test set (one value per gene)
    test_gene_list = np.unique(genes_test)
    y_test_gene = np.array([y_test[genes_test == g].mean() for g in test_gene_list])

    print(f"  Train: {train_mask.sum()} oligos from {len(train_genes)} genes")
    print(f"  Test:  {test_mask.sum()} oligos from {len(test_genes)} genes")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    all_metrics = {}
    print(f"\n{'Model':<25}  {'Pearson r':>10}  {'R²':>8}  {'RMSE':>8}")
    print("-" * 57)

    # ------------------------------------------------------------------
    # 1. Linear Regression
    # ------------------------------------------------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train_sc, y_train)
    _, y_pred_lr = gene_level_predict(lr_model, X_test, genes_test)
    # align gene order
    y_pred_lr_aligned = np.array([y_pred_lr[np.where(np.unique(genes_test) == g)[0][0]] for g in test_gene_list])
    all_metrics["LinearRegression"] = evaluate(y_test_gene, y_pred_lr_aligned, "LinearRegression")
    scatter_plot(y_test_gene, y_pred_lr_aligned, "LinearRegression", output_dir)

    # ------------------------------------------------------------------
    # 2. XGBoost
    # ------------------------------------------------------------------
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        eval_metric="rmse",
        early_stopping_rounds=30,
        device="cuda" if use_gpu else "cpu",
        random_state=42,
        n_jobs=-1,
        feature_names=all_feat_names,
    )
    xgb_reg.fit(X_train_sc, y_train, eval_set=[(X_test_sc, y_test)], verbose=50)

    _, y_pred_xgb = gene_level_predict(xgb_reg, X_test, genes_test, scaler=scaler)
    y_pred_xgb_aligned = np.array([y_pred_xgb[np.where(np.unique(genes_test) == g)[0][0]] for g in test_gene_list])
    all_metrics["XGBoost"] = evaluate(y_test_gene, y_pred_xgb_aligned, "XGBoost")
    scatter_plot(y_test_gene, y_pred_xgb_aligned, "XGBoost", output_dir)
    xgb_reg.save_model(os.path.join(output_dir, "xgboost_regressor.json"))

    importance = xgb_reg.get_booster().get_score(importance_type="gain")
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50]
        fnames, fvals = zip(*sorted_imp)
        fig, ax = plt.subplots(figsize=(7, max(4, len(fnames) * 0.28)))
        ax.barh(range(len(fnames)), fvals[::-1])
        ax.set_yticks(range(len(fnames)))
        ax.set_yticklabels(list(fnames[::-1]), fontsize=7)
        ax.set_xlabel("Gain")
        ax.set_title("XGBoost Top-50 Feature Importance (oligo-level)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance_xgboost.png"), dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics["_config"] = {
        "ag_preds_glob": ag_preds_glob,
        "lfc_threshold": lfc_threshold,
        "ase_only": ase_only,
        "target": target,
        "target_label": target_label,
        "n_train_oligos": int(train_mask.sum()),
        "n_test_oligos": int(test_mask.sum()),
        "n_train_genes": len(train_genes),
        "n_test_genes": len(test_genes),
        "n_ag_features": len(ag_feat_cols),
        "n_oligo_features": len(oligo_feat_cols),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n=== Summary ===")
    for name, m in all_metrics.items():
        if name.startswith("_"):
            continue
        print(f"  {name:<22}  r={m['pearson_r']:.4f}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Oligo-level LFC regression: AG scalars + per-oligo MPRA features"
    )
    parser.add_argument(
        "--ag-preds-glob",
        type=str,
        default="results/all_genes/*.tsv",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lfc-threshold", type=float, default=0.0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--no-ase-only",
        action="store_true",
        help="Include all genes (default: ASE genes only)",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["lfc", "log10_lfc"],
        default="lfc",
    )
    args = parser.parse_args()

    ase_only = not args.no_ase_only

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ase_tag = "_ase" if ase_only else ""
    output_dir = args.output_dir or f"results/ag_preds_oligo_{args.target}{ase_tag}_{run_ts}"

    main(
        ag_preds_glob=args.ag_preds_glob,
        output_dir=output_dir,
        lfc_threshold=args.lfc_threshold,
        use_gpu=args.gpu,
        ase_only=ase_only,
        target=args.target,
    )
