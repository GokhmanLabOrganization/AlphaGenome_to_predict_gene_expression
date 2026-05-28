"""
Stacked LFC regression: TPM XGBoost predictions + differential MPRA features.

Strategy (avoids leakage):
  1. Split genes 80/20 (train / test).
  2. Run 5-fold CV on the 80 % train genes to get out-of-fold (OOF)
     human-TPM and chimp-TPM predictions.
  3. Train final TPM XGBoost models on all 80 % train genes; predict test genes.
  4. Build stacking features for every gene:
       - pred_log10_human, pred_log10_chimp  (OOF for train; final-model for test)
       - pred_log_ratio = pred_log10_human - pred_log10_chimp
       - differential MPRA features (frac_positive_oligos,
         logFC_derived_vs_ancestral mean/std, n_oligos, ...)
       - AG LFC scalars (ag_signed_log10_lfc, HumanGeneExpression,
         ChimpGeneExpression)
  5. Train stacking models (LinearRegression / Ridge / XGBoost) on train
     stacking features; target = ExpLBM_LFC_human_ref.
  6. Evaluate on test stacking features.

Outputs saved to --output-dir:
  - metrics.json
  - scatter_<model>.png   for each stacking model
  - feature_importance_xgboost_stack.png
  - scatter_xgboost_human_tpm.png / _chimp_tpm.png  (TPM quality checks)
  - feature_names_tpm.json / feature_names_stack.json
  - scaler_tpm.pkl / scaler_stack.pkl
  - xgboost_tpm_human.json / xgboost_tpm_chimp.json
  - xgboost_stack.json
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import config

MPRA_DIR = os.path.join(os.path.dirname(__file__), "..", "XGBoost_mpra_to_gene_expression")
sys.path.insert(0, os.path.abspath(MPRA_DIR))
import dataMaker as dm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scatter_plot(y_true, y_pred, title, output_path, xlabel="True LFC",
                 ylabel="Predicted LFC", config_label=""):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return
    r, p = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    density = gaussian_kde(np.vstack([y_true, y_pred]))(np.vstack([y_true, y_pred]))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(y_true, y_pred, c=density, cmap="viridis", alpha=0.7, s=20, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Density")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="x = y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.05, 0.95,
            f"N = {len(y_true)}\nr = {r:.3f}\nR² = {r2:.3f}\np = {p:.2e}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    if config_label:
        fig.text(0.5, 0.01, config_label, ha="center", va="bottom",
                 fontsize=7, color="grey", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate(y_true, y_pred, model_name):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    r, pval = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {model_name:<30}  r={r:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"pearson_r": float(r), "pearson_pval": float(pval),
            "r2": float(r2), "rmse": float(rmse)}


def feat_importance_plot(model, feat_names, output_path, title, config_label=""):
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:40]
    fnames, fvals = zip(*sorted_imp)
    fig, ax = plt.subplots(figsize=(7, max(4, len(fnames) * 0.28)))
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
# MPRA features — extended with differential logFC aggregation
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

    joined = (
        filtered_mpra
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

    # sign per oligo
    final_data = final_data.with_columns(
        (pl.col("logFC_derived_vs_ancestral") > 0).alias("logFC_sign")
    )

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

    # Oligo-level features aggregated per gene (mean + std)
    oligo_features = [
        "logFC_derived_vs_ancestral",   # ← key differential signal
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
# TPM XGBoost builder
# ---------------------------------------------------------------------------

def make_tpm_xgb(use_gpu, feat_names):
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
        feature_names=feat_names,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ag_preds_glob: str, output_dir: str, use_gpu: bool, ase_only: bool,
         target: str = "lfc", n_folds: int = 5,
         active_oligos_only: bool = True, elite_lineage_only: bool = False,
         min_barcode_counts: int = 50):
    os.makedirs(output_dir, exist_ok=True)

    config_label = (
        f"ASE only: {'yes' if ase_only else 'no'} | "
        f"active oligos (both): {'yes' if active_oligos_only else 'no'} | "
        f"elite lineage: {'yes' if elite_lineage_only else 'no'} | "
        f"min barcodes: {min_barcode_counts} | "
        f"CV folds: {n_folds} | target: {target}"
    )

    # ------------------------------------------------------------------
    # Load AG predictions
    # ------------------------------------------------------------------
    print("Loading AlphaGenome predictions...")
    ag = pl.scan_csv(ag_preds_glob, separator="\t").collect()
    if "GeneSymbol" in ag.columns and "Gene" not in ag.columns:
        ag = ag.rename({"GeneSymbol": "Gene"})
    lfc_arr = ag["LFC"].to_numpy()
    ag = ag.with_columns(
        pl.Series("ag_signed_log10_lfc",
                  np.sign(lfc_arr) * np.log10(np.abs(lfc_arr) + 1))
    )
    print(f"  {ag.height} genes loaded")

    # ------------------------------------------------------------------
    # Load hybrid labels
    # ------------------------------------------------------------------
    hybrids = pl.read_csv(
        config.HUMAN_CHIMP_HYBRIDS_DATA_PATH_WEXAC, separator="\t"
    ).select([
        "Gene",
        "ExpLBM_LFC_human_ref",
        "ExpLBM_TPM_human_allele",
        "ExpLBM_TPM_chimp_allele",
        "ExpLBM_gene_ase_type",
    ])

    joined = ag.join(hybrids, on="Gene", how="inner")
    print(f"  {joined.height} genes after join with hybrid labels")

    if ase_only:
        joined = joined.filter(pl.col("ExpLBM_gene_ase_type") == "ASE")
        print(f"  {joined.height} genes after ASE filter")

    # ------------------------------------------------------------------
    # Build MPRA gene features (with logFC_derived_vs_ancestral)
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
    # Build shared feature matrix for TPM models
    # ------------------------------------------------------------------
    ag_feat_cols = ["HumanGeneExpression", "ChimpGeneExpression", "ag_signed_log10_lfc"]
    tpm_feat_names = ag_feat_cols + mpra_feat_cols

    genes_list, X_tpm_list = [], []
    y_human_list, y_chimp_list, y_lfc_list = [], [], []

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
            lfc   = float(row["ExpLBM_LFC_human_ref"])
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite([tpm_h, tpm_c, lfc])):
            continue
        if tpm_h < 0 or tpm_c < 0:
            continue

        genes_list.append(gene)
        X_tpm_list.append(ag_feats + mpra_lookup[gene])
        y_human_list.append(tpm_h)
        y_chimp_list.append(tpm_c)
        y_lfc_list.append(lfc)

    if not genes_list:
        raise ValueError("No samples remain after filters.")

    genes_arr = np.array(genes_list)
    ag_X     = np.array([r[:len(ag_feat_cols)] for r in X_tpm_list], dtype=np.float32)
    mpra_X   = np.nan_to_num(
        np.array([r[len(ag_feat_cols):] for r in X_tpm_list], dtype=np.float32), nan=0.0
    )
    X_tpm    = np.hstack([ag_X, mpra_X])
    y_human  = np.log10(np.array(y_human_list, dtype=np.float32) + 1)
    y_chimp  = np.log10(np.array(y_chimp_list, dtype=np.float32) + 1)
    y_lfc    = np.array(y_lfc_list, dtype=np.float32)

    if target == "log10_lfc":
        y_lfc_target = (np.sign(y_lfc) * np.log10(np.abs(y_lfc) + 1)).astype(np.float32)
        target_label = "sign(LFC)×log10(|LFC|+1)"
    else:
        y_lfc_target = y_lfc
        target_label = "LFC (raw)"

    print(f"\n  Dataset: {len(genes_arr)} genes × {X_tpm.shape[1]} TPM features")
    print(f"  Target: {target_label}")

    with open(os.path.join(output_dir, "feature_names_tpm.json"), "w") as f:
        json.dump(tpm_feat_names, f, indent=2)

    # ------------------------------------------------------------------
    # Gene-level 80 / 20 train / test split
    # ------------------------------------------------------------------
    unique_genes = np.unique(genes_arr)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(unique_genes)
    n_test      = max(1, int(len(shuffled) * 0.2))
    test_genes  = set(shuffled[:n_test])
    train_genes = set(shuffled[n_test:])

    train_idx = np.where([g in train_genes for g in genes_arr])[0]
    test_idx  = np.where([g in test_genes  for g in genes_arr])[0]

    X_tr, X_te = X_tpm[train_idx], X_tpm[test_idx]
    yh_tr, yh_te = y_human[train_idx], y_human[test_idx]
    yc_tr, yc_te = y_chimp[train_idx], y_chimp[test_idx]
    ylfc_te = y_lfc_target[test_idx]
    genes_te = genes_arr[test_idx]

    tpm_scaler = StandardScaler()
    X_tr_sc = tpm_scaler.fit_transform(X_tr)
    X_te_sc = tpm_scaler.transform(X_te)

    with open(os.path.join(output_dir, "scaler_tpm.pkl"), "wb") as f:
        pickle.dump(tpm_scaler, f)

    print(f"  Train: {len(train_idx)} genes | Test: {len(test_idx)} genes")

    # ------------------------------------------------------------------
    # Step 1: 5-fold CV on train genes → OOF TPM predictions
    # ------------------------------------------------------------------
    print(f"\nRunning {n_folds}-fold CV for OOF TPM predictions...")
    oof_pred_human = np.zeros(len(train_idx), dtype=np.float32)
    oof_pred_chimp = np.zeros(len(train_idx), dtype=np.float32)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (fold_tr, fold_val) in enumerate(kf.split(X_tr_sc)):
        print(f"  Fold {fold + 1}/{n_folds} ...")
        fold_scaler = StandardScaler()
        Xf_tr = fold_scaler.fit_transform(X_tr[fold_tr])
        Xf_val = fold_scaler.transform(X_tr[fold_val])

        for species, y_tr_fold, oof_arr in [
            ("human", yh_tr[fold_tr], oof_pred_human),
            ("chimp", yc_tr[fold_tr], oof_pred_chimp),
        ]:
            m = make_tpm_xgb(use_gpu, tpm_feat_names)
            m.fit(Xf_tr, y_tr_fold,
                  eval_set=[(Xf_val, yh_tr[fold_val] if species == "human"
                              else yc_tr[fold_val])],
                  verbose=False)
            oof_arr[fold_val] = m.predict(Xf_val)

    print(f"  OOF human TPM: r={pearsonr(yh_tr, oof_pred_human)[0]:.3f}")
    print(f"  OOF chimp TPM: r={pearsonr(yc_tr, oof_pred_chimp)[0]:.3f}")

    # ------------------------------------------------------------------
    # Step 2: Final TPM models trained on all train genes
    # ------------------------------------------------------------------
    print("\nTraining final TPM models on all train genes...")
    xgb_human_final = make_tpm_xgb(use_gpu, tpm_feat_names)
    xgb_human_final.fit(X_tr_sc, yh_tr, eval_set=[(X_te_sc, yh_te)], verbose=50)
    te_pred_human = xgb_human_final.predict(X_te_sc)

    xgb_chimp_final = make_tpm_xgb(use_gpu, tpm_feat_names)
    xgb_chimp_final.fit(X_tr_sc, yc_tr, eval_set=[(X_te_sc, yc_te)], verbose=50)
    te_pred_chimp = xgb_chimp_final.predict(X_te_sc)

    xgb_human_final.save_model(os.path.join(output_dir, "xgboost_tpm_human.json"))
    xgb_chimp_final.save_model(os.path.join(output_dir, "xgboost_tpm_chimp.json"))

    # TPM quality-check plots
    scatter_plot(yh_te, te_pred_human,
                 "XGBoost TPM — human (test)",
                 os.path.join(output_dir, "scatter_xgboost_human_tpm.png"),
                 xlabel="True log₁₀(TPM human + 1)",
                 ylabel="Predicted log₁₀(TPM human + 1)",
                 config_label=config_label)
    scatter_plot(yc_te, te_pred_chimp,
                 "XGBoost TPM — chimp (test)",
                 os.path.join(output_dir, "scatter_xgboost_chimp_tpm.png"),
                 xlabel="True log₁₀(TPM chimp + 1)",
                 ylabel="Predicted log₁₀(TPM chimp + 1)",
                 config_label=config_label)

    all_metrics = {
        "XGBoost_TPM_human": evaluate(yh_te, te_pred_human, "XGBoost_TPM_human"),
        "XGBoost_TPM_chimp": evaluate(yc_te, te_pred_chimp, "XGBoost_TPM_chimp"),
    }

    # ------------------------------------------------------------------
    # Step 3: Build stacking feature matrices
    # ------------------------------------------------------------------
    # Train stacking features use OOF predictions (no leakage)
    oof_log_ratio = oof_pred_human - oof_pred_chimp
    te_log_ratio  = te_pred_human  - te_pred_chimp

    # Additional AG scalars per gene for the stacking layer
    ag_stack_cols = ["HumanGeneExpression", "ChimpGeneExpression", "ag_signed_log10_lfc"]
    ag_stack_lookup = {
        row["Gene"]: [float(row[c]) for c in ag_stack_cols]
        for row in joined.iter_rows(named=True)
        if row["Gene"] in (train_genes | test_genes)
    }

    def build_stack_X(idxs, oof_h, oof_c, oof_ratio):
        rows = []
        for i, idx in enumerate(idxs):
            gene = genes_arr[idx]
            ag_s = ag_stack_lookup.get(gene, [0.0] * len(ag_stack_cols))
            mpra_s = mpra_lookup.get(gene, [0.0] * len(mpra_feat_cols))
            rows.append([oof_h[i], oof_c[i], oof_ratio[i]] + ag_s + mpra_s)
        return np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0)

    stack_feat_names = (
        ["pred_log10_human", "pred_log10_chimp", "pred_log_ratio"]
        + ag_stack_cols
        + mpra_feat_cols
    )

    X_stack_tr = build_stack_X(train_idx, oof_pred_human, oof_pred_chimp, oof_log_ratio)
    X_stack_te = build_stack_X(test_idx,  te_pred_human,  te_pred_chimp,  te_log_ratio)
    y_stack_tr = y_lfc_target[train_idx]
    y_stack_te = ylfc_te

    with open(os.path.join(output_dir, "feature_names_stack.json"), "w") as f:
        json.dump(stack_feat_names, f, indent=2)

    stack_scaler = StandardScaler()
    X_stack_tr_sc = stack_scaler.fit_transform(X_stack_tr)
    X_stack_te_sc = stack_scaler.transform(X_stack_te)

    with open(os.path.join(output_dir, "scaler_stack.pkl"), "wb") as f:
        pickle.dump(stack_scaler, f)

    print(f"\n  Stacking features: {X_stack_tr.shape[1]}")
    print(f"  Stacking train: {len(y_stack_tr)} genes  |  test: {len(y_stack_te)} genes")
    print(f"\n{'Model':<30}  {'Pearson r':>10}  {'R²':>8}  {'RMSE':>8}")
    print("-" * 63)

    # ------------------------------------------------------------------
    # Stacking models
    # ------------------------------------------------------------------

    # 1. Linear Regression
    lr = LinearRegression().fit(X_stack_tr_sc, y_stack_tr)
    y_pred = lr.predict(X_stack_te_sc)
    all_metrics["LinearRegression_stack"] = evaluate(y_stack_te, y_pred, "LinearRegression_stack")
    scatter_plot(y_stack_te, y_pred,
                 f"LinearRegression stack  [{target_label}]",
                 os.path.join(output_dir, "scatter_LinearRegression_stack.png"),
                 xlabel=f"True {target_label}", ylabel=f"Predicted {target_label}",
                 config_label=config_label)

    # 2. Ridge (regularised — good for small N)
    ridge = Ridge(alpha=1.0).fit(X_stack_tr_sc, y_stack_tr)
    y_pred = ridge.predict(X_stack_te_sc)
    all_metrics["Ridge_stack"] = evaluate(y_stack_te, y_pred, "Ridge_stack")
    scatter_plot(y_stack_te, y_pred,
                 f"Ridge stack  [{target_label}]",
                 os.path.join(output_dir, "scatter_Ridge_stack.png"),
                 xlabel=f"True {target_label}", ylabel=f"Predicted {target_label}",
                 config_label=config_label)

    # 3. XGBoost stacking model
    xgb_stack = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        eval_metric="rmse",
        early_stopping_rounds=20,
        device="cuda" if use_gpu else "cpu",
        random_state=42,
        n_jobs=-1,
        feature_names=stack_feat_names,
    )
    xgb_stack.fit(X_stack_tr_sc, y_stack_tr,
                  eval_set=[(X_stack_te_sc, y_stack_te)], verbose=30)
    y_pred = xgb_stack.predict(X_stack_te_sc)
    all_metrics["XGBoost_stack"] = evaluate(y_stack_te, y_pred, "XGBoost_stack")
    scatter_plot(y_stack_te, y_pred,
                 f"XGBoost stack  [{target_label}]",
                 os.path.join(output_dir, "scatter_XGBoost_stack.png"),
                 xlabel=f"True {target_label}", ylabel=f"Predicted {target_label}",
                 config_label=config_label)
    xgb_stack.save_model(os.path.join(output_dir, "xgboost_stack.json"))
    feat_importance_plot(xgb_stack, stack_feat_names,
                         os.path.join(output_dir, "feature_importance_xgboost_stack.png"),
                         "XGBoost Stack — Top-40 Feature Importance",
                         config_label=config_label)

    # ------------------------------------------------------------------
    # Baseline: ratio-only linear model (no MPRA, just TPM ratio)
    # ------------------------------------------------------------------
    lr_ratio = LinearRegression().fit(
        X_stack_tr[:, [2]],   # pred_log_ratio only
        y_stack_tr
    )
    y_pred = lr_ratio.predict(X_stack_te[:, [2]])
    all_metrics["LinearRegression_ratio_only"] = evaluate(
        y_stack_te, y_pred, "LinearRegression_ratio_only"
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics["_config"] = {
        "ag_preds_glob": ag_preds_glob,
        "ase_only": ase_only,
        "active_oligos_only": active_oligos_only,
        "elite_lineage_only": elite_lineage_only,
        "min_barcode_counts": min_barcode_counts,
        "target": target,
        "target_label": target_label,
        "n_folds": n_folds,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_tpm_features": X_tpm.shape[1],
        "n_stack_features": X_stack_tr.shape[1],
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n=== Summary ===")
    for name, m in all_metrics.items():
        if name.startswith("_"):
            continue
        rmse_str = f"  RMSE={m['rmse']:.4f}" if "rmse" in m else ""
        print(f"  {name:<32}  r={m['pearson_r']:.4f}  R²={m['r2']:.4f}{rmse_str}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stacked LFC regression: TPM XGBoost OOF predictions + differential MPRA"
    )
    parser.add_argument("--ag-preds-glob", type=str, default="results/all_genes/*.tsv")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--ase-only", action="store_true",
                        help="Restrict to ASE genes (recommended)")
    parser.add_argument("--target", type=str, choices=["lfc", "log10_lfc"], default="lfc")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="CV folds for OOF TPM predictions (default: 5)")
    parser.add_argument("--no-active-oligos-only", action="store_true")
    parser.add_argument("--elite-lineage-only", action="store_true")
    parser.add_argument("--min-barcode-counts", type=int, default=50)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ase_tag = "_ase" if args.ase_only else ""
    output_dir = (args.output_dir or
                  f"results/ag_preds_lfc_stacked_{args.target}{ase_tag}_{run_ts}")

    main(
        ag_preds_glob=args.ag_preds_glob,
        output_dir=output_dir,
        use_gpu=args.gpu,
        ase_only=args.ase_only,
        target=args.target,
        n_folds=args.n_folds,
        active_oligos_only=not args.no_active_oligos_only,
        elite_lineage_only=args.elite_lineage_only,
        min_barcode_counts=args.min_barcode_counts,
    )
