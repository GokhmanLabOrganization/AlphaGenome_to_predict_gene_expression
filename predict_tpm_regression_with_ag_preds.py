"""
Regression models predicting allele-specific TPM from AlphaGenome scalar gene
predictions + MPRA gene-level features.

Features:
  - HumanGeneExpression  (raw AG predicted human expression)
  - ChimpGeneExpression   (raw AG predicted chimp expression)
  - sign(LFC) * log10(|LFC| + 1)  (signed log10 of predicted LFC)
  - MPRA gene-level features (oligo-aggregated activity, coverage, etc.)

Models:
  - LinearRegression  (OLS baseline)
  - XGBoostRegressor  (gradient-boosted trees)
  - FCNet             (3-layer fully-connected neural network, PyTorch)

Target: ExpLBM_TPM_human_allele  or  ExpLBM_TPM_chimp_allele
        (allele-specific expression from human-chimp hybrids)

Outputs saved to --output-dir:
  - metrics.json
  - scatter_LinearRegression.png
  - scatter_XGBoost.png
  - scatter_FCNet.png
  - scatter_log10_tpm_diagnostic.png
  - feature_importance_xgboost.png
  - feature_names.json
  - scaler.pkl
  - xgboost_regressor.json
  - fcnet_weights.pt
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
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import gaussian_kde, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config

MPRA_DIR = os.path.join(os.path.dirname(__file__), "..", "XGBoost_mpra_to_gene_expression")
sys.path.insert(0, os.path.abspath(MPRA_DIR))
import dataMaker as dm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scatter_plot(y_true, y_pred, model_name, output_dir, ylabel="Predicted TPM"):
    r, _ = pearsonr(y_true, y_pred)
    density = gaussian_kde(np.vstack([y_true, y_pred]))(np.vstack([y_true, y_pred]))
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sc = ax.scatter(y_true, y_pred, c=density, cmap="viridis", alpha=0.6, s=15, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Density")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xlabel("True TPM")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_name}  (Pearson r={r:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_{model_name.replace(' ', '_')}.png"), dpi=150)
    plt.close()


def scatter_log10_tpm_diagnostic(ag_pred_expr, exp_tpm, species, output_path):
    """Sections-7.1/7.2-style: log10 scale, mean-centred, identity line."""
    x = np.log10(np.asarray(exp_tpm,      dtype=float))
    y = np.log10(np.asarray(ag_pred_expr,  dtype=float))
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
    ax.set_xlabel(f"Experimental {species} TPM  log₁₀ [mean-centred]")
    ax.set_ylabel(f"AG predicted {species} expression  log₁₀ [mean-centred]")
    ax.set_title(f"AG Predicted vs Experimental TPM — {species}  (log₁₀, mean-centred)")
    ax.text(
        0.05, 0.95,
        f"N = {len(x)}\nr = {r:.3f}\nR² = {r2:.3f}\np = {p:.2e}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved {output_path}")


def evaluate(y_true, y_pred, model_name):
    r, pval = pearsonr(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    rmse    = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {model_name:<25}  Pearson r={r:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"pearson_r": float(r), "pearson_pval": float(pval),
            "r2": float(r2), "rmse": float(rmse)}


# ---------------------------------------------------------------------------
# FCNet
# ---------------------------------------------------------------------------

class FCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_fcnet(X_train, y_train, X_val, y_val,
                hidden_dim=256, dropout=0.3,
                lr=1e-3, epochs=300, batch_size=32,
                patience=30, device="cpu"):
    model     = FCNet(X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn   = nn.MSELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch + 1}  (best val MSE={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# MPRA gene features
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

    # No ASE-type filter — keep all genes with sufficient oligos
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

def main(ag_preds_glob: str, output_dir: str, species: str, target: str,
         use_gpu: bool, ase_only: bool, hidden_dim: int = 256, epochs: int = 300):
    os.makedirs(output_dir, exist_ok=True)

    tpm_col = f"ExpLBM_TPM_{species}_allele"
    ag_expr_col = "HumanGeneExpression" if species == "human" else "ChimpGeneExpression"

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
    ).select(["Gene", tpm_col, "ExpLBM_gene_ase_type"])

    joined = ag.join(hybrids, on="Gene", how="inner")
    print(f"  {joined.height} genes after join with hybrid labels")

    if ase_only:
        joined = joined.filter(pl.col("ExpLBM_gene_ase_type") == "ASE")
        print(f"  {joined.height} genes after --ase-only filter")

    # ------------------------------------------------------------------
    # Diagnostic scatter: AG predicted expression vs experimental TPM
    # ------------------------------------------------------------------
    print("Generating diagnostic TPM scatter...")
    scatter_log10_tpm_diagnostic(
        ag_pred_expr=joined[ag_expr_col].to_numpy(),
        exp_tpm=joined[tpm_col].to_numpy(),
        species=species,
        output_path=os.path.join(output_dir, "scatter_log10_tpm_diagnostic.png"),
    )

    # ------------------------------------------------------------------
    # Build MPRA gene features
    # ------------------------------------------------------------------
    print("Building MPRA gene features...")
    mpra_gene = build_mpra_gene_features()
    print(f"  {mpra_gene.height} genes, {mpra_gene.width} columns")

    mpra_feat_cols = [c for c in mpra_gene.columns if c not in ("Gene", "chromosome")]
    mpra_lookup = {
        row["Gene"]: [float(row[c]) if row[c] is not None else 0.0 for c in mpra_feat_cols]
        for row in mpra_gene.iter_rows(named=True)
    }

    # ------------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------------
    ag_feat_cols = ["HumanGeneExpression", "ChimpGeneExpression", "ag_signed_log10_lfc"]
    all_feat_names = ag_feat_cols + mpra_feat_cols

    rows_ag   = []
    rows_mpra = []
    y_list    = []
    genes_kept = []

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
            y_val = float(row[tpm_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(y_val) or y_val < 0:
            continue
        rows_ag.append(ag_feats)
        rows_mpra.append(mpra_lookup[gene])
        y_list.append(y_val)
        genes_kept.append(gene)

    if not rows_ag:
        raise ValueError(
            "No samples remain after all filters. "
            "Check that gene names match between AG predictions and MPRA data."
        )

    ag_X   = np.array(rows_ag,   dtype=np.float32)
    mpra_X = np.nan_to_num(np.array(rows_mpra, dtype=np.float32), nan=0.0)
    X      = np.hstack([ag_X, mpra_X])
    y      = np.array(y_list, dtype=np.float32)

    if target == "log10_tpm":
        y = np.log10(y + 1).astype(np.float32)
        target_label = "log10(TPM + 1)"
    else:
        target_label = "TPM (raw)"

    print(f"\n  Final dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  ({len(ag_feat_cols)} AG + {len(mpra_feat_cols)} MPRA)")
    print(f"  Target: {target_label}  [{species} allele]")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]  mean={y.mean():.3f}")

    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(all_feat_names, f, indent=2)

    # ------------------------------------------------------------------
    # Train / test split + scale
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    all_metrics = {}
    ylabel = f"Predicted {target_label} [{species}]"
    print(f"\n{'Model':<25}  {'Pearson r':>10}  {'R²':>8}  {'RMSE':>8}")
    print("-" * 57)

    # ------------------------------------------------------------------
    # 1. Linear Regression
    # ------------------------------------------------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    all_metrics["LinearRegression"] = evaluate(y_test, y_pred, "LinearRegression")
    scatter_plot(y_test, y_pred, "LinearRegression", output_dir, ylabel=ylabel)

    # ------------------------------------------------------------------
    # 2. XGBoost Regressor
    # ------------------------------------------------------------------
    xgb_reg = xgb.XGBRegressor(
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
    xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    y_pred = xgb_reg.predict(X_test)
    all_metrics["XGBoost"] = evaluate(y_test, y_pred, "XGBoost")
    scatter_plot(y_test, y_pred, "XGBoost", output_dir, ylabel=ylabel)
    xgb_reg.save_model(os.path.join(output_dir, "xgboost_regressor.json"))

    importance = xgb_reg.get_booster().get_score(importance_type="gain")
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:50]
        fnames, fvals = zip(*sorted_imp)
        fig, ax = plt.subplots(figsize=(7, max(4, len(fnames) * 0.25)))
        ax.barh(range(len(fnames)), fvals[::-1])
        ax.set_yticks(range(len(fnames)))
        ax.set_yticklabels(list(fnames[::-1]), fontsize=7)
        ax.set_xlabel("Gain")
        ax.set_title("XGBoost Top-50 Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance_xgboost.png"), dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 3. FCNet
    # ------------------------------------------------------------------
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"\nTraining FCNet on device={device} ...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=0
    )
    fcnet = train_fcnet(
        X_tr, y_tr, X_val, y_val,
        hidden_dim=hidden_dim, epochs=epochs, device=device,
    )
    fcnet.eval()
    with torch.no_grad():
        y_pred = fcnet(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        ).cpu().numpy()
    all_metrics["FCNet"] = evaluate(y_test, y_pred, "FCNet")
    scatter_plot(y_test, y_pred, "FCNet", output_dir, ylabel=ylabel)
    torch.save(fcnet.state_dict(), os.path.join(output_dir, "fcnet_weights.pt"))

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics["_config"] = {
        "ag_preds_glob": ag_preds_glob,
        "species": species,
        "target": target,
        "target_label": target_label,
        "tpm_col": tpm_col,
        "ase_only": ase_only,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_ag_features": len(ag_feat_cols),
        "n_mpra_features": len(mpra_feat_cols),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
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
        description="TPM regression from AlphaGenome scalar predictions + MPRA features"
    )
    parser.add_argument(
        "--ag-preds-glob",
        type=str,
        default="results/all_genes/*.tsv",
        help="Glob for AG prediction TSV files (default: results/all_genes/*.tsv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["human", "chimp"],
        default="human",
        help="Allele to predict: 'human' = ExpLBM_TPM_human_allele (default), 'chimp' = ExpLBM_TPM_chimp_allele",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["tpm", "log10_tpm"],
        default="log10_tpm",
        help="Target transform: 'tpm' = raw TPM, 'log10_tpm' = log10(TPM+1) (default)",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--ase-only",
        action="store_true",
        help="Restrict to genes classified as ASE in ExpLBM",
    )
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="FCNet hidden layer size (default: 256)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Max FCNet training epochs (default: 300)")
    args = parser.parse_args()

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    ase_tag = "_ase" if args.ase_only else ""
    output_dir = args.output_dir or (
        f"results/ag_preds_tpm_{args.species}_{args.target}{ase_tag}_{run_ts}"
    )

    main(
        ag_preds_glob=args.ag_preds_glob,
        output_dir=output_dir,
        species=args.species,
        target=args.target,
        use_gpu=args.gpu,
        ase_only=args.ase_only,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )
