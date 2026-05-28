"""
Regression models predicting LFC from paired human/chimp AlphaGenome exon-position embeddings.

Each AlphaGenome parquet row contains one gene's full exon_embeddings array of shape
(n_exon_positions, 1536). For each ortholog pair (matched by gene_symbol), we zip
the human and chimp embedding arrays position-by-position up to min(n_human, n_chimp).
Every resulting sample's label is the gene's ExpLBM_LFC_human_ref from hybrid cells.
Many samples share the same label (one per exon position within a gene).

Features per sample: [human_emb (1536), chimp_emb (1536)] = 3072 dims.
Optional --include-diff adds (human - chimp) for 4608 dims total.

Models:
  - LinearRegression  (OLS baseline)
  - XGBoostRegressor
  - FCNet             (3-layer fully-connected neural network, PyTorch)

Train/test split is gene-level to prevent label leakage.
Evaluation reports both per-position metrics and gene-level metrics
(predictions averaged per gene, then correlated with gene LFC).

Outputs saved to --output-dir:
  - metrics.json
  - scatter_*.png
  - xgboost_regressor.json
  - fcnet_weights.pt
  - scaler.pkl
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import gaussian_kde, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
import config


# ---------------------------------------------------------------------------
# Helpers (reused from predict_lfc_regression_with_ag_preds.py)
# ---------------------------------------------------------------------------

def scatter_plot(y_true, y_pred, model_name, output_dir, suffix=""):
    r, _ = pearsonr(y_true, y_pred)
    density = gaussian_kde(np.vstack([y_true, y_pred]))(np.vstack([y_true, y_pred]))
    fig, ax = plt.subplots(figsize=(5.8, 5))
    sc = ax.scatter(y_true, y_pred, c=density, cmap="viridis", alpha=0.6, s=15, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Density")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xscale("symlog", base=2, linthresh=0.1)
    ax.set_yscale("symlog", base=2, linthresh=0.1)
    ax.set_xlabel("True LFC")
    ax.set_ylabel("Predicted LFC")
    ax.set_title(f"{model_name}{suffix}  (Pearson r={r:.3f})")
    plt.tight_layout()
    safe = model_name.replace(" ", "_") + suffix.replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"scatter_{safe}.png"), dpi=150)
    plt.close()


def evaluate(y_true, y_pred, label):
    r, pval = pearsonr(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    rmse    = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"  {label:<35}  Pearson r={r:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"pearson_r": float(r), "pearson_pval": float(pval),
            "r2": float(r2), "rmse": float(rmse)}


# ---------------------------------------------------------------------------
# FCNet (reused)
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
                lr=1e-3, epochs=300, batch_size=512,
                patience=30, device="cpu", log_fn=None):
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

        if (epoch + 1) % 10 == 0:
            print(f"    [epoch {epoch + 1:4d}]  val_MSE={val_loss:.5f}  best={best_val_loss:.5f}",
                  flush=True)
            if log_fn is not None:
                log_fn({"fcnet/epoch": epoch + 1,
                        "fcnet/val_mse": val_loss,
                        "fcnet/best_val_mse": best_val_loss})

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def _job_number(fpath):
    m = re.search(r"job_(\d+)", fpath)
    return int(m.group(1)) if m else 0


def build_gene_index(emb_dir: str) -> dict:
    """Build {gene_symbol: (file_path, row_group_idx, status)} from parquet footers only.

    Each file has 1 row per row group, and row group statistics store min==max==gene_symbol,
    so we can read the entire index from file footers without touching any column data.
    """
    files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")), key=_job_number)
    index = {}
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        meta = pf.metadata
        # find column indices for gene_symbol and status once per file
        schema = pf.schema_arrow
        col_names = schema.names
        gs_col  = col_names.index("gene_symbol")
        sts_col = col_names.index("status")
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            sym = rg.column(gs_col).statistics.min
            st  = rg.column(sts_col).statistics.min
            if sym is not None:
                index[sym] = (fpath, rg_idx, st)
    return index


def load_embeddings_for_genes(emb_dir: str, target_genes: set,
                               max_positions: int | None = None) -> dict:
    """Load exon embeddings only for genes in target_genes.

    Step 1: build a gene→(file, row_group) index from parquet footers only
            (no column data reads — uses min/max statistics stored in each row group).
    Step 2: for each target gene, read only its single row group.

    Returns: dict gene_symbol -> np.float32 array (n_positions, emb_dim).
    """
    print(f"  Building index from {emb_dir} footers...", end=" ", flush=True)
    index = build_gene_index(emb_dir)
    print(f"{len(index)} genes indexed")

    # Group target genes by file to minimise file opens
    file_to_rgs: dict[str, list[tuple[str, int]]] = {}
    for sym in target_genes:
        if sym not in index:
            continue
        fpath, rg_idx, st = index[sym]
        if st != "ok":
            continue
        file_to_rgs.setdefault(fpath, []).append((sym, rg_idx))

    result = {}
    for fpath, gene_rg_list in file_to_rgs.items():
        pf = pq.ParquetFile(fpath)
        for sym, rg_idx in gene_rg_list:
            row = pf.read_row_group(rg_idx, columns=["exon_embeddings"]).to_pydict()
            emb = row["exon_embeddings"][0]
            if not emb:
                continue
            arr = np.array(emb, dtype=np.float32)
            if max_positions is not None:
                arr = arr[:max_positions]
            result[sym] = arr
    return result


# ---------------------------------------------------------------------------
# Gene-level evaluation helper
# ---------------------------------------------------------------------------

def gene_level_metrics(y_true_pos, y_pred_pos, gene_ids):
    """Average per-position predictions per gene, then evaluate."""
    unique_genes = np.unique(gene_ids)
    y_gene_true = np.array([y_true_pos[gene_ids == g].mean() for g in unique_genes])
    y_gene_pred = np.array([y_pred_pos[gene_ids == g].mean() for g in unique_genes])
    return y_gene_true, y_gene_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(human_emb_dir: str, chimp_emb_dir: str, ag_preds_glob: str,
         output_dir: str, lfc_threshold: float, ase_only: bool,
         target: str, include_diff: bool, max_positions: int | None,
         max_genes: int | None, n_samples_per_gene: int | None,
         use_gpu: bool, hidden_dim: int, epochs: int,
         use_wandb: bool = False, wandb_project: str = "ag-lfc-regression"):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Wandb init (lazy import — script works without wandb installed)
    # ------------------------------------------------------------------
    wb = None
    if use_wandb:
        import wandb as _wandb
        wb = _wandb
        wb.init(
            project=wandb_project,
            config=dict(
                lfc_threshold=lfc_threshold, ase_only=ase_only, target=target,
                include_diff=include_diff, max_positions=max_positions,
                max_genes=max_genes, hidden_dim=hidden_dim, epochs=epochs,
            ),
        )

    # ------------------------------------------------------------------
    # Load LFC labels from hybrid data
    # ------------------------------------------------------------------
    print("Loading hybrid LFC labels...")
    hybrids = pl.read_csv(
        config.HUMAN_CHIMP_HYBRIDS_DATA_PATH_WEXAC, separator="\t"
    ).select(["Gene", "ExpLBM_LFC_human_ref", "ExpLBM_gene_ase_type"])

    if ase_only:
        hybrids = hybrids.filter(pl.col("ExpLBM_gene_ase_type") == "ASE")
        print(f"  ASE-only filter: {hybrids.height} genes")

    if lfc_threshold > 0:
        hybrids = hybrids.filter(pl.col("ExpLBM_LFC_human_ref").abs() >= lfc_threshold)
        print(f"  |LFC| >= {lfc_threshold} filter: {hybrids.height} genes")

    lfc_lookup = {
        row["Gene"]: float(row["ExpLBM_LFC_human_ref"])
        for row in hybrids.iter_rows(named=True)
        if row["ExpLBM_LFC_human_ref"] is not None
    }
    target_genes = set(lfc_lookup.keys())
    if max_genes is not None:
        target_genes = set(sorted(target_genes)[:max_genes])
        lfc_lookup = {g: v for g, v in lfc_lookup.items() if g in target_genes}
        print(f"  --max-genes {max_genes}: using {len(target_genes)} genes")
    print(f"  {len(target_genes)} genes with LFC labels")

    # ------------------------------------------------------------------
    # Load embeddings (only for genes with labels)
    # ------------------------------------------------------------------
    print(f"\nLoading human embeddings for {len(target_genes)} target genes...")
    human_embs = load_embeddings_for_genes(human_emb_dir, target_genes, max_positions)
    print(f"  Loaded {len(human_embs)} human genes with embeddings")

    print(f"Loading chimp embeddings for {len(target_genes)} target genes...")
    chimp_embs = load_embeddings_for_genes(chimp_emb_dir, target_genes, max_positions)
    print(f"  Loaded {len(chimp_embs)} chimp genes with embeddings")

    # ------------------------------------------------------------------
    # Build paired dataset gene-by-gene
    # ------------------------------------------------------------------
    print("\nBuilding paired dataset...")
    common_genes = set(human_embs) & set(chimp_embs) & target_genes
    print(f"  {len(common_genes)} genes present in human, chimp, and label set")

    rows_H, rows_C, labels, gene_ids = [], [], [], []
    gene_n_samples: dict[str, int] = {}
    skipped_no_overlap = 0
    rng = np.random.default_rng(42)

    for sym in sorted(common_genes):
        H = human_embs[sym]   # (n_h, emb_dim)
        C = chimp_embs[sym]   # (n_c, emb_dim)
        n = min(len(H), len(C))
        if n == 0:
            skipped_no_overlap += 1
            continue
        k = min(n, n_samples_per_gene) if n_samples_per_gene is not None else n
        if k < n:
            idx = np.sort(rng.choice(n, size=k, replace=False))
        else:
            idx = np.arange(n)
        rows_H.append(H[idx])
        rows_C.append(C[idx])
        labels.extend([lfc_lookup[sym]] * k)
        gene_ids.extend([sym] * k)
        gene_n_samples[sym] = k

    if skipped_no_overlap:
        print(f"  Skipped {skipped_no_overlap} genes with zero overlap positions")

    if not rows_H:
        raise ValueError("No paired samples built. Check that gene symbols match between embeddings and labels.")

    H_mat = np.vstack(rows_H)   # (N, emb_dim)
    C_mat = np.vstack(rows_C)   # (N, emb_dim)

    if include_diff:
        X = np.hstack([H_mat, C_mat, H_mat - C_mat])
    else:
        X = np.hstack([H_mat, C_mat])

    y = np.array(labels, dtype=np.float32)
    gene_ids = np.array(gene_ids)

    if target == "log10_lfc":
        y = (np.sign(y) * np.log10(np.abs(y) + 1)).astype(np.float32)
        target_label = "sign(LFC)×log10(|LFC|+1)"
    else:
        target_label = "LFC (raw)"

    emb_dim = H_mat.shape[1]
    print(f"\n  Embedding dim  : {emb_dim}")
    print(f"  Feature dim    : {X.shape[1]}  ({'H+C+diff' if include_diff else 'H+C'})")
    print(f"  Total samples  : {X.shape[0]}  ({len(common_genes)} genes)")
    print(f"  Target         : {target_label}")
    print(f"  y range        : [{y.min():.3f}, {y.max():.3f}]  mean={y.mean():.3f}")

    # ------------------------------------------------------------------
    # Gene-level train/test split
    # ------------------------------------------------------------------
    unique_genes = np.unique(gene_ids)
    train_genes, test_genes = train_test_split(unique_genes, test_size=0.2, random_state=42)
    train_mask = np.isin(gene_ids, train_genes)
    test_mask  = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    gids_test = gene_ids[test_mask]

    print(f"\n  Train: {train_mask.sum()} positions ({len(train_genes)} genes)")
    print(f"  Test : {test_mask.sum()} positions ({len(test_genes)} genes)")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    all_metrics = {}
    print(f"\n{'Model':<35}  {'Pearson r':>10}  {'R²':>8}  {'RMSE':>8}")
    print("-" * 68)

    def run_model(name, y_pred_test):
        pos_m = evaluate(y_test, y_pred_test, f"{name} [position]")
        yt_gene, yp_gene = gene_level_metrics(y_test, y_pred_test, gids_test)
        gene_m = evaluate(yt_gene, yp_gene, f"{name} [gene-avg]")
        scatter_plot(y_test, y_pred_test, name, output_dir, "_position")
        scatter_plot(yt_gene, yp_gene, name, output_dir, "_gene_avg")
        if wb is not None:
            key = name.replace(" ", "_")
            wb.log({
                f"{key}/position/pearson_r": pos_m["pearson_r"],
                f"{key}/position/r2":        pos_m["r2"],
                f"{key}/position/rmse":      pos_m["rmse"],
                f"{key}/gene/pearson_r":     gene_m["pearson_r"],
                f"{key}/gene/r2":            gene_m["r2"],
                f"{key}/gene/rmse":          gene_m["rmse"],
                f"{key}/scatter_position":   wb.Image(os.path.join(output_dir, f"scatter_{name}_position.png")),
                f"{key}/scatter_gene_avg":   wb.Image(os.path.join(output_dir, f"scatter_{name}_gene_avg.png")),
            })
        return {"position_level": pos_m, "gene_level": gene_m}

    # ------------------------------------------------------------------
    # 1. Linear Regression
    # ------------------------------------------------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    all_metrics["LinearRegression"] = run_model("LinearRegression", lr_model.predict(X_test))

    # ------------------------------------------------------------------
    # 2. XGBoost Regressor
    # ------------------------------------------------------------------
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.1,   # 10% of 3072 = ~307 features/tree
        eval_metric="rmse",
        early_stopping_rounds=30,
        device="cuda" if use_gpu else "cpu",
        random_state=42,
        n_jobs=-1,
    )
    xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    all_metrics["XGBoost"] = run_model("XGBoost", xgb_reg.predict(X_test))
    xgb_reg.save_model(os.path.join(output_dir, "xgboost_regressor.json"))

    # ------------------------------------------------------------------
    # 3. FCNet
    # ------------------------------------------------------------------
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"\nTraining FCNet on device={device} ...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
    fcnet = train_fcnet(
        X_tr, y_tr, X_val, y_val,
        hidden_dim=hidden_dim, epochs=epochs, batch_size=512, device=device,
        log_fn=wb.log if wb is not None else None,
    )
    fcnet.eval()
    with torch.no_grad():
        y_pred_fc = fcnet(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        ).cpu().numpy()
    all_metrics["FCNet"] = run_model("FCNet", y_pred_fc)
    torch.save(fcnet.state_dict(), os.path.join(output_dir, "fcnet_weights.pt"))

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics["_config"] = {
        "human_emb_dir": human_emb_dir,
        "chimp_emb_dir": chimp_emb_dir,
        "ag_preds_glob": ag_preds_glob,
        "lfc_threshold": lfc_threshold,
        "ase_only": ase_only,
        "target": target,
        "target_label": target_label,
        "include_diff": include_diff,
        "emb_dim": int(emb_dim),
        "feature_dim": int(X.shape[1]),
        "n_genes_total": int(len(common_genes)),
        "n_train_positions": int(train_mask.sum()),
        "n_test_positions": int(test_mask.sum()),
        "n_train_genes": int(len(train_genes)),
        "n_test_genes": int(len(test_genes)),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "max_positions_per_gene": max_positions,
        "n_samples_per_gene": n_samples_per_gene,
    }
    all_metrics["_gene_n_samples"] = gene_n_samples
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n=== Summary ===")
    for name, m in all_metrics.items():
        if name.startswith("_"):
            continue
        pos = m["position_level"]
        gen = m["gene_level"]
        print(f"  {name:<22}  pos r={pos['pearson_r']:.4f}  gene r={gen['pearson_r']:.4f}")

    print(f"\nResults saved to {output_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LFC regression from paired human/chimp AlphaGenome exon-position embeddings"
    )
    parser.add_argument("--human-emb-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "create_AG_embeddings", "embeddings", "human"),
                        help="Directory of human embedding parquet files")
    parser.add_argument("--chimp-emb-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "create_AG_embeddings", "embeddings", "chimp"),
                        help="Directory of chimp embedding parquet files")
    parser.add_argument("--ag-preds-glob", type=str,
                        default="results/all_genes/*.tsv",
                        help="Glob for AG prediction TSVs (unused here but kept for consistency)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lfc-threshold", type=float, default=0.0,
                        help="Minimum |ExpLBM_LFC_human_ref| to include a gene")
    parser.add_argument("--ase-only", action="store_true",
                        help="Restrict to genes classified as ASE in ExpLBM")
    parser.add_argument("--target", type=str, choices=["lfc", "log10_lfc"], default="lfc",
                        help="Training target: 'lfc' = raw, 'log10_lfc' = sign*log10(|lfc|+1)")
    parser.add_argument("--include-diff", action="store_true",
                        help="Append (human_emb - chimp_emb) as extra features")
    parser.add_argument("--max-positions-per-gene", type=int, default=None,
                        help="Truncate each gene to at most N exon positions (default: use all)")
    parser.add_argument("--max-genes", type=int, default=None,
                        help="Use only the first N genes (alphabetical) — useful for smoke tests")
    parser.add_argument("--n-samples-per-gene", type=int, default=None,
                        help="Randomly sample this many paired positions per gene (default: all). "
                             "If a gene has fewer positions, all are used.")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--wandb", action="store_true",
                        help="Log metrics and plots to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="ag-lfc-regression",
                        help="W&B project name (default: ag-lfc-regression)")
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ase_tag = "_ase" if args.ase_only else ""
    output_dir = args.output_dir or f"results/ag_preds_emb_lfc_{args.target}{ase_tag}_{run_ts}"

    main(
        human_emb_dir=args.human_emb_dir,
        chimp_emb_dir=args.chimp_emb_dir,
        ag_preds_glob=args.ag_preds_glob,
        output_dir=output_dir,
        lfc_threshold=args.lfc_threshold,
        ase_only=args.ase_only,
        target=args.target,
        include_diff=args.include_diff,
        max_positions=args.max_positions_per_gene,
        max_genes=args.max_genes,
        n_samples_per_gene=args.n_samples_per_gene,
        use_gpu=args.gpu,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
