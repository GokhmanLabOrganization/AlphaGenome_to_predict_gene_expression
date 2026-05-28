"""
Binary classification: upregulated (LFC > 0) vs downregulated (LFC < 0) from
paired human/chimp AlphaGenome exon-position embeddings.

Shares the same data pipeline as predict_lfc_paired_embeddings_with_ag_preds.py:
  - Embeddings are per exon position (one row per gene in parquet, exon_embeddings
    is a list of 1536-dim vectors).
  - Genes are matched by gene_symbol across human/chimp parquets.
  - Within each gene, human and chimp positions are paired by index.
  - Gene-level train/test split to prevent label leakage.

Label: 1 = upregulated (LFC > 0), 0 = downregulated (LFC < 0).
Genes with LFC == 0 are excluded (negligible in practice).

Models:
  - LogisticRegression  (linear baseline)
  - XGBoostClassifier
  - FCNet               (BCEWithLogitsLoss, single logit output)

Evaluation per model:
  - Accuracy, AUC-ROC, F1 (position-level)
  - Gene-level: average predicted probability per gene → threshold at 0.5

Outputs saved to --output-dir:
  - metrics.json
  - roc_curve_<model>.png
  - confusion_matrix_<model>.png
  - xgboost_classifier.json
  - fcnet_weights.pt
  - scaler.pkl
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
import config
from predict_lfc_paired_embeddings_with_ag_preds import (
    load_embeddings_for_genes,
    gene_level_metrics,
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc(y_true, y_prob, model_name, output_dir, suffix=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name}{suffix} — ROC curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    safe = (model_name + suffix).replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"roc_curve_{safe}.png"), dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, model_name, output_dir, suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["down (pred)", "up (pred)"])
    ax.set_yticklabels(["down (true)", "up (true)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{model_name}{suffix}")
    plt.tight_layout()
    safe = (model_name + suffix).replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{safe}.png"), dpi=150)
    plt.close()


def evaluate_clf(y_true, y_prob, label, output_dir, suffix=""):
    y_pred = (y_prob >= 0.5).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {label:<40}  acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
    model_name = label.split(" [")[0]
    plot_roc(y_true, y_prob, model_name, output_dir, suffix)
    plot_confusion(y_true, y_pred, model_name, output_dir, suffix)
    return {"accuracy": float(acc), "auc_roc": float(auc), "f1": float(f1)}


# ---------------------------------------------------------------------------
# FCNet for binary classification (single logit output)
# ---------------------------------------------------------------------------

class FCNetBinary(nn.Module):
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
        return self.net(x).squeeze(1)  # raw logit


def train_fcnet_binary(X_train, y_train, X_val, y_val,
                       hidden_dim=256, dropout=0.3,
                       lr=1e-3, epochs=300, batch_size=512,
                       patience=30, device="cpu", pos_weight=None, log_fn=None):
    model     = FCNetBinary(X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

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
                print(f"    Early stop at epoch {epoch + 1}  (best val loss={best_val_loss:.5f})")
                break

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_v)).cpu().numpy()
            try:
                from sklearn.metrics import roc_auc_score as _auc
                val_auc = _auc(y_val, val_probs)
                auc_str = f"  val_AUC={val_auc:.4f}"
            except Exception:
                val_auc = None
                auc_str = ""
            print(f"    [epoch {epoch + 1:4d}]  val_loss={val_loss:.5f}  best={best_val_loss:.5f}{auc_str}",
                  flush=True)
            if log_fn is not None:
                log_payload = {"fcnet/epoch": epoch + 1,
                               "fcnet/val_loss": val_loss,
                               "fcnet/best_val_loss": best_val_loss}
                if val_auc is not None:
                    log_payload["fcnet/val_auc"] = val_auc
                log_fn(log_payload)

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(human_emb_dir: str, chimp_emb_dir: str,
         output_dir: str, lfc_threshold: float, ase_only: bool,
         include_diff: bool, max_positions: int | None, max_genes: int | None,
         n_samples_per_gene: int | None,
         use_gpu: bool, hidden_dim: int, epochs: int,
         use_wandb: bool = False, wandb_project: str = "ag-regulation-clf"):
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
                lfc_threshold=lfc_threshold, ase_only=ase_only,
                include_diff=include_diff, max_positions=max_positions,
                max_genes=max_genes, hidden_dim=hidden_dim, epochs=epochs,
            ),
        )

    # ------------------------------------------------------------------
    # Load LFC labels and convert to binary
    # Thought: exclude LFC == 0 (ambiguous direction). In practice this
    # is essentially never exactly 0 for continuous measurements, but
    # we guard for it anyway.
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

    # Build binary label lookup: 1 = up, 0 = down, skip LFC == 0
    lfc_lookup = {}
    for row in hybrids.iter_rows(named=True):
        lfc = row["ExpLBM_LFC_human_ref"]
        if lfc is None or lfc == 0.0:
            continue
        lfc_lookup[row["Gene"]] = 1 if lfc > 0 else 0

    target_genes = set(lfc_lookup.keys())
    n_up   = sum(v == 1 for v in lfc_lookup.values())
    n_down = sum(v == 0 for v in lfc_lookup.values())
    print(f"  {len(target_genes)} genes: {n_up} up ({n_up/len(target_genes):.1%}), "
          f"{n_down} down ({n_down/len(target_genes):.1%})")

    # Class imbalance ratio — used for XGBoost and FCNet weighting.
    # Thought: if the dataset is balanced, pos_weight ≈ 1 and has no effect.
    pos_weight = n_down / n_up if n_up > 0 else 1.0
    print(f"  pos_weight (down/up) = {pos_weight:.3f}")

    if max_genes is not None:
        target_genes = set(sorted(target_genes)[:max_genes])
        lfc_lookup = {g: v for g, v in lfc_lookup.items() if g in target_genes}
        print(f"  --max-genes {max_genes}: using {len(target_genes)} genes")

    # ------------------------------------------------------------------
    # Load embeddings (same loader as regression script)
    # ------------------------------------------------------------------
    print(f"\nLoading human embeddings...")
    human_embs = load_embeddings_for_genes(human_emb_dir, target_genes, max_positions)
    print(f"  {len(human_embs)} human genes loaded")

    print("Loading chimp embeddings...")
    chimp_embs = load_embeddings_for_genes(chimp_emb_dir, target_genes, max_positions)
    print(f"  {len(chimp_embs)} chimp genes loaded")

    # ------------------------------------------------------------------
    # Build paired dataset (identical logic to regression script)
    # ------------------------------------------------------------------
    print("\nBuilding paired dataset...")
    common_genes = set(human_embs) & set(chimp_embs) & target_genes
    print(f"  {len(common_genes)} genes present in all three sources")

    rows_H, rows_C, labels, gene_ids = [], [], [], []
    gene_n_samples: dict[str, int] = {}
    rng = np.random.default_rng(42)

    for sym in sorted(common_genes):
        H = human_embs[sym]
        C = chimp_embs[sym]
        n = min(len(H), len(C))
        if n == 0:
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

    H_mat = np.vstack(rows_H)
    C_mat = np.vstack(rows_C)
    X = np.hstack([H_mat, C_mat, H_mat - C_mat]) if include_diff else np.hstack([H_mat, C_mat])
    y = np.array(labels, dtype=np.float32)
    gene_ids = np.array(gene_ids)

    n_up_pos   = int(y.sum())
    n_down_pos = int(len(y) - n_up_pos)
    print(f"\n  Feature dim : {X.shape[1]}")
    print(f"  Total samples: {X.shape[0]}  ({len(common_genes)} genes)")
    print(f"  Up positions : {n_up_pos}  Down positions: {n_down_pos}")

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
    print(f"\n{'Model':<40}  {'acc':>7}  {'AUC':>7}  {'F1':>7}")
    print("-" * 62)

    def run_model(name, y_prob_test):
        pos_m = evaluate_clf(y_test, y_prob_test, f"{name} [position]", output_dir, "_position")
        yt_gene, yp_gene = gene_level_metrics(y_test, y_prob_test, gids_test)
        gene_m = evaluate_clf(yt_gene, yp_gene, f"{name} [gene-avg]", output_dir, "_gene_avg")
        if wb is not None:
            key = name.replace(" ", "_")
            wb.log({
                f"{key}/position/accuracy": pos_m["accuracy"],
                f"{key}/position/auc_roc":  pos_m["auc_roc"],
                f"{key}/position/f1":       pos_m["f1"],
                f"{key}/gene/accuracy":     gene_m["accuracy"],
                f"{key}/gene/auc_roc":      gene_m["auc_roc"],
                f"{key}/gene/f1":           gene_m["f1"],
                f"{key}/roc_position":      wb.Image(os.path.join(output_dir, f"roc_curve_{name}_position.png")),
                f"{key}/roc_gene_avg":      wb.Image(os.path.join(output_dir, f"roc_curve_{name}_gene_avg.png")),
                f"{key}/cm_position":       wb.Image(os.path.join(output_dir, f"confusion_matrix_{name}_position.png")),
                f"{key}/cm_gene_avg":       wb.Image(os.path.join(output_dir, f"confusion_matrix_{name}_gene_avg.png")),
            })
        return {"position_level": pos_m, "gene_level": gene_m}

    # ------------------------------------------------------------------
    # 1. Logistic Regression
    # Thought: balanced class_weight handles any imbalance automatically.
    # ------------------------------------------------------------------
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_clf.fit(X_train, y_train)
    all_metrics["LogisticRegression"] = run_model(
        "LogisticRegression", lr_clf.predict_proba(X_test)[:, 1]
    )

    # ------------------------------------------------------------------
    # 2. XGBoost Classifier
    # Thought: scale_pos_weight mirrors BCEWithLogitsLoss pos_weight for
    # gradient-boosted trees. We keep colsample_bytree=0.1 (same as
    # regression) to stay tractable at 3072 features.
    # ------------------------------------------------------------------
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.1,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=30,
        device="cuda" if use_gpu else "cpu",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    all_metrics["XGBoost"] = run_model("XGBoost", xgb_clf.predict_proba(X_test)[:, 1])
    xgb_clf.save_model(os.path.join(output_dir, "xgboost_classifier.json"))

    # ------------------------------------------------------------------
    # 3. FCNet (BCEWithLogitsLoss)
    # Thought: we pass pos_weight so the network learns despite imbalance.
    # sigmoid is NOT part of the network — we apply it during inference to
    # convert logits to probabilities.
    # ------------------------------------------------------------------
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"\nTraining FCNet on device={device} ...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
    fcnet = train_fcnet_binary(
        X_tr, y_tr, X_val, y_val,
        hidden_dim=hidden_dim, epochs=epochs, batch_size=512,
        device=device, pos_weight=float(pos_weight),
        log_fn=wb.log if wb is not None else None,
    )
    fcnet.eval()
    with torch.no_grad():
        logits = fcnet(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    y_prob_fc = 1 / (1 + np.exp(-logits))  # sigmoid
    all_metrics["FCNet"] = run_model("FCNet", y_prob_fc)
    torch.save(fcnet.state_dict(), os.path.join(output_dir, "fcnet_weights.pt"))

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    all_metrics["_config"] = {
        "human_emb_dir": human_emb_dir,
        "chimp_emb_dir": chimp_emb_dir,
        "lfc_threshold": lfc_threshold,
        "ase_only": ase_only,
        "include_diff": include_diff,
        "feature_dim": int(X.shape[1]),
        "n_genes_total": int(len(common_genes)),
        "n_train_positions": int(train_mask.sum()),
        "n_test_positions": int(test_mask.sum()),
        "n_train_genes": int(len(train_genes)),
        "n_test_genes": int(len(test_genes)),
        "pos_weight": float(pos_weight),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "max_positions_per_gene": max_positions,
        "max_genes": max_genes,
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
        print(f"  {name:<22}  pos AUC={pos['auc_roc']:.4f}  gene AUC={gen['auc_roc']:.4f}")

    print(f"\nResults saved to {output_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binary classification (up/down) from paired human/chimp exon embeddings"
    )
    parser.add_argument("--human-emb-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "create_AG_embeddings", "embeddings", "human"))
    parser.add_argument("--chimp-emb-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "create_AG_embeddings", "embeddings", "chimp"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lfc-threshold", type=float, default=0.0)
    parser.add_argument("--ase-only", action="store_true")
    parser.add_argument("--include-diff", action="store_true",
                        help="Append (human_emb - chimp_emb) as extra features")
    parser.add_argument("--max-positions-per-gene", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=None,
                        help="Use only the first N genes (alphabetical) — for smoke tests")
    parser.add_argument("--n-samples-per-gene", type=int, default=None,
                        help="Randomly sample this many paired positions per gene (default: all). "
                             "If a gene has fewer positions, all are used.")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--wandb", action="store_true",
                        help="Log metrics and plots to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="ag-regulation-clf",
                        help="W&B project name (default: ag-regulation-clf)")
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ase_tag = "_ase" if args.ase_only else ""
    output_dir = args.output_dir or f"results/ag_preds_emb_clf{ase_tag}_{run_ts}"

    main(
        human_emb_dir=args.human_emb_dir,
        chimp_emb_dir=args.chimp_emb_dir,
        output_dir=output_dir,
        lfc_threshold=args.lfc_threshold,
        ase_only=args.ase_only,
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
