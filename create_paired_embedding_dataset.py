"""
One-time preprocessing script: stream human and chimp AlphaGenome exon-position
embeddings, pair them by gene, subsample positions, and save the assembled
matrices to disk for fast reuse by training scripts.

Outputs (in --output-dir):
  H_mat.npy      float32 (N, emb_dim)  — human position embeddings
  C_mat.npy      float32 (N, emb_dim)  — chimp position embeddings
  gene_ids.npy   object  (N,)          — gene symbol for each row
  lfc.npy        float32 (N,)          — ExpLBM_LFC_human_ref per row
  ase_type.npy   object  (N,)          — ExpLBM_gene_ase_type per row
  metadata.json                        — creation parameters

Training scripts load these files directly instead of re-reading parquet.
All filtering (lfc_threshold, ase_only, etc.) happens at training time.

Usage:
  python create_paired_embedding_dataset.py \\
      --human-emb-dir ../create_AG_embeddings/embeddings/human \\
      --chimp-emb-dir ../create_AG_embeddings/embeddings/chimp \\
      --n-samples-per-gene 200 \\
      --output-dir data/paired_embeddings_n200
"""

import argparse
import json
import os
import re
import sys
import glob
from datetime import datetime

import numpy as np
import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
import config


# ---------------------------------------------------------------------------
# Parquet index helpers (same as in predict_lfc_paired_embeddings_with_ag_preds.py)
# ---------------------------------------------------------------------------

def _job_number(fpath):
    m = re.search(r"job_(\d+)", fpath)
    return int(m.group(1)) if m else 0


def build_gene_index(emb_dir: str) -> dict:
    """Build {gene_symbol: (file_path, row_group_idx, status)} from parquet footers."""
    files = sorted(glob.glob(os.path.join(emb_dir, "*.parquet")), key=_job_number)
    index = {}
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        meta = pf.metadata
        schema = pf.schema_arrow
        col_names = schema.names
        gs_col = col_names.index("gene_symbol")
        sts_col = col_names.index("status")
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            sym = rg.column(gs_col).statistics.min
            st = rg.column(sts_col).statistics.min
            if sym is not None:
                index[sym] = (fpath, rg_idx, st)
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(human_emb_dir: str, chimp_emb_dir: str, output_dir: str,
         n_samples_per_gene: int | None, max_positions: int | None,
         random_seed: int):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load hybrid labels (LFC + ASE type for all genes)
    # ------------------------------------------------------------------
    print("Loading hybrid LFC labels...")
    hybrids = pl.read_csv(
        config.HUMAN_CHIMP_HYBRIDS_DATA_PATH_WEXAC, separator="\t"
    ).select(["Gene", "ExpLBM_LFC_human_ref", "ExpLBM_gene_ase_type"])

    lfc_lookup: dict[str, float] = {}
    ase_lookup: dict[str, str] = {}
    for row in hybrids.iter_rows(named=True):
        if row["ExpLBM_LFC_human_ref"] is not None:
            lfc_lookup[row["Gene"]] = float(row["ExpLBM_LFC_human_ref"])
            ase_lookup[row["Gene"]] = str(row["ExpLBM_gene_ase_type"] or "")

    target_genes = set(lfc_lookup.keys())
    print(f"  {len(target_genes)} genes with LFC labels")

    # ------------------------------------------------------------------
    # Build embedding indices (reads only parquet footers — fast)
    # ------------------------------------------------------------------
    print(f"\nBuilding human embedding index from {human_emb_dir} ...")
    human_index = build_gene_index(human_emb_dir)
    print(f"  {len(human_index)} genes indexed")

    print(f"Building chimp embedding index from {chimp_emb_dir} ...")
    chimp_index = build_gene_index(chimp_emb_dir)
    print(f"  {len(chimp_index)} genes indexed")

    # Common genes: have labels + embeddings in both species with status "ok"
    common_genes = {
        sym for sym in target_genes
        if sym in human_index and human_index[sym][2] == "ok"
        and sym in chimp_index and chimp_index[sym][2] == "ok"
    }
    print(f"  {len(common_genes)} genes present in labels, human, and chimp")

    # ------------------------------------------------------------------
    # Stream embeddings one gene at a time and build paired rows
    # ------------------------------------------------------------------
    print(f"\nStreaming paired embeddings "
          f"(n_samples_per_gene={n_samples_per_gene}, max_positions={max_positions}) ...")

    rng = np.random.default_rng(random_seed)

    # Group by (human_file, chimp_file) to open each file pair only once
    file_pair_to_genes: dict[tuple, list] = {}
    for sym in common_genes:
        key = (human_index[sym][0], chimp_index[sym][0])
        file_pair_to_genes.setdefault(key, []).append(sym)

    rows_H: list = []
    rows_C: list = []
    gene_id_list: list = []
    lfc_list: list = []
    ase_list: list = []
    n_skipped = 0
    n_processed = 0

    for (h_fpath, c_fpath), syms in sorted(file_pair_to_genes.items()):
        h_pf = pq.ParquetFile(h_fpath)
        c_pf = pq.ParquetFile(c_fpath)
        for sym in sorted(syms):
            _, rg_h, _ = human_index[sym]
            _, rg_c, _ = chimp_index[sym]

            h_emb = h_pf.read_row_group(rg_h, columns=["exon_embeddings"]).to_pydict()["exon_embeddings"][0]
            if not h_emb:
                n_skipped += 1
                continue
            H = np.array(h_emb, dtype=np.float32)
            if max_positions is not None:
                H = H[:max_positions]

            c_emb = c_pf.read_row_group(rg_c, columns=["exon_embeddings"]).to_pydict()["exon_embeddings"][0]
            if not c_emb:
                n_skipped += 1
                continue
            C = np.array(c_emb, dtype=np.float32)
            if max_positions is not None:
                C = C[:max_positions]

            n = min(len(H), len(C))
            if n == 0:
                n_skipped += 1
                continue

            k = min(n, n_samples_per_gene) if n_samples_per_gene is not None else n
            idx = np.sort(rng.choice(n, size=k, replace=False)) if k < n else np.arange(n)

            rows_H.append(H[idx])
            rows_C.append(C[idx])
            gene_id_list.extend([sym] * k)
            lfc_list.extend([lfc_lookup[sym]] * k)
            ase_list.extend([ase_lookup[sym]] * k)
            n_processed += 1

            if n_processed % 1000 == 0:
                print(f"  {n_processed}/{len(common_genes)} genes processed ...", flush=True)

    if n_skipped:
        print(f"  Skipped {n_skipped} genes (empty embeddings or zero overlap)")
    print(f"  {n_processed} genes processed")

    if not rows_H:
        raise ValueError("No paired samples built — check that gene symbols match across embeddings and labels.")

    # ------------------------------------------------------------------
    # Assemble and save
    # ------------------------------------------------------------------
    print("\nAssembling matrices ...")
    H_mat = np.vstack(rows_H)
    del rows_H
    C_mat = np.vstack(rows_C)
    del rows_C

    gene_ids = np.array(gene_id_list, dtype=object)
    lfc_arr  = np.array(lfc_list, dtype=np.float32)
    ase_arr  = np.array(ase_list, dtype=object)

    print(f"  H_mat shape : {H_mat.shape}  ({H_mat.nbytes / 1e9:.1f} GB)")
    print(f"  C_mat shape : {C_mat.shape}  ({C_mat.nbytes / 1e9:.1f} GB)")
    print(f"  Total rows  : {len(gene_ids)}  ({len(np.unique(gene_ids))} unique genes)")

    print(f"\nSaving to {output_dir} ...")
    np.save(os.path.join(output_dir, "H_mat.npy"), H_mat)
    np.save(os.path.join(output_dir, "C_mat.npy"), C_mat)
    np.save(os.path.join(output_dir, "gene_ids.npy"), gene_ids)
    np.save(os.path.join(output_dir, "lfc.npy"), lfc_arr)
    np.save(os.path.join(output_dir, "ase_type.npy"), ase_arr)

    metadata = {
        "created": datetime.now().isoformat(),
        "human_emb_dir": human_emb_dir,
        "chimp_emb_dir": chimp_emb_dir,
        "n_samples_per_gene": n_samples_per_gene,
        "max_positions": max_positions,
        "random_seed": random_seed,
        "n_genes": int(len(np.unique(gene_ids))),
        "n_rows": int(len(gene_ids)),
        "emb_dim": int(H_mat.shape[1]),
        "H_mat_gb": round(H_mat.nbytes / 1e9, 2),
        "C_mat_gb": round(C_mat.nbytes / 1e9, 2),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"  Files saved to: {output_dir}")
    print(f"  Load with: np.load('{output_dir}/H_mat.npy', mmap_mode='r')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-time creation of paired human/chimp embedding dataset from parquet files."
    )
    parser.add_argument("--human-emb-dir", required=True,
                        help="Directory containing human AlphaGenome embedding parquet files")
    parser.add_argument("--chimp-emb-dir", required=True,
                        help="Directory containing chimp AlphaGenome embedding parquet files")
    parser.add_argument("--output-dir", default="data/paired_embeddings",
                        help="Directory to write output .npy files (default: data/paired_embeddings)")
    parser.add_argument("--n-samples-per-gene", type=int, default=None,
                        help="Randomly sample this many positions per gene (default: all). "
                             "If a gene has fewer positions, all are used.")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Truncate each gene to at most N positions before sampling "
                             "(default: use all positions)")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    main(
        human_emb_dir=args.human_emb_dir,
        chimp_emb_dir=args.chimp_emb_dir,
        output_dir=args.output_dir,
        n_samples_per_gene=args.n_samples_per_gene,
        max_positions=args.max_positions,
        random_seed=args.random_seed,
    )
