#!/usr/bin/env python3
"""
evaluate_gene.py
================
Evaluate the AlphaGenome human/chimp expression prediction for a single gene.

QUICK START
-----------
  python evaluate_gene.py --gene SAMD11
  python evaluate_gene.py --gene NOC2L   --save_fig results/NOC2L_eval.png
  python evaluate_gene.py --gene KLHL17  --results_dir /path/to/all_genes --save_fig out.png

WHAT IT DOES
------------
  1. Looks up the gene in pre-computed results (results/all_genes/).
  2. If not found, runs the AlphaGenome prediction live (needs API key in config.py).
  3. Prints a text report with: predicted LFC, expression levels, genome-wide
     percentile rank, and a comparison against multi-cell-type hybrid ASE data.
  4. Optionally saves a 3-panel figure (LFC histogram | expression bars | cell-type ASE).

ARGUMENTS
---------
  --gene         Gene symbol, e.g. SAMD11  (required)
  --results_dir  Folder with lfc_df_*.tsv chunks  (default: results/all_genes)
  --save_fig     Save figure to this path, e.g. results/SAMD11_eval.png
                 A matching .txt report is also saved automatically.
  --no_live      Fail instead of running a live prediction when gene is missing.

OUTPUTS
-------
  Text report printed to stdout.
  If --save_fig is given: <name>.png (figure) + <name>.txt (report).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")          # no display needed — works on any cluster node
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Path) -> pl.DataFrame | None:
    files = sorted(results_dir.glob("lfc_df_*.tsv"))
    if not files:
        return None
    return pl.concat([pl.read_csv(f, separator="\t") for f in files])


def load_hybrids() -> pl.DataFrame:
    return pl.read_csv(config.HUMAN_CHIMP_HYBRIDS_DATA_PATH_WEXAC, separator="\t")


def run_live_prediction(gene_symbol: str) -> dict:
    """Run the full AlphaGenome pipeline for one gene on the fly."""
    from alphagenome.models import dna_client
    import pandas as pd
    import ag_predictions as ag

    human_gene_gff, human_exons_gff = ag.make_human_gff_usable()
    chimp_gene_gff, chimp_exons_gff = ag.make_chimp_gff_usable()

    human_row = human_gene_gff.filter(pl.col("Name") == gene_symbol)
    if human_row.height == 0:
        raise ValueError(f"'{gene_symbol}' not found in human GFF.")

    human_id = str(human_row["gff_ID"].item())
    chimp_id = str(human_row["chimp_ortholog_GeneID"].item())

    human_records = ag.get_sequence_records(config.HUMAN_RECORDS)
    chimp_records = ag.get_sequence_records(config.CHIMP_RECORDS)
    ag_model = dna_client.create(config.AG_API_KEY)
    gtf = pd.read_feather(
        "https://storage.googleapis.com/alphagenome/reference/gencode/"
        "hg38/gencode.v46.annotation.gtf.gz.feather"
    )

    lfc, h_expr, c_expr, h_sym, c_sym = ag.human_chimp_lfc(
        human_id, chimp_id,
        human_records, chimp_records,
        human_gene_gff, human_exons_gff,
        chimp_gene_gff, chimp_exons_gff,
        ag_model, gtf,
    )
    return {
        "GeneSymbol": h_sym,
        "HumanGeneID": human_id,
        "ChimpGeneID": chimp_id,
        "LFC": lfc,
        "HumanGeneExpression": h_expr,
        "ChimpGeneExpression": c_expr,
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(gene: dict, all_df: pl.DataFrame | None, hybrids: pl.DataFrame) -> str:
    lines = []
    sep = "=" * 60
    gene_name = gene["GeneSymbol"]
    lfc       = gene["LFC"]
    h_exp     = gene["HumanGeneExpression"]
    c_exp     = gene["ChimpGeneExpression"]

    lines += [sep, f"  Evaluation: {gene_name}", sep, ""]

    # --- Prediction --------------------------------------------------------
    lines.append("PREDICTION  (AlphaGenome — chondrocyte cell type)")
    if lfc is None:
        lines.append("  Could not be computed (gene likely has no exon matches).")
    else:
        direction = "human-biased" if lfc > 0 else "chimp-biased"
        lines.append(f"  Predicted LFC  (log2 human/chimp) : {lfc:+.4f}  [{direction}]")
        lines.append(f"  Human expression                  : {h_exp:.6f}")
        lines.append(f"  Chimp expression                  : {c_exp:.6f}")
        lines.append(f"  Human / Chimp ratio               : {h_exp / (c_exp + 1e-12):.2f}x")
    lines.append("")

    # --- Genome-wide context -----------------------------------------------
    if all_df is not None and lfc is not None:
        all_lfc = all_df["LFC"].drop_nulls().to_numpy()
        pct  = float(np.mean(all_lfc < lfc) * 100)
        z    = (lfc - np.mean(all_lfc)) / np.std(all_lfc)
        rank = int(np.sum(all_lfc > lfc)) + 1 if lfc > 0 else int(np.sum(all_lfc < lfc)) + 1

        lines.append("GENOME-WIDE CONTEXT")
        lines.append(f"  Total genes with predictions      : {len(all_lfc):,}")
        lines.append(f"  LFC percentile                    : {pct:.1f}th")
        lines.append(f"  Z-score                           : {z:+.2f}")
        if lfc > 0:
            lines.append(f"  Rank among human-upregulated      : #{rank:,}")
        else:
            lines.append(f"  Rank among chimp-upregulated      : #{rank:,}")
        lines.append(f"  All-gene LFC mean ± SD            : {np.mean(all_lfc):+.3f} ± {np.std(all_lfc):.3f}")
        lines.append("")

    # --- Hybrid ASE comparison ---------------------------------------------
    lines.append("HYBRID ASE DATA  (multi-cell-type)")
    gene_ase = hybrids.filter(pl.col("Gene") == gene_name)

    if gene_ase.height == 0:
        lines.append("  Gene not found in hybrid ASE dataset.")
    else:
        row = gene_ase.row(0, named=True)

        exp_lfc  = row.get("ExpLBM_LFC_human_ref")
        exp_padj = row.get("ExpLBM_LFC_padj_human_ref")
        h_tpm    = row.get("ExpLBM_TPM_human_allele")
        c_tpm    = row.get("ExpLBM_TPM_chimp_allele")
        ase_type = row.get("ExpLBM_gene_ase_type", "n/a")
        ase_cnt  = row.get("ASE_count", "n/a")

        def fmt(v, fmt_str):
            return fmt_str.format(v) if v is not None else "n/a"

        lines.append(f"  Experimental LFC  (ExpLBM)        : {fmt(exp_lfc, '{:+.4f}')}")
        lines.append(f"  Adjusted p-value                  : {fmt(exp_padj, '{:.2e}')}")
        lines.append(f"  Human TPM  (hybrid)               : {fmt(h_tpm, '{:.4f}')}")
        lines.append(f"  Chimp TPM  (hybrid)               : {fmt(c_tpm, '{:.4f}')}")
        lines.append(f"  ASE classification  (ExpLBM)      : {ase_type}")
        lines.append(f"  N cell types with ASE signal      : {ase_cnt}")

        if lfc is not None and exp_lfc is not None:
            match = "YES ✓" if (lfc > 0) == (exp_lfc > 0) else "NO  ✗"
            lines += [
                "",
                "  PREDICTION vs EXPERIMENT",
                f"    Direction match               : {match}",
                f"    Predicted LFC                 : {lfc:+.4f}",
                f"    Experimental LFC              : {exp_lfc:+.4f}",
                f"    Difference (pred - exp)       : {lfc - exp_lfc:+.4f}",
            ]

        # Top / bottom cell types
        lfc_cols = [c for c in hybrids.columns
                    if c.endswith("_LFC_human_ref") and not c.startswith("ExpLBM")]
        ct = [(c.replace("_LFC_human_ref", ""), row.get(c))
              for c in lfc_cols
              if row.get(c) is not None and np.isfinite(float(row.get(c)))]
        ct = sorted(ct, key=lambda x: float(x[1]), reverse=True)

        if ct:
            lines += ["", "  TOP HUMAN-BIASED CELL TYPES:"]
            for name, val in ct[:5]:
                lines.append(f"    {name:<52}: {float(val):+.3f}")
            lines += ["", "  TOP CHIMP-BIASED CELL TYPES:"]
            for name, val in reversed(ct[-5:]):
                lines.append(f"    {name:<52}: {float(val):+.3f}")

    lines += ["", sep]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(gene: dict, all_df: pl.DataFrame | None, hybrids: pl.DataFrame) -> plt.Figure:
    gene_name = gene["GeneSymbol"]
    lfc   = gene["LFC"]
    h_exp = gene.get("HumanGeneExpression") or 0
    c_exp = gene.get("ChimpGeneExpression")  or 0

    gene_ase = hybrids.filter(pl.col("Gene") == gene_name)
    has_ase  = gene_ase.height > 0

    ncols = 3 if has_ase else 2
    fig   = plt.figure(figsize=(7 * ncols, 5))
    gs    = gridspec.GridSpec(1, ncols, figure=fig)

    # Panel 1: LFC histogram
    ax1 = fig.add_subplot(gs[0])
    if all_df is not None:
        ax1.hist(all_df["LFC"].drop_nulls().to_numpy(), bins=80,
                 color="steelblue", alpha=0.7, edgecolor="none", label="All genes")
    if lfc is not None:
        ax1.axvline(lfc, color="red", linewidth=2, linestyle="--",
                    label=f"{gene_name}  ({lfc:+.2f})")
    ax1.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax1.set_xlabel("log₂ FC (Human / Chimp)")
    ax1.set_ylabel("Gene count")
    ax1.set_title("LFC distribution — gene position")
    ax1.legend()

    # Panel 2: predicted expression
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(["Human\n(predicted)", "Chimp\n(predicted)"],
                   [h_exp, c_exp], color=["steelblue", "tomato"], alpha=0.8)
    for bar, val in zip(bars, [h_exp, c_exp]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Predicted RNA-seq signal")
    ax2.set_title(f"{gene_name} — Predicted expression")

    # Panel 3: cell-type ASE
    if has_ase:
        ax3 = fig.add_subplot(gs[2])
        row = gene_ase.row(0, named=True)
        lfc_cols = [c for c in hybrids.columns
                    if c.endswith("_LFC_human_ref") and not c.startswith("ExpLBM")]
        ct_names, ct_vals = [], []
        for col in lfc_cols:
            val = row.get(col)
            if val is not None and np.isfinite(float(val)):
                ct_names.append(col.replace("_LFC_human_ref", ""))
                ct_vals.append(float(val))
        if ct_names:
            order    = np.argsort(ct_vals)
            ct_names = [ct_names[i] for i in order]
            ct_vals  = [ct_vals[i]  for i in order]
            colors   = ["tomato" if v < 0 else "steelblue" for v in ct_vals]
            ax3.barh(range(len(ct_names)), ct_vals, color=colors, alpha=0.8)
            ax3.set_yticks(range(len(ct_names)))
            ax3.set_yticklabels(ct_names, fontsize=7)
            ax3.axvline(0, color="black", linewidth=0.8)
            if lfc is not None:
                ax3.axvline(lfc, color="red", linewidth=1.5, linestyle="--",
                            label=f"Predicted LFC = {lfc:+.2f}")
                ax3.legend(fontsize=8)
            ax3.set_xlabel("log₂ FC (Human / Chimp)")
            ax3.set_title(f"{gene_name} — Experimental ASE by cell type")

    fig.suptitle(f"AlphaGenome evaluation: {gene_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_gene.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--gene", required=True, metavar="SYMBOL",
        help="Gene symbol to evaluate (e.g. SAMD11, NOC2L, KLHL17)",
    )
    parser.add_argument(
        "--results_dir", default="results/all_genes", metavar="DIR",
        help="Folder containing lfc_df_*.tsv result chunks  (default: results/all_genes)",
    )
    parser.add_argument(
        "--save_fig", default=None, metavar="PATH",
        help="Save figure to this file (PNG/PDF). A .txt report is saved alongside it.",
    )
    parser.add_argument(
        "--no_live", action="store_true",
        help="Do not run a live prediction if the gene is missing from the results folder.",
    )
    args = parser.parse_args()

    gene_name   = args.gene.upper()
    results_dir = Path(args.results_dir)

    # 1. Load pre-computed results
    print(f"[1/4] Loading results from '{results_dir}' ...")
    all_df = load_all_results(results_dir)
    if all_df is None:
        print(f"      No result files found — folder is empty or doesn't exist yet.")

    # 2. Find gene in results
    gene_row = None
    if all_df is not None:
        match = all_df.filter(pl.col("GeneSymbol") == gene_name)
        if match.height > 0:
            gene_row = match.row(0, named=True)
            print(f"      Found {gene_name} in pre-computed results.")

    # 3. Live prediction fallback
    if gene_row is None:
        if args.no_live:
            print(f"ERROR: '{gene_name}' not found in results and --no_live was set.", file=sys.stderr)
            sys.exit(1)
        print(f"      '{gene_name}' not in results — running live prediction (this takes ~1 min) ...")
        try:
            gene_row = run_live_prediction(gene_name)
            print("      Prediction complete.")
        except Exception as e:
            print(f"ERROR: live prediction failed — {e}", file=sys.stderr)
            sys.exit(1)

    # 4. Load hybrid ASE data
    print("[2/4] Loading hybrid ASE data ...")
    try:
        hybrids = load_hybrids()
        print(f"      {hybrids.height:,} genes in hybrid dataset.")
    except Exception as e:
        print(f"      WARNING: could not load hybrid data ({e}) — skipping ASE comparison.")
        hybrids = pl.DataFrame({"Gene": pl.Series([], dtype=pl.Utf8)})

    # 5. Build and print report
    print("[3/4] Building report ...")
    report = build_report(gene_row, all_df, hybrids)
    print()
    print(report)

    # 6. Figure + saved files
    print("[4/4] Generating figure ...")
    fig = make_figure(gene_row, all_df, hybrids)

    if args.save_fig:
        save_path = Path(args.save_fig)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        txt_path = save_path.with_suffix(".txt")
        txt_path.write_text(report)
        print(f"      Figure saved : {save_path}")
        print(f"      Report saved : {txt_path}")
    else:
        default = Path(f"results/{gene_name}_eval.png")
        default.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(default, dpi=150, bbox_inches="tight")
        txt_path = default.with_suffix(".txt")
        txt_path.write_text(report)
        print(f"      Figure saved : {default}")
        print(f"      Report saved : {txt_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
