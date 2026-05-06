# AlphaGenome to Predict Gene Expression

Uses [AlphaGenome](https://deepmind.google/technologies/alphagenome/) to predict and compare RNA-seq signal between human and chimpanzee orthologous genes. For each gene pair the pipeline extracts a 1 Mb genomic window from each species' reference genome, runs it through the AlphaGenome DNA model, and summarises the predicted RNA-seq signal over the union of exonic intervals to produce a per-gene **log₂ fold-change (LFC)** between species.

---

## Background

The goal is to identify genes with large predicted expression differences between humans and chimpanzees — a computational complement to hybrid-cell ASE experiments. The AlphaGenome model predicts multi-track RNA-seq signal directly from DNA sequence, making it possible to attribute expression differences to cis-regulatory sequence changes without cross-species RNA experiments.

---

## Repository structure

```
.
├── ag_predictions.py               # Core pipeline: all functions + CLI entry-point
├── config.template.py              # Template — copy to config.py and fill in your API key
├── submit_alphagenome_pred_over_genes_jobs.sh   # LSF batch-submission script (WEXAC cluster)
├── evaluate_AG_gene_preds.ipynb    # Notebook for exploring and plotting results
├── test.sh                         # Bash test suite (39 tests)
└── results/
    ├── code_test.txt               # Saved output of the last test run
    └── test/
        └── lfc_df_0_5.tsv          # Pilot run — first 5 gene pairs
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/itanini/AlphaGenome_to_predict_gene_expression.git
cd AlphaGenome_to_predict_gene_expression
```

### 2. Install dependencies

```bash
pip install numpy polars pandas biopython matplotlib alphagenome
```

### 3. Configure

```bash
cp config.template.py config.py
```

Open `config.py` and set your AlphaGenome API key:

```python
AG_API_KEY = 'YOUR_ALPHAGENOME_API_KEY'
```

All other paths in `config.py` point to shared data on the WEXAC cluster (`/home/labs/davidgo/...`). Update them if running elsewhere.

> `config.py` is listed in `.gitignore` and will never be committed — keep your API key out of version control.

### 4. Genome FASTA files

The pipeline expects two FASTA files inside a `genomes/` directory (also in `.gitignore`):

| File | Species | Source |
|------|---------|--------|
| `genomes/hg38/hg38.fa` | Human (GRCh38) | UCSC / NCBI |
| `genomes/GCF_028858775.2_NHGRI_mPanTro3-v2.0_pri_genomic.fna` | Chimpanzee (mPanTro3) | NCBI GCF_028858775.2 |

On WEXAC these are symlinked from the shared genomes directory and already present.

---

## Running the pipeline

### Single chunk (interactive / testing)

```bash
python ag_predictions.py --job_i 0 --chunk_size 5 --output_dir results/test
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--job_i` | Zero-based chunk index | required |
| `--chunk_size` | Number of gene pairs per chunk | 500 |
| `--output_dir` | Directory to write the TSV | `results/AG_predictions_<timestamp>` |

Genes processed: `[job_i × chunk_size, (job_i+1) × chunk_size)`.

### Full run on WEXAC cluster (LSF)

```bash
bash submit_alphagenome_pred_over_genes_jobs.sh /path/to/output_dir
```

This submits one GPU job per 500-gene chunk across all ~17 888 orthologous gene pairs (≈36 jobs). Each job requests 1 GPU, 4 cores, and 16 GB RAM on the `medium` queue. Already-completed output files are skipped automatically.

---

## Output format

Each run produces a tab-separated file `lfc_df_{start}_{end}.tsv`:

| Column | Description |
|--------|-------------|
| `GeneSymbol` | Human gene symbol |
| `HumanGeneID` | NCBI human gene ID |
| `ChimpGeneID` | NCBI chimp ortholog gene ID |
| `LFC` | log₂(human expression / chimp expression) |
| `HumanGeneExpression` | Mean predicted RNA-seq signal over human exons (+ strand) |
| `ChimpGeneExpression` | Mean predicted RNA-seq signal over chimp exons (+ strand) |

Positive LFC → higher predicted expression in humans; negative → higher in chimps.

### Example (first 5 genes)

| GeneSymbol | LFC | HumanExpr | ChimpExpr |
|------------|-----|-----------|-----------|
| SAMD11 | 5.61 | 0.302 | 0.0062 |
| NOC2L | 5.30 | 1.665 | 0.0423 |
| KLHL17 | 13.25 | 0.333 | 0.000034 |
| PLEKHN1 | 10.24 | 0.032 | 0.000026 |
| PERM1 | −4.69 | 0.0012 | 0.0315 |

---

## How it works

```
For each orthologous gene pair:
  1. get_gene_1mb_region()       — find gene coordinates in GFF, compute 1 Mb window
  2. seq_retrieve_interval()     — extract + N-pad the DNA sequence to exactly 1 048 576 bp
  3. AG_gene_prediction()        — call AlphaGenome to predict RNA-seq tracks (CL:0000138, chondrocytes)
  4. coding_region_mean_expression() — average predicted signal over union-merged exon intervals
  5. human_chimp_lfc()           — compute log₂(human / chimp)
```

The predicted RNA-seq has two channels (+ and − strand); the channel matching the gene's annotated strand is used.

Exon intervals are union-merged with `union_intervals()` before signal extraction to avoid double-counting overlapping exons from multiple transcripts.

---

## Testing

```bash
bash test.sh
```

The suite runs 39 tests across 6 categories and exits with code 0 on full pass:

| Section | Tests |
|---------|-------|
| Import checks | 9 — all required packages |
| config.py constants | 1 — all 8 required keys present |
| File / path existence | 8 — genomes, GFFs, annotation TSVs |
| Pure-function unit tests | 18 — `union_intervals`, `get_gene_1mb_region`, `coding_region_mean_expression` |
| CLI argument parsing | 2 — module import + argparse |
| Output directory | 1 — results/ is writable |

---

## Key functions (`ag_predictions.py`)

| Function | Purpose |
|----------|---------|
| `union_intervals(intervals)` | Merge overlapping exon intervals |
| `get_gene_1mb_region(gene_id, gff)` | Compute 1 Mb window coordinates for a gene |
| `seq_retrieve_interval(chr, start, end, records)` | Extract and pad a DNA sequence to 1 Mb |
| `AG_gene_prediction(model, gtf, seq, gene)` | Run AlphaGenome RNA-seq prediction |
| `coding_region_mean_expression(exon_gff, gene_id, strand, start_1mb, pred)` | Summarise predicted signal over exons |
| `calc_gene_expression(gene_id, records, ...)` | End-to-end expression for one gene |
| `human_chimp_lfc(human_id, chimp_id, ...)` | LFC for one ortholog pair |
| `go_over_all_genes(start_i, end_i)` | Batch loop over a chunk of gene pairs |

---

## Dependencies

| Package | Version tested | Purpose |
|---------|---------------|---------|
| `alphagenome` | 0.6.1 | DNA → RNA-seq prediction model |
| `polars` | 1.36 | GFF / annotation DataFrames |
| `numpy` | 2.2 | Signal aggregation |
| `biopython` | 1.87 | FASTA parsing |
| `pandas` | 2.3 | GTF loading (AlphaGenome requirement) |
| `matplotlib` | 3.10 | Visualization (optional, used in `show=True` mode) |
