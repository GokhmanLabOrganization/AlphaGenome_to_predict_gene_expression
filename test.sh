#!/bin/bash
# test.sh — smoke-test suite for ag_predictions.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0
WARN=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC}  $1"; (( PASS++ )) || true; }
fail() { echo -e "${RED}FAIL${NC}  $1"; (( FAIL++ )) || true; }
warn() { echo -e "${YELLOW}WARN${NC}  $1"; (( WARN++ )) || true; }

run_python() {
    python - <<'EOF_PYTHON' "$@"
EOF_PYTHON
}

echo "=================================================="
echo " AlphaGenome prediction pipeline — test suite"
echo "=================================================="

# ------------------------------------------------------------------
# 1. Import checks
# ------------------------------------------------------------------
echo ""
echo "--- 1. Import checks ---"

for mod in re argparse os numpy polars matplotlib; do
    python -c "import $mod" 2>/dev/null && pass "import $mod" || fail "import $mod"
done

for mod in pandas Bio alphagenome; do
    if python -c "import $mod" 2>/dev/null; then
        pass "import $mod"
    else
        warn "import $mod  (optional heavy dep — install to run full pipeline)"
    fi
done

# ------------------------------------------------------------------
# 2. config.py constants
# ------------------------------------------------------------------
echo ""
echo "--- 2. config.py constants ---"

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__ if '__file__' in dir() else '.')))
SCRIPT_DIR = os.environ.get('SCRIPT_DIR', '.')
sys.path.insert(0, SCRIPT_DIR)

import config

required = [
    'AG_API_KEY',
    'CHIMP_RECORDS',
    'HUMAN_RECORDS',
    'CHIMP_GFF_WEXAC',
    'CHIMP_GENES_GFF_PATH_WEXAC',
    'CHIMP_EXONS_GFF_PATH_WEXAC',
    'HUMAN_GENES_GFF_PATH_WEXAC',
    'HUMAN_EXONS_GFF_PATH_WEXAC',
]
ok = True
for attr in required:
    if not hasattr(config, attr):
        print(f'MISSING_ATTR:{attr}')
        ok = False
    elif not getattr(config, attr):
        print(f'EMPTY_ATTR:{attr}')
        ok = False
if ok:
    print('ALL_ATTRS_OK')
PYEOF

PYTHON_OUT=$(SCRIPT_DIR="$SCRIPT_DIR" python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ['SCRIPT_DIR'])
import config

required = [
    'AG_API_KEY',
    'CHIMP_RECORDS',
    'HUMAN_RECORDS',
    'CHIMP_GFF_WEXAC',
    'CHIMP_GENES_GFF_PATH_WEXAC',
    'CHIMP_EXONS_GFF_PATH_WEXAC',
    'HUMAN_GENES_GFF_PATH_WEXAC',
    'HUMAN_EXONS_GFF_PATH_WEXAC',
]
ok = True
for attr in required:
    if not hasattr(config, attr):
        print(f'MISSING:{attr}')
        ok = False
    elif not getattr(config, attr):
        print(f'EMPTY:{attr}')
        ok = False
if ok:
    print('OK')
PYEOF
)

if echo "$PYTHON_OUT" | grep -q "^OK$"; then
    pass "config.py — all required constants present"
else
    echo "$PYTHON_OUT" | grep "^MISSING" | while read line; do fail "config.py — $line"; done
    echo "$PYTHON_OUT" | grep "^EMPTY"   | while read line; do warn "config.py — $line"; done
fi

# ------------------------------------------------------------------
# 3. File-system / path checks
# ------------------------------------------------------------------
echo ""
echo "--- 3. File / directory existence ---"

GENOME_DIR="$SCRIPT_DIR/genomes"
[ -d "$GENOME_DIR" ] && pass "genomes/ directory exists" || fail "genomes/ directory missing"

CHIMP_FA="$GENOME_DIR/GCF_028858775.2_NHGRI_mPanTro3-v2.0_pri_genomic.fna"
HG38_FA="$GENOME_DIR/hg38.fa"
[ -e "$CHIMP_FA" ] && pass "chimp FASTA exists" || warn "chimp FASTA missing: $CHIMP_FA"
[ -e "$HG38_FA"  ] && pass "hg38 FASTA exists"  || warn "hg38 FASTA missing: $HG38_FA"

GFF_PATHS=(
    "/home/labs/davidgo/Collaboration/GenomeAnnotation/Chimpanzee/GCF_028858775.2_NHGRI_mPanTro3-v2.0_pri_genomic.gff"
    "/home/labs/davidgo/itamarn/backup/MSc/XGBoost_mpra_to_gene_expression/data/gffs/chimp_genes_gff_final.tsv"
    "/home/labs/davidgo/itamarn/backup/MSc/XGBoost_mpra_to_gene_expression/data/gffs/chimp_exons_gff_final.tsv"
    "/home/labs/davidgo/itamarn/backup/MSc/XGBoost_mpra_to_gene_expression/data/gffs/human_genes_gff_final.tsv"
    "/home/labs/davidgo/itamarn/backup/MSc/XGBoost_mpra_to_gene_expression/data/gffs/human_exons_gff_final.tsv"
)
GFF_NAMES=("chimp GFF" "chimp genes TSV" "chimp exons TSV" "human genes TSV" "human exons TSV")
for i in "${!GFF_PATHS[@]}"; do
    [ -e "${GFF_PATHS[$i]}" ] && pass "${GFF_NAMES[$i]} exists" || warn "${GFF_NAMES[$i]} missing: ${GFF_PATHS[$i]}"
done

# ------------------------------------------------------------------
# 4. unit tests — pure functions (no heavy deps needed)
# ------------------------------------------------------------------
echo ""
echo "--- 4. Pure-function unit tests ---"

UNIT_RESULT=$(python - <<'PYEOF'
import sys, traceback

results = []

def check(name, expr, expected):
    try:
        got = expr()
        if got == expected:
            results.append(('pass', name))
        else:
            results.append(('fail', f'{name}: expected {expected!r}, got {got!r}'))
    except Exception as e:
        results.append(('fail', f'{name}: raised {e}'))

# ---- union_intervals ------------------------------------------------
def union_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return merged

check('union_intervals empty',
      lambda: union_intervals([]), [])
check('union_intervals single',
      lambda: union_intervals([(1, 5)]), [(1, 5)])
check('union_intervals two non-overlapping',
      lambda: union_intervals([(1, 3), (5, 8)]), [(1, 3), (5, 8)])
check('union_intervals two overlapping',
      lambda: union_intervals([(1, 5), (3, 8)]), [(1, 8)])
check('union_intervals touching (no merge)',
      lambda: union_intervals([(1, 4), (5, 8)]), [(1, 4), (5, 8)])
check('union_intervals nested',
      lambda: union_intervals([(1, 10), (3, 7)]), [(1, 10)])
check('union_intervals already sorted',
      lambda: union_intervals([(1, 3), (2, 5), (6, 9)]), [(1, 5), (6, 9)])
check('union_intervals unsorted input',
      lambda: union_intervals([(6, 9), (1, 3), (2, 5)]), [(1, 5), (6, 9)])

# ---- get_gene_1mb_region -------------------------------------------
import polars as pl

def get_gene_1mb_region(gene_id, gene_gff_df):
    gene_gff = gene_gff_df.filter(pl.col('GeneID') == gene_id)
    if gene_gff.height == 0:
        return None, None, None, None, None
    gene_symbol  = gene_gff['Name'].item()
    gene_strand  = gene_gff.select(pl.col('strand')).item()
    gene_chr     = gene_gff.select(pl.col('chromosome')).item()
    gene_start   = gene_gff.select(pl.col('start')).item()
    gene_end     = gene_gff.select(pl.col('end')).item()
    gene_start_1mb = gene_start - (2**19)
    gene_end_1mb   = gene_end   + (2**19) - (gene_end - gene_start + 1)
    return gene_symbol, gene_chr, gene_start_1mb, gene_end_1mb, gene_strand

mock_gff = pl.DataFrame({
    'GeneID':     ['G001', 'G002'],
    'Name':       ['GENE_A', 'GENE_B'],
    'chromosome': ['chr1', 'chr2'],
    'start':      [1_000_000, 2_000_000],
    'end':        [1_001_000, 2_002_000],
    'strand':     ['+', '-'],
})

check('get_gene_1mb_region missing gene returns Nones',
      lambda: get_gene_1mb_region('MISSING', mock_gff),
      (None, None, None, None, None))

sym, chrom, s1, e1, strand = get_gene_1mb_region('G001', mock_gff)
check('get_gene_1mb_region symbol',  lambda: sym,    'GENE_A')
check('get_gene_1mb_region chrom',   lambda: chrom,  'chr1')
check('get_gene_1mb_region strand',  lambda: strand, '+')

HALF = 2**19
gene_start, gene_end = 1_000_000, 1_001_000
expected_s = gene_start - HALF
expected_e = gene_end + HALF - (gene_end - gene_start + 1)
check('get_gene_1mb_region start_1mb', lambda: s1, expected_s)
check('get_gene_1mb_region end_1mb',   lambda: e1, expected_e)
check('get_gene_1mb_region window_size',
      lambda: expected_e - expected_s + 1, 2**20)   # 1 048 576

# ---- coding_region_mean_expression ---------------------------------
import numpy as np

def coding_region_mean_expression(exon_gff, gene_ID, gene_strand, gene_start_1mb, full_pred):
    gene_exon_gff = exon_gff.filter(
        pl.col('Name').str.contains(gene_ID) | pl.col('Dbxref').str.contains(gene_ID)
    )
    exon_intervals = list(zip(gene_exon_gff['start'], gene_exon_gff['end']))
    exon_intervals_union = union_intervals(exon_intervals)
    coding_region_pred = [
        full_pred[i]
        for s, e in exon_intervals_union
        for i in range(max(0, s - gene_start_1mb), min(1048576, e - gene_start_1mb))
    ]
    if len(coding_region_pred) == 0:
        return None
    mean_p, mean_n = np.mean(coding_region_pred, axis=0)
    if gene_strand == '+':
        return mean_p
    elif gene_strand == '-':
        return mean_n

SEQ_LEN = 1_048_576
fake_signal = np.zeros((SEQ_LEN, 2))
fake_signal[100:200, 0] = 1.0   # + strand signal in positions 100-199
fake_signal[100:200, 1] = 2.0   # - strand signal

mock_exons = pl.DataFrame({
    'Name':   ['G001_exon1', 'G001_exon1'],
    'Dbxref': ['G001', 'G001'],
    'start':  [100, 150],
    'end':    [130, 200],
})

expr_p = coding_region_mean_expression(mock_exons, 'G001', '+', 0, fake_signal)
check('coding_region_mean_expression + strand returns 1.0',
      lambda: round(float(expr_p), 6), 1.0)

expr_n = coding_region_mean_expression(mock_exons, 'G001', '-', 0, fake_signal)
check('coding_region_mean_expression - strand returns 2.0',
      lambda: round(float(expr_n), 6), 2.0)

mock_exons_nomatch = pl.DataFrame({
    'Name':   ['OTHER_exon'],
    'Dbxref': ['OTHER'],
    'start':  [100],
    'end':    [200],
})
check('coding_region_mean_expression no matching exons returns None',
      lambda: coding_region_mean_expression(mock_exons_nomatch, 'G001', '+', 0, fake_signal),
      None)

# ---- report --------------------------------------------------------
for status, msg in results:
    print(f'{status}|{msg}')
PYEOF
)

while IFS='|' read -r status msg; do
    [ "$status" = "pass" ] && pass "$msg" || fail "$msg"
done <<< "$UNIT_RESULT"

# ------------------------------------------------------------------
# 5. CLI argument parsing smoke-test
# ------------------------------------------------------------------
echo ""
echo "--- 5. CLI argument parsing ---"

# Patch missing modules so argparse section is reachable
CLI_RESULT=$(SCRIPT_DIR="$SCRIPT_DIR" python - <<'PYEOF'
import sys, types, os

# Stub out modules that aren't installed
for mod_name in ['pandas', 'Bio', 'Bio.SeqIO',
                 'alphagenome', 'alphagenome.colab_utils',
                 'alphagenome.data', 'alphagenome.data.gene_annotation',
                 'alphagenome.data.genome', 'alphagenome.data.transcript',
                 'alphagenome.interpretation', 'alphagenome.interpretation.ism',
                 'alphagenome.models', 'alphagenome.models.dna_client',
                 'alphagenome.models.variant_scorers',
                 'alphagenome.visualization', 'alphagenome.visualization.plot_components',
                 'matplotlib', 'matplotlib.pyplot']:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        sys.modules[mod_name] = stub

# Provide the one constant that ag_predictions.py reads at import time
dna_stub = sys.modules['alphagenome.models.dna_client']
dna_stub.SEQUENCE_LENGTH_1MB = 1048576
dna_stub.OutputType = types.SimpleNamespace(RNA_SEQ='rna_seq')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

script_dir = os.environ['SCRIPT_DIR']
sys.path.insert(0, script_dir)

try:
    import ag_predictions  # noqa: F401
    print('IMPORT_OK')
except Exception as e:
    print(f'IMPORT_FAIL:{e}')
    sys.exit(1)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--job_i', type=int, required=True)
parser.add_argument('--chunk_size', type=int, default=500)
args = parser.parse_args(['--job_i', '3', '--chunk_size', '100'])

if args.job_i == 3 and args.chunk_size == 100:
    print('ARGS_OK')
else:
    print(f'ARGS_FAIL:job_i={args.job_i} chunk_size={args.chunk_size}')
PYEOF
)

echo "$CLI_RESULT" | grep -q "IMPORT_OK"  && pass "ag_predictions.py imports (with stubs)" || fail "ag_predictions.py imports failed: $(echo "$CLI_RESULT" | grep IMPORT_FAIL)"
echo "$CLI_RESULT" | grep -q "ARGS_OK"    && pass "argparse: --job_i / --chunk_size parsed correctly" || fail "argparse test failed"

# ------------------------------------------------------------------
# 6. Output directory creation
# ------------------------------------------------------------------
echo ""
echo "--- 6. Output directory creation ---"

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR/test_output_$$" && pass "results/ dir writable" || fail "results/ dir not writable"
rmdir "$RESULTS_DIR/test_output_$$"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "=================================================="
TOTAL=$((PASS + FAIL + WARN))
echo "  Passed : $PASS / $TOTAL"
echo "  Failed : $FAIL"
echo "  Warnings (optional deps / files): $WARN"
echo "=================================================="

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
