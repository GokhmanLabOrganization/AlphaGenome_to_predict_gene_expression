#!/bin/bash
# One-time job: stream all human/chimp AG embeddings, pair them by gene,
# subsample positions, and save assembled matrices to data/ for reuse.
#
# Memory sizing:
#   N_SAMPLES_PER_GENE=200 → ~38 GB per species → ~76 GB output files
#   N_SAMPLES_PER_GENE=100 → ~19 GB per species → ~38 GB output files
#   N_SAMPLES_PER_GENE=50  → ~10 GB per species → ~20 GB output files
#
# Usage:
#   bash submit_create_paired_embedding_dataset.sh
#   N_SAMPLES_PER_GENE=100 bash submit_create_paired_embedding_dataset.sh
#   N_SAMPLES_PER_GENE=200 OUTPUT_SUFFIX=_n200 bash submit_create_paired_embedding_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/create_paired_embedding_dataset.py"

HUMAN_EMB_DIR=${HUMAN_EMB_DIR:-"$SCRIPT_DIR/../create_AG_embeddings/embeddings/human"}
CHIMP_EMB_DIR=${CHIMP_EMB_DIR:-"$SCRIPT_DIR/../create_AG_embeddings/embeddings/chimp"}
N_SAMPLES_PER_GENE=${N_SAMPLES_PER_GENE:-200}
MAX_POSITIONS=${MAX_POSITIONS:-""}
RANDOM_SEED=${RANDOM_SEED:-42}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-"_n${N_SAMPLES_PER_GENE}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/data/paired_embeddings${OUTPUT_SUFFIX}"}
LOG_DIR=${LOG_DIR:-/home/labs/davidgo/itamarn/log/create_paired_emb_log}
QUEUE=${QUEUE:-long}
CONDA_ENV=${CONDA_ENV:-mpra_model_env}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}

mkdir -p "$LOG_DIR"

JOB_NAME="create_paired_emb"

# Memory estimate (conservative): 3 × (N_SAMPLES_PER_GENE / 200) × 100 GB
# With N=200: request 320 GB. With N=100: 160 GB. With N=50: 80 GB.
MEM_MB=$(( (N_SAMPLES_PER_GENE + 199) / 200 * 320000 ))
if [ "$MEM_MB" -lt 64000 ]; then MEM_MB=64000; fi

echo "Submitting paired embedding dataset creation job"
echo "  Script           : $SCRIPT_PATH"
echo "  Human emb dir    : $HUMAN_EMB_DIR"
echo "  Chimp emb dir    : $CHIMP_EMB_DIR"
echo "  N samples/gene   : $N_SAMPLES_PER_GENE"
echo "  Max positions    : ${MAX_POSITIONS:-none}"
echo "  Output dir       : $OUTPUT_DIR"
echo "  Memory request   : ${MEM_MB} MB"
echo "  Queue            : $QUEUE"

JOB_SCRIPT=$(mktemp "${LOG_DIR}/${JOB_NAME}.XXXXXX.sh")
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
source "$CONDA_SH" && conda activate "$CONDA_ENV" &&
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}" &&
cd "$SCRIPT_DIR" &&
python "$SCRIPT_PATH" \\
    --human-emb-dir "$HUMAN_EMB_DIR" \\
    --chimp-emb-dir "$CHIMP_EMB_DIR" \\
    --n-samples-per-gene "$N_SAMPLES_PER_GENE" \\
    --output-dir "$OUTPUT_DIR" \\
    --random-seed "$RANDOM_SEED" \\
    \$([ -n "$MAX_POSITIONS" ] && echo "--max-positions $MAX_POSITIONS")
EOF

bsub -J "$JOB_NAME" \
    -q "$QUEUE" \
    -n 4 \
    -R "rusage[mem=${MEM_MB}] span[ptile=4]" \
    -o "${LOG_DIR}/${JOB_NAME}.%J.o" \
    -e "${LOG_DIR}/${JOB_NAME}.%J.e" \
    bash "$JOB_SCRIPT"

echo "Job submitted. Logs: ${LOG_DIR}/${JOB_NAME}.<jobid>.{o,e}"
echo "Output will be in: $OUTPUT_DIR"
