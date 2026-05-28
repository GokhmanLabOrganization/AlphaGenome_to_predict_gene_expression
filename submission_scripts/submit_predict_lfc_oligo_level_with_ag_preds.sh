#!/bin/bash
# Submit oligo-level LFC regression job using AlphaGenome scalar predictions + per-oligo MPRA features.
# Default: ASE genes only.
#
# Usage:
#   bash submit_predict_lfc_oligo_level_with_ag_preds.sh [lfc|log10_lfc]
#
# Examples:
#   bash submit_predict_lfc_oligo_level_with_ag_preds.sh              # lfc, ASE only (defaults)
#   bash submit_predict_lfc_oligo_level_with_ag_preds.sh log10_lfc    # log10 target
#   NO_ASE_ONLY=1 bash submit_predict_lfc_oligo_level_with_ag_preds.sh lfc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/predict_lfc_oligo_level_with_ag_preds.py"

TARGET=${1:-lfc}
if [[ "$TARGET" != "lfc" && "$TARGET" != "log10_lfc" ]]; then
    echo "Error: target must be 'lfc' or 'log10_lfc', got '$TARGET'" >&2
    exit 1
fi

AG_PREDS_GLOB=${AG_PREDS_GLOB:-"$SCRIPT_DIR/results/all_genes/*.tsv"}
NO_ASE_ONLY=${NO_ASE_ONLY:-0}
LFC_THRESHOLD=${LFC_THRESHOLD:-0.0}
USE_GPU=${USE_GPU:-0}
LOG_DIR=${LOG_DIR:-/home/labs/davidgo/itamarn/log/ag_preds_mpra_regression_log}
QUEUE=${QUEUE:-short}
CONDA_ENV=${CONDA_ENV:-mpra_model_env}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}

mkdir -p "$LOG_DIR"

JOB_NAME="ag_mpra_oligo_${TARGET}"

echo "Submitting AG predictions + MPRA oligo-level LFC regression job"
echo "  Script      : $SCRIPT_PATH"
echo "  Target      : $TARGET"
echo "  AG preds    : $AG_PREDS_GLOB"
echo "  LFC thresh  : $LFC_THRESHOLD"
echo "  ASE only    : $([ "$NO_ASE_ONLY" = "1" ] && echo "no" || echo "yes")"
echo "  Use GPU     : $USE_GPU"
echo "  Queue       : $QUEUE"

JOB_SCRIPT=$(mktemp "${LOG_DIR}/${JOB_NAME}.XXXXXX.sh")
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
source "$CONDA_SH" && conda activate "$CONDA_ENV" &&
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}" &&
cd "$SCRIPT_DIR" &&
python "$SCRIPT_PATH" \\
    --ag-preds-glob "$AG_PREDS_GLOB" \\
    --lfc-threshold "$LFC_THRESHOLD" \\
    --target "$TARGET" \\
    \$([ "$NO_ASE_ONLY" = "1" ] && echo "--no-ase-only") \\
    \$([ "$USE_GPU"     = "1" ] && echo "--gpu")
EOF

if [ "$USE_GPU" = "1" ]; then
    bsub -J "$JOB_NAME" \
        -q "${QUEUE}-gpu" \
        -n 4 \
        -gpu "num=1:mode=exclusive_process" \
        -R "rusage[mem=32000] span[ptile=4] select[gpu]" \
        -o "${LOG_DIR}/${JOB_NAME}.%J.o" \
        -e "${LOG_DIR}/${JOB_NAME}.%J.e" \
        bash "$JOB_SCRIPT"
else
    bsub -J "$JOB_NAME" \
        -q "$QUEUE" \
        -n 4 \
        -R "rusage[mem=32000] span[ptile=4]" \
        -o "${LOG_DIR}/${JOB_NAME}.%J.o" \
        -e "${LOG_DIR}/${JOB_NAME}.%J.e" \
        bash "$JOB_SCRIPT"
fi

echo "Job submitted. Logs: ${LOG_DIR}/${JOB_NAME}.<jobid>.{o,e}"
