#!/bin/bash
# Submit TPM regression job using AlphaGenome scalar predictions + MPRA features.
# Models: LinearRegression (baseline) + XGBoost + FCNet.
#
# Usage:
#   bash submit_predict_tpm_regression_with_ag_preds.sh [human|chimp] [tpm|log10_tpm]
#
# Examples:
#   bash submit_predict_tpm_regression_with_ag_preds.sh                      # human, log10_tpm (defaults)
#   bash submit_predict_tpm_regression_with_ag_preds.sh chimp                # chimp, log10_tpm
#   bash submit_predict_tpm_regression_with_ag_preds.sh human tpm            # human, raw TPM
#   ASE_ONLY=1 bash submit_predict_tpm_regression_with_ag_preds.sh chimp log10_tpm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/predict_tpm_regression_with_ag_preds.py"

SPECIES=${1:-human}
TARGET=${2:-log10_tpm}

if [[ "$SPECIES" != "human" && "$SPECIES" != "chimp" ]]; then
    echo "Error: species must be 'human' or 'chimp', got '$SPECIES'" >&2
    exit 1
fi
if [[ "$TARGET" != "tpm" && "$TARGET" != "log10_tpm" ]]; then
    echo "Error: target must be 'tpm' or 'log10_tpm', got '$TARGET'" >&2
    exit 1
fi

AG_PREDS_GLOB=${AG_PREDS_GLOB:-"$SCRIPT_DIR/results/all_genes/*.tsv"}
ASE_ONLY=${ASE_ONLY:-0}
USE_GPU=${USE_GPU:-0}
HIDDEN_DIM=${HIDDEN_DIM:-256}
EPOCHS=${EPOCHS:-300}
LOG_DIR=${LOG_DIR:-/home/labs/davidgo/itamarn/log/ag_preds_mpra_regression_log}
QUEUE=${QUEUE:-short}
CONDA_ENV=${CONDA_ENV:-mpra_model_env}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}

mkdir -p "$LOG_DIR"

JOB_NAME="ag_mpra_tpm_${SPECIES}_${TARGET}"

echo "Submitting AG predictions + MPRA TPM regression job"
echo "  Script      : $SCRIPT_PATH"
echo "  Species     : $SPECIES"
echo "  Target      : $TARGET"
echo "  AG preds    : $AG_PREDS_GLOB"
echo "  ASE only    : $ASE_ONLY"
echo "  Use GPU     : $USE_GPU"
echo "  Hidden dim  : $HIDDEN_DIM"
echo "  Epochs      : $EPOCHS"
echo "  Queue       : $QUEUE"

JOB_SCRIPT=$(mktemp "${LOG_DIR}/${JOB_NAME}.XXXXXX.sh")
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
source "$CONDA_SH" && conda activate "$CONDA_ENV" &&
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}" &&
cd "$SCRIPT_DIR" &&
python "$SCRIPT_PATH" \\
    --ag-preds-glob "$AG_PREDS_GLOB" \\
    --species "$SPECIES" \\
    --target "$TARGET" \\
    --hidden-dim "$HIDDEN_DIM" \\
    --epochs "$EPOCHS" \\
    \$([ "$ASE_ONLY" = "1" ] && echo "--ase-only") \\
    \$([ "$USE_GPU"  = "1" ] && echo "--gpu")
EOF

if [ "$USE_GPU" = "1" ]; then
    bsub -J "$JOB_NAME" \
        -q "${QUEUE}-gpu" \
        -n 4 \
        -gpu "num=1:j_exclusive=yes:gmem=128GB" \
        -R "rusage[mem=64000] span[ptile=4] select[gpu]" \
        -o "${LOG_DIR}/${JOB_NAME}.%J.o" \
        -e "${LOG_DIR}/${JOB_NAME}.%J.e" \
        bash "$JOB_SCRIPT"
else
    bsub -J "$JOB_NAME" \
        -q "$QUEUE" \
        -n 4 \
        -R "rusage[mem=64000] span[ptile=4]" \
        -o "${LOG_DIR}/${JOB_NAME}.%J.o" \
        -e "${LOG_DIR}/${JOB_NAME}.%J.e" \
        bash "$JOB_SCRIPT"
fi

echo "Job submitted. Logs: ${LOG_DIR}/${JOB_NAME}.<jobid>.{o,e}"
