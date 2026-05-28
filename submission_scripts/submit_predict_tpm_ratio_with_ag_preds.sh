#!/bin/bash
# Submit joint human+chimp TPM ratio regression job.
# Trains XGBoost for human and chimp on the same split,
# then produces a scatter of predicted log10(human/chimp) vs actual.
#
# Usage:
#   bash submit_predict_tpm_ratio_with_ag_preds.sh
#   ASE_ONLY=1 bash submit_predict_tpm_ratio_with_ag_preds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/predict_tpm_ratio_with_ag_preds.py"

AG_PREDS_GLOB=${AG_PREDS_GLOB:-"$SCRIPT_DIR/results/all_genes/*.tsv"}
ASE_ONLY=${ASE_ONLY:-0}
USE_GPU=${USE_GPU:-0}
LOG_DIR=${LOG_DIR:-/home/labs/davidgo/itamarn/log/ag_preds_mpra_regression_log}
QUEUE=${QUEUE:-short}
CONDA_ENV=${CONDA_ENV:-mpra_model_env}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}

mkdir -p "$LOG_DIR"

JOB_NAME="ag_tpm_ratio"

echo "Submitting AG predictions + MPRA TPM ratio regression job"
echo "  Script      : $SCRIPT_PATH"
echo "  AG preds    : $AG_PREDS_GLOB"
echo "  ASE only    : $ASE_ONLY"
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
    \$([ "$ASE_ONLY" = "1" ] && echo "--ase-only") \\
    \$([ "$USE_GPU"  = "1" ] && echo "--gpu")
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
        -R "rusage[mem=16000] span[ptile=4]" \
        -o "${LOG_DIR}/${JOB_NAME}.%J.o" \
        -e "${LOG_DIR}/${JOB_NAME}.%J.e" \
        bash "$JOB_SCRIPT"
fi

echo "Job submitted. Logs: ${LOG_DIR}/${JOB_NAME}.<jobid>.{o,e}"
