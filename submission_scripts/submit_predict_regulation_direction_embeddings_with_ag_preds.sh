#!/bin/bash
# Submit binary classification job predicting up/down regulation from
# paired human/chimp AlphaGenome exon-position embeddings.
# Models: LogisticRegression (baseline) + XGBoost + FCNet.
#
# Usage:
#   bash submit_predict_regulation_direction_embeddings_with_ag_preds.sh
#
# Examples:
#   bash submit_predict_regulation_direction_embeddings_with_ag_preds.sh
#   ASE_ONLY=1 bash submit_predict_regulation_direction_embeddings_with_ag_preds.sh
#   LFC_THRESHOLD=0.5 bash submit_predict_regulation_direction_embeddings_with_ag_preds.sh
#   MAX_POSITIONS=500 USE_GPU=1 bash submit_predict_regulation_direction_embeddings_with_ag_preds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/predict_regulation_direction_embeddings_with_ag_preds.py"

HUMAN_EMB_DIR=${HUMAN_EMB_DIR:-"$SCRIPT_DIR/../create_AG_embeddings/embeddings/human"}
CHIMP_EMB_DIR=${CHIMP_EMB_DIR:-"$SCRIPT_DIR/../create_AG_embeddings/embeddings/chimp"}
LFC_THRESHOLD=${LFC_THRESHOLD:-0.0}
ASE_ONLY=${ASE_ONLY:-0}
INCLUDE_DIFF=${INCLUDE_DIFF:-0}
MAX_POSITIONS=${MAX_POSITIONS:-""}
USE_GPU=${USE_GPU:-0}
HIDDEN_DIM=${HIDDEN_DIM:-256}
EPOCHS=${EPOCHS:-300}
LOG_DIR=${LOG_DIR:-/home/labs/davidgo/itamarn/log/ag_preds_emb_clf_log}
QUEUE=${QUEUE:-short}
CONDA_ENV=${CONDA_ENV:-mpra_model_env}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}

mkdir -p "$LOG_DIR"

JOB_NAME="ag_emb_clf"

echo "Submitting paired embeddings up/down classification job"
echo "  Script       : $SCRIPT_PATH"
echo "  Human emb    : $HUMAN_EMB_DIR"
echo "  Chimp emb    : $CHIMP_EMB_DIR"
echo "  LFC thresh   : $LFC_THRESHOLD"
echo "  ASE only     : $ASE_ONLY"
echo "  Include diff : $INCLUDE_DIFF"
echo "  Max positions: ${MAX_POSITIONS:-all}"
echo "  Use GPU      : $USE_GPU"
echo "  Hidden dim   : $HIDDEN_DIM"
echo "  Epochs       : $EPOCHS"
echo "  Queue        : $QUEUE"

JOB_SCRIPT=$(mktemp "${LOG_DIR}/${JOB_NAME}.XXXXXX.sh")
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
source "$CONDA_SH" && conda activate "$CONDA_ENV" &&
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}" &&
cd "$SCRIPT_DIR" &&
python "$SCRIPT_PATH" \\
    --human-emb-dir "$HUMAN_EMB_DIR" \\
    --chimp-emb-dir "$CHIMP_EMB_DIR" \\
    --lfc-threshold "$LFC_THRESHOLD" \\
    --hidden-dim "$HIDDEN_DIM" \\
    --epochs "$EPOCHS" \\
    \$([ -n "$MAX_POSITIONS" ] && echo "--max-positions-per-gene $MAX_POSITIONS") \\
    \$([ "$ASE_ONLY"     = "1" ] && echo "--ase-only") \\
    \$([ "$INCLUDE_DIFF" = "1" ] && echo "--include-diff") \\
    \$([ "$USE_GPU"      = "1" ] && echo "--gpu")
EOF

if [ "$USE_GPU" = "1" ]; then
    bsub -J "$JOB_NAME" \
        -q "${QUEUE}-gpu" \
        -n 4 \
        -gpu "num=1:mode=exclusive_process" \
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
