TOTAL_GENES=17888
GENES_PER_JOB=500

NUM_JOBS=$(( (TOTAL_GENES + GENES_PER_JOB - 1) / GENES_PER_JOB ))
OUTPUT_DIR="$1"

echo "Total genes: $TOTAL_GENES"
echo "Genes per job: $GENES_PER_JOB"
echo "Number of jobs: $NUM_JOBS"
echo "Output dir: $OUTPUT_DIR"

echo "Submitting job ${job_i}/${NUM_JOBS}"
# Submit jobs
for ((job_i=0; job_i<NUM_JOBS; job_i++)); do

    OUTPUT_FILE="$OUTPUT_DIR/lfc_df_job${job_i}.tsv"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping job ${job_i} (output exists)"
        continue
    fi

    echo "Submitting job ${job_i}/${NUM_JOBS}"

    bsub -J gene_job_${job_i} \
        -q medium \
        -n 4 \
        -gpu "num=1:mode=exclusive_process" \
        -R "rusage[mem=16000] span[ptile=4] select[gpu]" \
        -o /home/labs/davidgo/itamarn/log/AG_gene_pred_log/gene_job_${job_i}.%J.o \
        -e /home/labs/davidgo/itamarn/log/AG_gene_pred_log/gene_job_${job_i}.%J.e \
        python "/home/labs/davidgo/itamarn/backup/MSc/XGBoost_mpra_to_gene_expression/ag_predictions.py" \
        --job_i "$job_i" \
        --chunk_size "$GENES_PER_JOB" \
        --output_dir "$OUTPUT_DIR" \
        $EXTRA_ARGS

done

echo "hi"
echo "All jobs submitted!"
