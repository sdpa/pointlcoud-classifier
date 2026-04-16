#!/bin/bash
#SBATCH --export=NONE               
#SBATCH --job-name=pointcloud-classifier
#SBATCH --time=1-00:00:00           
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=24          
#SBATCH --mem=180G                  
#SBATCH --gres=gpu:a100:1           


module purge
module load GCCcore/12.2.0 Python/3.10.8 CUDA/11.3.1 

source venv/bin/activate

# ── helper: prints a banner + runs a command, reports wall time and exit code ──
run_timed() {
    local label="$1"
    shift
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  START  │ $label"
    echo "  CMD    │ $*"
    echo "  TIME   │ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"

    local start_ts
    start_ts=$(date +%s)

    "$@"
    local exit_code=$?

    local end_ts elapsed
    end_ts=$(date +%s)
    elapsed=$(( end_ts - start_ts ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    echo "════════════════════════════════════════════════════════════"
    echo "  END    │ $label"
    echo "  STATUS │ exit code $exit_code"
    echo "  WALL   │ ${mins}m ${secs}s  (${elapsed}s total)"
    echo "  TIME   │ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"
    echo ""

    return $exit_code
}

# ── job start banner ──────────────────────────────────────────────────────────
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

JOB_START=$(date +%s)

# ── training runs ─────────────────────────────────────────────────────────────
run_timed "PointNet   | epochs=1  batch=64" \
    python train.py --model pointnet --epochs 10 --batch_size 64

run_timed "Transformer | epochs=1 batch=64" \
    python train.py --model transformer --epochs 10 --batch_size 64

# ── job summary ───────────────────────────────────────────────────────────────
JOB_END=$(date +%s)
JOB_ELAPSED=$(( JOB_END - JOB_START ))
JOB_MINS=$(( JOB_ELAPSED / 60 ))
JOB_SECS=$(( JOB_ELAPSED % 60 ))

echo "════════════════════════════════════════════════════════════"
echo "  JOB COMPLETE"
echo "  Total wall time : ${JOB_MINS}m ${JOB_SECS}s  (${JOB_ELAPSED}s)"
echo "  Finished        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"