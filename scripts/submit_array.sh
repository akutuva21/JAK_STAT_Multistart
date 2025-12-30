#!/bin/bash
#SBATCH --job-name=neha_fit_array
#SBATCH --partition=any_cpu
#SBATCH --array=1-200%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=array_logs/fit_%a.out
#SBATCH --error=array_logs/fit_%a.err
#SBATCH --mail-type=ALL

set -euo pipefail
shopt -s nullglob

# Allow running as a normal (non-array) job for debugging.
# If this is a real array job, Slurm will provide these variables.
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-1}}"
N_STARTS="${SLURM_ARRAY_TASK_MAX:-${N_STARTS:-1}}"

echo "========================================================"
echo "STARTING Job ID: $SLURM_JOB_ID, Task ID: $TASK_ID on $(hostname)"
echo "========================================================"

# --- 0. LOAD CONFIGURATION ---
source ./user_config_slurm.sh

# --- 1. DEFINE PATHS AND SCRATCH DIRECTORY ---
# PROJECT_HOME and IL6_HOME are now loaded from user_config_slurm.sh
SCRDIR="/scr/${SLURM_JOB_ID}"
mkdir -p "$SCRDIR"

# --- 2. SETUP TRAP COMMAND ---
# On exit, copy the single result file back (if it exists).
trap 'files=("$SCRDIR"/results/run_'"$TASK_ID"'*.jld2); if [ ${#files[@]} -gt 0 ]; then rsync -av "${files[@]}" "$PROJECT_HOME"/results/; else echo "No result file found to rsync for task '"$TASK_ID"'"; fi' EXIT

# --- 3. PREPARE DIRECTORIES IN YOUR HOME FOLDER ---
mkdir -p "$PROJECT_HOME"/array_logs
mkdir -p "$PROJECT_HOME"/results

# --- 4. COPY FILES TO SCRATCH ---
mkdir -p "$SCRDIR"/src
cp "$PROJECT_HOME"/src/run_single_task.jl "$SCRDIR"/src/
cp "$PROJECT_HOME"/src/python_compatible_nllh.jl "$SCRDIR"/src/
cp -r "$PROJECT_HOME"/petab_files "$SCRDIR"/
cp "$PROJECT_HOME"/variable_JAK_STAT_SOCS_degrad_model.net "$SCRDIR"/

# --- 5. EXECUTE THE JOB ---
cd "$SCRDIR"
export OPENBLAS_NUM_THREADS=1

# Note: Adjust Julia project path and sysimage as needed for your setup
# Use local scratch for Julia depot to avoid stale NFS cache issues
export JULIA_DEPOT_PATH="$SCRDIR/.julia:$HOME/.julia"

echo "Running Julia (task-id=$TASK_ID, n-starts=$N_STARTS)"
if julia "--project=$IL6_HOME/bngl_julia" -J "$IL6_HOME/SysImage/bngl_full.so" src/run_single_task.jl \
  --n-starts "$N_STARTS" \
  --task-id "$TASK_ID" \
  --max-iter 1000; then
  :
else
  rc=$?
  echo "Julia failed with exit code: $rc"
  exit $rc
fi

echo "Julia finished successfully"

echo "--- FINISHED TASK $TASK_ID ---"
