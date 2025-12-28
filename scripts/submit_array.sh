#!/bin/bash
#SBATCH --job-name=neha_fit_array
#SBATCH --partition=any_cpu
#SBATCH --array=1-100%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=array_logs/fit_%a.out
#SBATCH --error=array_logs/fit_%a.err
#SBATCH --mail-type=ALL

echo "========================================================"
echo "STARTING Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "========================================================"

# --- 0. LOAD CONFIGURATION ---
source ./user_config_slurm.sh

# --- 1. DEFINE PATHS AND SCRATCH DIRECTORY ---
# PROJECT_HOME and IL6_HOME are now loaded from user_config_slurm.sh
SCRDIR="/scr/${SLURM_JOB_ID}"
mkdir -p "$SCRDIR"

# --- 2. SETUP TRAP COMMAND ---
# On exit, copy the single result file back.
trap 'rsync -av "$SCRDIR"/results/run_${SLURM_ARRAY_TASK_ID}*.jld2 "$PROJECT_HOME"/results/' EXIT

# --- 3. PREPARE DIRECTORIES IN YOUR HOME FOLDER ---
mkdir -p "$PROJECT_HOME"/array_logs
mkdir -p "$PROJECT_HOME"/results

# --- 4. COPY FILES TO SCRATCH ---
mkdir -p "$SCRDIR"/src
cp "$PROJECT_HOME"/src/run_single_task.jl "$SCRDIR"/src/
cp -r "$PROJECT_HOME"/petab_files "$SCRDIR"/
cp "$PROJECT_HOME"/variable_JAK_STAT_SOCS_degrad_model.net "$SCRDIR"/

# --- 5. EXECUTE THE JOB ---
cd "$SCRDIR"
export OPENBLAS_NUM_THREADS=1

# Note: Adjust Julia project path and sysimage as needed for your setup
# Use local scratch for Julia depot to avoid stale NFS cache issues
export JULIA_DEPOT_PATH="$SCRDIR/.julia:$HOME/.julia"

# Use IL6_TGFB sysimage since it has the same packages

julia --project="$IL6_HOME/bngl_julia" --sysimage="$IL6_HOME/SysImage/bngl_full.so" src/run_single_task.jl \
  --n-starts "$SLURM_ARRAY_TASK_MAX" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --max-iter 1000

echo "--- FINISHED TASK $SLURM_ARRAY_TASK_ID ---"
