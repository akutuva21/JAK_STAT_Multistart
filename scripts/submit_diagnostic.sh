#!/bin/bash
#SBATCH --job-name=neha_diagnostic
#SBATCH --partition=any_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diagnostic_%j.out
#SBATCH --error=logs/diagnostic_%j.err
#SBATCH --mail-type=ALL

echo "========================================================"
echo "STARTING Diagnostic Job ID: $SLURM_JOB_ID on $(hostname)"
echo "========================================================"

# --- 0. LOAD CONFIGURATION ---
source ./user_config_slurm.sh

# --- 1. DEFINE PATHS AND SCRATCH DIRECTORY ---
SCRDIR="/scr/${SLURM_JOB_ID}"
mkdir -p "$SCRDIR"

# --- 2. SETUP TRAP COMMAND ---
# On exit, copy output file back
trap 'rsync -av "$SCRDIR"/diagnostic_output.txt "$PROJECT_HOME"/logs/ 2>/dev/null || true' EXIT

# --- 3. PREPARE DIRECTORIES ---
mkdir -p "$PROJECT_HOME"/logs

# --- 4. COPY FILES TO SCRATCH ---
mkdir -p "$SCRDIR"/src
cp "$PROJECT_HOME"/src/diagnostic_model.jl "$SCRDIR"/src/
cp -r "$PROJECT_HOME"/petab_files "$SCRDIR"/
cp "$PROJECT_HOME"/variable_JAK_STAT_SOCS_degrad_model.net "$SCRDIR"/
cp "$PROJECT_HOME"/best_parameters.csv "$SCRDIR"/ 2>/dev/null || echo "No best_parameters.csv found"

# --- 5. EXECUTE THE JOB ---
cd "$SCRDIR"
export OPENBLAS_NUM_THREADS=1
export JULIA_DEPOT_PATH="$SCRDIR/.julia:$HOME/.julia"

echo "Running diagnostic..."
julia --project="$IL6_HOME/bngl_julia" --sysimage="$IL6_HOME/SysImage/bngl_full.so" src/diagnostic_model.jl 2>&1 | tee diagnostic_output.txt

echo "--- DIAGNOSTIC COMPLETE ---"
