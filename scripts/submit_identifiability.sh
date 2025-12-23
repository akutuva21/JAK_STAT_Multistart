#!/bin/bash
#SBATCH --job-name=neha_ident
#SBATCH --partition=any_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=31G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ident_%j.out
#SBATCH --error=logs/ident_%j.err
#SBATCH --mail-type=END,FAIL

echo "--- Starting Structural Identifiability Analysis on $(hostname) ---"

# --- 0. LOAD CONFIGURATION ---
source ./user_config_slurm.sh

# --- 1. DEFINE PATHS AND SCRATCH DIRECTORY ---
# PROJECT_HOME and IL6_HOME derived from config
SCRDIR="/scr/${SLURM_JOB_ID}"
mkdir -p "$SCRDIR"

# On exit, copy back results and clean up
trap 'rsync -av "$SCRDIR"/identifiability_results.txt "$PROJECT_HOME"/ 2>/dev/null || true; \
      rm -rf "$SCRDIR"' EXIT

# --- 2. PREPARE DIRECTORIES ---
mkdir -p "$PROJECT_HOME"/logs

# --- 3. COPY FILES TO SCRATCH ---
mkdir -p "$SCRDIR"/src
cp "$PROJECT_HOME"/src/structural_identifiability.jl "$SCRDIR"/src/
cp "$PROJECT_HOME"/variable_JAK_STAT_SOCS_degrad_model.net "$SCRDIR"/
cp -r "$PROJECT_HOME"/petab_files "$SCRDIR"/

# --- 4. MOVE INTO SCRATCH DIRECTORY ---
cd "$SCRDIR"

# --- 5. RUN THE ANALYSIS ---
echo "Running structural_identifiability.jl..."
# Note: Not using sysimage because StructuralIdentifiability isn't in it
julia --project="$IL6_HOME/bngl_julia" \
    src/structural_identifiability.jl

echo "--- Analysis Finished ---"
