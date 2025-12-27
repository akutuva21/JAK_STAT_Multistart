#!/bin/bash
#SBATCH --job-name=neha_collate
#SBATCH --partition=any_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --output=logs/collate_%j.out
#SBATCH --error=logs/collate_%j.err
#SBATCH --mail-type=END,FAIL

echo "--- Starting Collation & Analysis on $(hostname) ---"

# --- 0. LOAD CONFIGURATION ---
source ./user_config_slurm.sh

# --- 1. DEFINE PATHS AND SCRATCH DIRECTORY ---
# PROJECT_HOME and IL6_HOME derived from config
SCRDIR="/scr/${SLURM_JOB_ID}"
mkdir -p "$SCRDIR"

# Results directory
RESULTS_DIR="$PROJECT_HOME/results/sameaspaper"

# On exit, copy back all results and clean up
trap 'rsync -av "$SCRDIR"/src/best_parameters.csv "$RESULTS_DIR"/; \
      rsync -av "$SCRDIR"/src/optimization_summary.csv "$RESULTS_DIR"/; \
      rsync -av "$SCRDIR"/src/best_fit.jld2 "$RESULTS_DIR"/ 2>/dev/null || true; \
      rsync -av "$SCRDIR"/results/sameaspaper/final_results_plots/ "$RESULTS_DIR"/final_results_plots/ 2>/dev/null || true; \
      rsync -av "$SCRDIR"/results/sameaspaper/likelihood_profiles/ "$RESULTS_DIR"/likelihood_profiles/ 2>/dev/null || true; \
      rm -rf "$SCRDIR"' EXIT

# --- 2. PREPARE DIRECTORIES IN YOUR HOME FOLDER ---
mkdir -p "$PROJECT_HOME"/logs
mkdir -p "$RESULTS_DIR"/final_results_plots
mkdir -p "$RESULTS_DIR"/likelihood_profiles

# --- 3. COPY ALL NECESSARY FILES TO SCRATCH ---
mkdir -p "$SCRDIR"/src
cp "$PROJECT_HOME"/src/collect_results.jl "$SCRDIR"/src/
cp -r "$RESULTS_DIR" "$SCRDIR"/results/ 2>/dev/null || mkdir -p "$SCRDIR"/results
cp -r "$PROJECT_HOME"/petab_files "$SCRDIR"/
cp "$PROJECT_HOME"/variable_JAK_STAT_SOCS_degrad_model.net "$SCRDIR"/

# --- 4. MOVE INTO SCRATCH DIRECTORY ---
cd "$SCRDIR"

# --- 5. RUN THE SCRIPT WITH --ident FLAG ---
# Use IL6_TGFB project and sysimage for package availability
julia --project="$IL6_HOME/bngl_julia" --sysimage="$IL6_HOME/SysImage/bngl_full.so" \
    src/collect_results.jl --ident

echo "--- Collation & Analysis Finished ---"
