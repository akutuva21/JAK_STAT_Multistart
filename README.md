# Neha Multistart Optimization

This repository contains the codebase for running multistart parameter estimation and identifiability analysis for the JAK-STAT-SOCS degradation model.

## Directory Structure

| Directory | Description |
| :--- | :--- |
| `src/` | Julia source code (`collect_results.jl`, `run_multistart.jl`, etc.) |
| `scripts/` | SLURM submission scripts (Bash) for cluster execution. |
| `data/` | Input data (pTempest trajectories, parameters). |
| `results/` | Output directory for optimization runs (local only). |
| `petab_files/` | PEtab model definition. |
| `STAT_models/` | BNGL/Net formats of the model. |

## System Overview

The pipeline consists of three main stages:
1.  **Optimization**: Runs hundreds of independent fits in parallel (SLURM Array).
2.  **Collation**: Aggregates results to find the global minimum.
3.  **Identifiability**: Performs structural identifiability analysis using Hessian eigenvalues.

## Configuration & Execution

1.  **Setup**: Create `user_config_slurm.sh` with your paths (see `scripts/` for examples).
2.  **Run**: Use `sbatch scripts/submit_array.sh` to start the optimization array.
3.  **Analyze**: Run `julia src/collect_results.jl` to generate plots and reports.
