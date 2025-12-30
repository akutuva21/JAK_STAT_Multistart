# JAK-STAT Multistart Optimization

This repository contains the codebase for running multistart parameter estimation and identifiability analysis for the JAK-STAT-SOCS degradation model.

## Directory Structure

| Directory/File | Description |
| :--- | :--- |
| `src/` | Julia source code for optimization, diagnostics, and results collection. |
| `scripts/` | SLURM submission scripts (Bash) for cluster execution. |
| `Data/` | Input data (pTempest trajectories, pooled experimental data). |
| `petab_files/` | PEtab model definition (conditions, measurements, observables, parameters). |
| `STAT_models/` | BNGL/Net model variants and prediction scripts (submodule). |
| `old/` | Archived/deprecated scripts. |
| `variable_JAK_STAT_SOCS_degrad_model.bngl` | Main BNGL model file. |
| `variable_JAK_STAT_SOCS_degrad_model.net` | Generated network file. |

### Source Files (`src/`)

| File | Description |
| :--- | :--- |
| `run_single_task.jl` | Runs a single optimization task (used by SLURM array). |
| `run_multistart_from_files.jl` | Runs multistart optimization from pre-generated parameter files. |
| `collect_results.jl` | Aggregates results and generates summary plots. |
| `compare_results.jl` | Compares results across different optimization runs. |
| `diagnostic_model.jl` | Model diagnostics and validation utilities. |
| `inspect_petab.jl` | Utilities for inspecting PEtab problem setup. |

### Scripts (`scripts/`)

| File | Description |
| :--- | :--- |
| `submit_array.sh` | Main SLURM array job submission script. |
| `submit_collate.sh` | Collates results from array jobs. |
| `submit_compare.sh` | Submits comparison analysis job. |
| `submit_diagnostic.sh` | Submits model diagnostic job. |
| `submit_identifiability.sh` | Submits identifiability analysis job. |
| `submit_jobs.sh` | Utility script for batch job submission. |

### Data Files (`Data/`)

| File | Description |
| :--- | :--- |
| `pSTAT1_trajs.csv` | pTempest trajectory samples for pSTAT1. |
| `pSTAT3_trajs.csv` | pTempest trajectory samples for pSTAT3. |
| `pSTAT1_pooled_data.xlsx` | Pooled experimental data for pSTAT1. |
| `pSTAT3_pooled_data.xlsx` | Pooled experimental data for pSTAT3. |
| `param_sets.csv` | pTempest parameter sets (used only for comparison/benchmarking). |

### PEtab Files (`petab_files/`)

| File | Description |
| :--- | :--- |
| `conditions.tsv` | Experimental conditions (ligand concentrations). |
| `measurements.tsv` | Experimental measurement data. |
| `observables.tsv` | Observable definitions. |
| `parameters.tsv` | Parameter definitions, bounds, and priors. |

## System Overview

The pipeline consists of three main stages:

1. **Optimization**: Runs hundreds of independent fits in parallel (SLURM Array).
2. **Collation**: Aggregates results to find the global minimum.
3. **Identifiability**: Performs structural identifiability analysis using Hessian eigenvalues.

## Configuration & Execution

1. **Setup**: Create `user_config_slurm.sh` with your paths and email:
   ```bash
   export MY_EMAIL="your-email@example.com"
   export PROJECT_DIR="/path/to/NehaMultistart"
   ```

2. **Run Optimization**: Submit the array job:
   ```bash
   sbatch scripts/submit_array.sh
   ```

3. **Collate Results**: After all tasks complete:
   ```bash
   sbatch scripts/submit_collate.sh
   # or locally:
   julia src/collect_results.jl
   ```

4. **Analyze**: Run comparison or diagnostics as needed:
   ```bash
   sbatch scripts/submit_compare.sh
   sbatch scripts/submit_diagnostic.sh
   ```
