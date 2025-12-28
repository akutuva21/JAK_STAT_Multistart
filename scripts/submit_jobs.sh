#!/bin/bash
# submit_jobs.sh - Helper to submit jobs using configuration

source ./user_config_slurm.sh

if [ -z "$EMAIL" ]; then
    echo "Error: EMAIL not set in user_config_slurm.sh"
    exit 1
fi

echo "Submitting jobs for user: $EMAIL"

# Usage examples (uncomment or run manually):
# sbatch --mail-user="$EMAIL" submit_array.sh
# sbatch --mail-user="$EMAIL" --dependency=afterok:<JOBID> submit_collate.sh

# echo "To submit the array job, run:"
# echo "sbatch --mail-user=\"$EMAIL\" submit_array.sh"

sbatch --mail-user="$EMAIL" scripts/submit_array.sh
sbatch --mail-user="$EMAIL" --dependency=afterok:<JOBID> scripts/submit_collate.sh


