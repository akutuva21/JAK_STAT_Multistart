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
ARRAY_JOBID=$(sbatch --mail-user="$EMAIL" scripts/submit_array.sh | awk '{print $4}')
echo "Submitted array job: $ARRAY_JOBID"

COLLATE_JOBID=$(sbatch --mail-user="$EMAIL" --dependency=afterok:${ARRAY_JOBID} scripts/submit_collate.sh | awk '{print $4}')
echo "Submitted collate job (afterok): $COLLATE_JOBID"


