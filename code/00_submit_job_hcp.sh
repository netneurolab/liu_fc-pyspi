#!/bin/bash -x

#SBATCH --time=72:00:00
#SBATCH --account=def-misic176
#SBATCH --job-name=pyspi-hcp
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

# Save some useful information to the "output" file
echo "================================================="
if [[ -z "${SLURM_ARRAY_JOB_ID}" ]]; then
    echo "Job ID: $SLURM_JOB_ID"
else
    echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
    echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
fi
echo "Start time: $(date)"
echo "Host name: $(hostname)"
echo "Working dir: $(pwd)"
echo "================================================="

module load apptainer

export PYTHONUNBUFFERED=TRUE
export APPTAINERENV_PYTHONUNBUFFERED=TRUE

CURR_WD="/scratch/zqliu/temp_code/202308-pyspi-rerun"

subj_id=$( sed "${SLURM_ARRAY_TASK_ID}q;d" ${CURR_WD}/data/Reinder327_subjects.txt )
parc_str=schaefer400x7

cmd="\
apptainer exec \
    --cleanenv \
    -B /home -B /scratch -B /localscratch \
    /scratch/zqliu/singularity/pyspi-0.4.1-optim.sif \
        python ${CURR_WD}/run_pyspi_hcp.py ${subj_id} ${parc_str}
"

eval $cmd
exitcode=$?
exit $exitcode