from pathlib import Path

from generation_projects.blimp.registry import generator_stems

top = """#!/bin/bash

# Generic job script for all experiments.

#SBATCH --cpus-per-task=7
#SBATCH --mem=16GB
#SBATCH -t24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=alexwarstadt@gmail.com

#PRINCE PRINCE_GPU_COMPUTE_MODE=default

# Log what we're running and where.
#echo $SLURM_JOBID - `hostname` - $SPINN_FLAGS >> ~/spinn_machine_assignments.txt

# Make sure we have access to HPC-managed libraries.



# Run.


cd ~/data_generation
python -m generation_projects.blimp.%s"""

project_root = Path(__file__).resolve().parents[3]
slurm_dir = project_root / "slurm"
slurm_dir.mkdir(exist_ok=True)
for stem in generator_stems():
    with open(slurm_dir / ("%s.sbatch" % stem), "w") as output_file:
        output_file.write(top % stem)
