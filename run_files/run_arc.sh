#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3000m
#SBATCH --time=48:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=standard

# The application(s) to execute along with its input arguments and options:
python run_batch_file_2.py --T 100 --filename data/cars_correct --noise 0.30