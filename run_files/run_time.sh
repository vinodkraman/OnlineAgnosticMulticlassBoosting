#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=example_job
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=72:00:00
#SBATCH --account=tewaria0
#SBATCH --partition=standard

# The application(s) to execute along with its input arguments and options:
python run_files/run_time.py --num_wl 100 --filename data/balance-scale --noise 0.0 --exp 3 --leaf nba --dep 1 --nom_att 0,1,2,3 --model all
