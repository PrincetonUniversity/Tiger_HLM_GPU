#!/bin/bash
#SBATCH --account=gvillari #account to charge to 
#SBATCH --job-name=test  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2     # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G is default)
#SBATCH --time=00:10:00          # maximum time needed (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --partition=grace
#SBATCH --gres=gpu:2

module load cudatoolkit/12.9
module load openmpi/gcc/4.1.6
module load hdf5/gcc/openmpi-4.1.6/1.14.4
module load netcdf/gcc/hdf5-1.14.4/openmpi-4.1.6/4.9.2

srun ./rk45_solver 