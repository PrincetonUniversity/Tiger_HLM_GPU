#!/bin/bash

# Exit if any command fails
set -e

module load cudatoolkit/12.9


# for della-gpu 
# module load gcc-toolset/14
# module load aocc/5.0.0
# module load hdf5/aocc-5.0.0/1.14.4
# module load netcdf/aocc-5.0.0/hdf5-1.14.4/4.9.2

#for gh
module load openmpi/gcc/4.1.6
module load hdf5/gcc/openmpi-4.1.6/1.14.4
module load netcdf/gcc/hdf5-1.14.4/openmpi-4.1.6/4.9.2

# Run make
make clean
make Parallel=1

# ./rk45_solver 