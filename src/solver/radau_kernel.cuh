//solver/radau_kernel.cuh
#pragma once

// Forward‐declaration of the implicit Radau IIA multi‐system kernel
// (parallel to rk45_kernel_multi).
// Must match the signature in radau_kernel.cu.

template <class ActiveModel>
__global__ void radau_kernel_multi(
    double* y0_all,        // [num_systems × ActiveModel::N_EQ]: initial states
    double* y_final_all,   // [num_systems × ActiveModel::N_EQ]: final states
    double* query_times,   // [num_queries]: sorted dense-output times
    double* dense_all,     // [num_systems × ActiveModel::N_EQ × num_queries]: dense outputs
    int     num_systems,   // total number of systems
    int     num_queries,   // total number of query times
    double  t0,            // start time
    double  tf             // end time
);
