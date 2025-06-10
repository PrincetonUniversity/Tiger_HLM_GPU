//solver/radau_kernel.cuh
#pragma once

// Forward‐declaration of the implicit Radau IIA multi‐system kernel
// (parallel to rk45_kernel_multi).
// Must match the signature in radau_kernel.cu.

template <class Model204>
__global__ void radau_kernel_multi(
    double* y0_all,
    double* y_final_all,
    double* query_times,
    double* dense_all,
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf,
    const typename Model204::SP_TYPE* d_sp,
    int*    stiff_system_indices,
    int     n_stiff
);




