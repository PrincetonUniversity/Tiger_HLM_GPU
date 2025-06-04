// src/solver/rk45.h
#ifndef RK45_H
#define RK45_H

#include <cuda_runtime.h>

/**
 * Templated GPU kernel: Batch‐mode RK45 integrator.
 *
 * Each Model must satisfy:
 *   • static constexpr int    N_EQ         // number of equations (1…10)
 *   • static constexpr unsigned short UID  // unique model ID
 *   • struct Parameters { double initialStep, safety, minScale, maxScale; }
 *   • __device__ static void rhs(double t, const double* y, double* dydt, int n)
 *   • extern __constant__ Model::Parameters devParams;
 *
 * On launch, each thread integrates one N_EQ‐dimensional system.
 *
 * Arguments:
 *   y0_all        [ num_systems × N_EQ ]  initial states
 *   y_final_all   [ num_systems × N_EQ ]  final states at t=tf
 *   query_times   [ num_queries ]         sorted query times in (t0,tf)
 *   dense_all     [ num_systems × N_EQ × num_queries ]  dense output
 *   num_systems, num_queries, t0, tf
 */
template <class Model>
__global__ void rk45_kernel_multi(
    double* y0_all,
    double* y_final_all,
    double* query_times,
    double* dense_all,
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf
);

#endif  // RK45_H
