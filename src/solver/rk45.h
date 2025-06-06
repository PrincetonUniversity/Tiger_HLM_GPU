#ifndef RK45_H
#define RK45_H

#include <cuda_runtime.h>

/**
 * GPU kernel that runs a single Dormand–Prince RK45 integrator on each thread.
 *
 * Each thread integrates one system of N_EQ coupled ODEs from t₀ to t_f.  The
 * caller already sets Model::devParams (initial step, rtol, atol, etc.)
 * in constant memory via setModelParameters<Model>().  This kernel:
 *
 *   1. Loads the initial state y[0..N_EQ-1] for system sys_id.
 *   2. Computes a trial step h₀ exactly as SciPy does to rescale the first step.
 *   3. Enters the main RK45 loop, performing adaptive steps until t ≥ t_f.
 *   4. Fills out dense_all[sys_id, i, q] whenever a query time q ∈ (t, t_next]
 *      falls inside the accepted step.
 *   5. Writes the final state y(t_f) into y_final_all[sys_id, i].
 *
 * Template parameters:
 *   Model::N_EQ       → number of equations per system
 *   Model::rhs        → device RHS function for evaluating slopes
 *   Model::devParams  → holds rtol, atol, initialStep, safety, minScale, maxScale
 *
 * Arguments:
 *   y0_all       : pointer to host-copied array of length (num_systems × N_EQ),
 *                  containing each system’s initial condition y(t₀).
 *   y_final_all  : pointer to device buffer of the same size, to receive y(t_f).
 *   query_times  : pointer to an array of length num_queries, sorted ascending,
 *                  giving the times at which dense output is requested.
 *   dense_all    : pointer to device buffer of size (num_systems × N_EQ × num_queries);
 *                  for each (sys_id, i, q), we store y_i at query_times[q].
 *   num_systems  : number of independent systems being integrated in parallel.
 *   num_queries  : number of dense-output timestamps per system.
 *   t0           : initial time of integration.
 *   tf           : final time of integration.
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
