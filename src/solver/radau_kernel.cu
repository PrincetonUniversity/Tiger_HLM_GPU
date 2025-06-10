// src/solver/radau_kernel.cu

#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"               // for query_times / y0_all types
#include "radau_kernel.cuh"     // declaration of radau_kernel_multi
#include "radau_step_dense.cuh"       // radau_step() and radau_dense()
#include "event_detector.cuh"   // reuse norm_inf, norm_inf_diff if needed
//#include "models/active_model.hpp"   // defines Model204 and extern __constant__ devParams
#include "models/model_204.hpp" // brings in Model204

// -----------------------------------------------------------------------------
// Implementation of radau_kernel_multi defined in radau_kernel.cuh
// -----------------------------------------------------------------------------
// This kernel integrates multiple systems in parallel using the Radau IIA method.

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
)
    
  {
    constexpr int N_EQ = Model204::N_EQ;

    // // Identify which system this thread integrates
    // int sys_id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (sys_id >= num_systems) return;

    // if (idx == 0) {
    // printf("Radau kernel running on systems: ");
    // for (int i = 0; i < n_stiff; ++i) printf("%d ", stiff_system_indices[i]);
    // printf("\n");
    // }


    // ─── DEBUG CHECK: only print once from sys_id == 0 ───
    // if (sys_id == 0) {
    //     printf("RADAU KERNEL: num_systems = %d, num_queries = %d\n", num_systems, num_queries);
    // }
    // ─────────────────────────────────────────────────────────

    // 1) Map thread → index in the stiff list
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_stiff) return;
    int sys_id = stiff_system_indices[idx];

    // 2) One‐time debug print
    if (idx == 0) {
        printf("Radau on systems:");
        for (int i = 0; i < n_stiff; ++i) printf(" %d", stiff_system_indices[i]);
        printf("\n");
    }

    // Load tolerances and initial step from constant memory
    double my_rtol = devParams.rtol;
    double my_atol = devParams.atol;
    double h       = devParams.initialStep;

    // ─── Integration state ───
    double t = t0;

    // Allocate registers for state, next state, and a dummy k
    double y[N_EQ], y_next[N_EQ];
    // we don't use explicit k, but radau_step expects a dummy array
    double k_dummy[7][N_EQ];
    double error_norm = 0.0;

    // Load initial condition
    for (int i = 0; i < N_EQ; ++i) {
        y[i] = y0_all[sys_id * N_EQ + i];
    }

    // Index of next dense‐output query
    int next_q_idx = 0;

    // ─── Main implicit Radau loop ───
    while (t < tf) {
        // adjust final step if needed
        if (t + h > tf) h = tf - t;

        // Take one Radau step (fills y_next and error_norm)
        radau_step<Model204>(t, y, y_next, N_EQ, h, my_rtol, my_atol, &error_norm, k_dummy, sys_id, d_sp);

        // Accept or reject based on embedded error
        if (error_norm <= 1.0) {
            // ─── Dense output for queries in (t, t+h] ───
            double t_next = t + h;
            while (next_q_idx < num_queries && query_times[next_q_idx] <= t_next) {
                double tq = query_times[next_q_idx];
                if (tq > t) {
                    double theta = (tq - t) / h;
                    double y_dense[N_EQ];
                    //radau_dense<Model204>(y, /*Z unused here*/ *(double(*)[N_EQ])k_dummy,
                    //                         N_EQ, h, theta, y_dense);
                    radau_dense<Model204>(y,(double (*)[N_EQ]) k_dummy,N_EQ, h, theta, y_dense);
                    // store
                    for (int comp = 0; comp < N_EQ; ++comp) {
                        int idx = sys_id*(N_EQ*num_queries)
                                + comp*(num_queries)
                                + next_q_idx;
                        dense_all[idx] = y_dense[comp];
                    }
                }
                ++next_q_idx;
            }

            // Advance
            for (int i = 0; i < N_EQ; ++i) {
                y[i] = y_next[i];
            }
            t = t_next;

            // Rescale h: stiff‐controller (e.g. simple power‐law)
            double fac = devParams.safety * pow(1.0 / (error_norm + 1e-16), 1.0/5.0);
            h *= fmin(fmax(fac, devParams.minScale), devParams.maxScale);

        } else {
            // Reject: shrink h and retry
            double fac = devParams.safety * pow(1.0 / (error_norm + 1e-16), 1.0/5.0);
            fac = fmin(fac, 1.0);
            fac = fmin(fmax(fac, devParams.minScale), devParams.maxScale);
            h *= fac;
            // loop will retry
        }
    }

    // Write final state
    for (int i = 0; i < N_EQ; ++i) {
        y_final_all[sys_id * N_EQ + i] = y[i];
    }
}

// ----------------------------------------------------------------------------
// Explicit instantiation of radau_kernel_multi for Model204
// ----------------------------------------------------------------------------
template __global__ void radau_kernel_multi<Model204>(
    double*, double*, double*, double*,
    int, int, double, double,
    const typename Model204::SP_TYPE*, 
    int*,     // stiff_system_indices
    int       // n_stiff
);

