// src/solver/radau_kernel.cu

#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"               // for query_times / y0_all types
#include "radau_kernel.cuh"     // declaration of radau_kernel_multi
#include "radau_step_dense.cuh"       // radau_step() and radau_dense()
#include "event_detector.cuh"   // reuse norm_inf, norm_inf_diff if needed
//#include "models/active_model.hpp"   // defines Model204 and extern __constant__ devParams
#include "models/model_204.hpp" // brings in Model204
#include "I_O/forcing_data.h"



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
    int     n_stiff,
    const float*  d_forc_data,  // ★ new
    int nForc        // ★ new: number of forcings
)
    
  {
    constexpr int N_EQ = Model204::N_EQ;


    // Map thread → index in the stiff list
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_stiff) return;
    int sys_id = stiff_system_indices[idx];

    // Load tolerances and initial step from constant memory
    double my_rtol = devParams.rtol;
    double my_atol = devParams.atol;
    double h       = devParams.initialStep;
    double t       = t0;

    // Load initial condition & set up state buffers
    double y[N_EQ], y_next[N_EQ], k_dummy[7][N_EQ], error_norm;
    for (int i = 0; i < N_EQ; ++i) {
        y[i] = y0_all[sys_id * N_EQ + i];
    }

    // Index of next dense‐output query
    int next_q_idx = 0;

    // ─── Main implicit Radau loop ───
    while (t < tf) {
        // adjust final step if needed
        if (t + h > tf) h = tf - t;

        // ─── gather forcings *at current t* ───
        // pick the value corresponding to the current time t for each forcing j
        float Fval_arr[MAX_FORCINGS];
        for (int j = 0; j < nForc; ++j) {
            // compute where we are in the forcing time series as a real number
            double sampleIdxReal = t / c_forc_dt[j];
            // how many total time steps we loaded for this forcing
            size_t nSamples = c_forc_nT[j];
            // convert to an index between 0 and nSamples-1:
            // - if sampleIdxReal is below zero, use 0
            // - if it's beyond the last step, use the last index
            // - otherwise, drop the decimal part to get a whole number
            size_t sampleIdx = (sampleIdxReal < 0.0 
                                ? 0 
                                : sampleIdxReal >= nSamples 
                                    ? nSamples - 1 
                                    : size_t(sampleIdxReal));
            // calculate where forcing j’s block starts in the big array
            size_t base = size_t(j) * (nSamples * num_systems);
            // pick the value for this particular stream (thread) at time sampleIdx
            Fval_arr[j] = d_forc_data[ base + sampleIdx * num_systems + sys_id ];
        }

        // Take one Radau step (fills y_next and error_norm)
        Model204::rhs(t, y, k_dummy[0], N_EQ, sys_id, d_sp, Fval_arr, nForc);
        radau_step<Model204>(t, y, y_next, N_EQ, h, my_rtol, my_atol, &error_norm, k_dummy, sys_id, d_sp, d_forc_data, nForc);

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
    int,       // n_stiff
    const float*, // d_forc_data      // forcing data
    int         // nForc               // number of forcings
);
