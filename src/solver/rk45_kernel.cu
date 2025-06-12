// src/solver/rk45_kernel.cu

#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"
#include "rk45_step_dense.cuh"
#include "event_detector.cuh"
#include "small_lu.cuh"
#include "radau_step_dense.cuh"
#include "models/model_204.hpp"
#include "I_O/forcing_data.h"


// -----------------------------------------------------------------------------
// Single‐kernel that does RK45 and flags stiffness, but no in‐kernel Radau.
// -----------------------------------------------------------------------------
template <class Model204>
__global__ void rk45_then_radau_multi(
    double* y0_all,       // [num_systems × N_EQ]
    double* y_final_all,  // [num_systems × N_EQ]
    double* query_times,  // [num_queries]
    double* dense_all,    // [num_systems × N_EQ × num_queries]
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf,
    const typename Model204::SP_TYPE* d_sp,
    int*    d_stiff,       // new: flag array
    const float*  d_forc_data,  // ★ new
    int           nForc       // ★ new: number of forcings
) {
    constexpr int N_EQ = Model204::N_EQ;
    int sys = blockIdx.x*blockDim.x + threadIdx.x;
    if (sys >= num_systems) return;

    // solver parameters
    double rtol = devParams.rtol;
    double atol = devParams.atol;
    // double h    = devParams.initialStep;
    // double t    = t0;


    // ─── State & forcing buffer ───
    double y[N_EQ], y_next[N_EQ], k45[7][N_EQ], err;
    for(int i=0; i<N_EQ; ++i) y[i] = y0_all[sys*N_EQ + i];

    int next_q = 0, reject_count = 0;
    bool stiff = false;

    // ─── 1) Explicit RK45 phase ───
    double t = t0;
    double h = devParams.initialStep;
    while(t < tf && !stiff) {
        if (t + h > tf) h = tf - t;

        // ─── gather forcings *at current t* ───
        // pick the value corresponding to the current time t for each forcing j
        float Fval_arr[MAX_FORCINGS];
        // for (int j = 0; j < nForc; ++j) {
        //     // compute where we are in the forcing time series as a real number
        //     // double sampleIdxReal = t / c_forc_dt[j];

        //     // convert this forcing’s dt (hours) → dt_min (minutes)
        //     double dt_min        = c_forc_dt[j] * 60.0;
        //     double sampleIdxReal = t / dt_min;

        //     // how many total time steps we loaded for this forcing
        //     size_t nSamples = c_forc_nT[j];
        //     // convert to an index between 0 and nSamples-1:
        //     // - if sampleIdxReal is below zero, use 0
        //     // - if it's beyond the last step, use the last index
        //     // - otherwise, drop the decimal part to get a whole number
        //     size_t sampleIdx = (sampleIdxReal < 0.0 
        //                         ? 0 
        //                         : sampleIdxReal >= nSamples 
        //                             ? nSamples - 1 
        //                             : size_t(sampleIdxReal));
        //     // calculate where forcing j’s block starts in the big array
        //     size_t base = size_t(j) * (nSamples * num_systems);
        //     // pick the value for this particular stream (thread) at time sampleIdx
        //     Fval_arr[j] = d_forc_data[ base + sampleIdx * num_systems + sys ];
        // }

        // ─── gather forcings at current t ───
        for (int j = 0; j < nForc; ++j) {
            // OLD (wrong): sampleIdxReal in hours
            // double sampleIdxReal = t / c_forc_dt[j];

            // NEW: convert dt from hours to minutes
            double dt_min        = c_forc_dt[j] * 60.0;
            double sampleIdxReal = t / dt_min;

            size_t nSamples = c_forc_nT[j];
            size_t sampleIdx = (sampleIdxReal < 0.0
                                ? 0
                                : sampleIdxReal >= nSamples
                                    ? nSamples - 1
                                    : size_t(sampleIdxReal));

            // compute the correct block start by summing the previous blocks
            size_t base = 0;
            for (int k = 0; k < j; ++k) {
                base += c_forc_nT[k] * size_t(num_systems);
            }
            Fval_arr[j] = d_forc_data[ base + sampleIdx * size_t(num_systems) + sys ];


            //size_t base = size_t(j) * (nSamples * num_systems);
            //Fval_arr[j] = d_forc_data[ base + sampleIdx * num_systems + sys ];
        }
        

        // pass forcings into RHS
        Model204::rhs(t, y, k45[0], N_EQ, sys, d_sp, Fval_arr, nForc);
        //Model204::rhs(t, y, k45[0], N_EQ, sys, d_sp);
        rk45_step<Model204>(t, y, y_next, N_EQ, h, rtol, atol, &err, k45, sys, d_sp, Fval_arr, nForc);

    // int next_q = 0, reject_count = 0;
    // bool stiff = false;

    // // ─── 1) Explicit RK45 phase ───
    // while(t < tf && !stiff) {
    //     if (t + h > tf) h = tf - t;
    //     //Model204::rhs(t, y, k45[0], N_EQ, sys, d_sp);
    //     Model204::rhs(t, y, k45[0], N_EQ, sys, d_sp, Fval_arr, nForc);

    //     rk45_step<Model204>(t, y, y_next, N_EQ, h, rtol, atol, &err, k45, sys, d_sp);

        if (err <= 1.0) {
            reject_count = 0;
            // slope‐jump detection…
            double jump = norm_inf_diff(k45[0], k45[1], N_EQ);
            if (jump > SLOPE_JUMP_THRESH) {
                h = fmax(h * 0.5, devParams.initialStep * MIN_STEP_FRACTION);
                continue;
            }
            // dense output…
            double t1 = t + h;
            while (next_q < num_queries && query_times[next_q] <= t1) {
                double tq = query_times[next_q];
                if (tq > t) {
                    double th = (tq - t)/h, yd[N_EQ];
                    rk45_dense<Model204>(y, k45, N_EQ, h, th, yd);
                    for (int c=0; c<N_EQ; ++c)
                        dense_all[sys*(N_EQ*num_queries) + c*num_queries + next_q] = yd[c];
                }
                ++next_q;
            }
            for (int i=0; i<N_EQ; ++i) y[i] = y_next[i];
            t = t1;
            double fac = devParams.safety * pow(1.0/(err + 1e-16), 0.2);
            h *= fmin(fmax(fac, devParams.minScale), devParams.maxScale);
        } else {
            // rejected
            ++reject_count;
            double fac = devParams.safety * pow(1.0/(err + 1e-16), 0.2);
            fac = fmin(fac, 1.0);
            fac = fmin(fmax(fac, devParams.minScale), devParams.maxScale);
            h *= fac;
            if (reject_count > 5 || h < (tf - t0)*MIN_STEP_FRACTION) {
                stiff = true;
            }
        }
    }

    // ─── 2) Flag stiff and bail ───
    if (stiff && t < tf) {
        d_stiff[sys] = 1;  // mark this system as stiff
        return;
    }

    // ─── 3) Never stiff: write final RK45 state ───
    for (int i=0; i<N_EQ; ++i) {
        y_final_all[sys*N_EQ + i] = y[i];
    }
}

// ----------------------------------------------------------------------------
// Explicit instantiation for Model204
// ----------------------------------------------------------------------------
// new — all 12 params in order:
template __global__ void rk45_then_radau_multi<Model204>(
    double*,          // y0_all
    double*,          // y_final_all
    double*,          // query_times
    double*,          // dense_all
    int,              // num_systems
    int,              // num_queries
    double,           // t0
    double,           // tf
    const Model204::SP_TYPE*, // d_sp
    int*,             // d_stiff
    const float*,     // d_forc_data
    int               // nForc
);


