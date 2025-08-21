// src/solver/rk45_kernel.cu

#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"
#include "rk45_step_dense.cuh"
#include "event_detector.cuh"
#include "small_lu.cuh"
#include "radau_step_dense.cuh"
#include "../models/model_Runoff5.hpp"
#include "../I_O/forcing_data.h"


// ───────────DEBUG BOUNDS MACRO ──────────────────────────────────────────────
#ifndef DBG_ASSERT
#define DBG_ASSERT(cond, msg, ...)                                  \
    do {                                                            \
        if (!(cond)) {                                              \
            printf("[BOUNDS-ERR] " msg "  (block %d, thread %d)\n", \
                   ##__VA_ARGS__,                                   \
                   blockIdx.x, threadIdx.x);                        \
            return; /* kill this thread, prevents UB */             \
        }                                                           \
    } while (0)
#endif
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
// Single‐kernel that does RK45 and flags stiffness, but no in‐kernel Radau.
// ────────────────────────────────────────────────────────────────────────────
template <class Runoff5>
__global__ void rk45_then_radau_multi(
    double* y0_all,       // [num_systems × N_EQ]
    double* y_final_all,  // [num_systems × N_EQ]
    double* query_times,  // [num_queries]
    //double* dense_all,    // [num_systems × N_EQ × num_queries]
    float* dense_all,
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf,
    const typename Runoff5::SP_TYPE* d_sp,
    int*    d_stiff       // flag array
) {

    
    // re-declare the symbols you need:
    extern __device__   float*   d_forc_data;
    extern __constant__ double   c_forc_dt[];
    extern __constant__ size_t   c_forc_nT[];
    extern __constant__ int      nForc;

    constexpr int N_EQ = Runoff5::N_EQ;
    int sys = blockIdx.x*blockDim.x + threadIdx.x;
    if (sys >= num_systems) return;

    // solver parameters
    double rtol = devParams.rtol;
    double atol = devParams.atol;



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
        
        for (int j = 0; j < nForc; ++j) {

            // compute which sample we need for this forcing
            double dt_min        = c_forc_dt[j] * 60.0;           // minutes
            double sampleIdxReal = (t - t0) / dt_min;            // fractional step
            size_t nSamples      = c_forc_nT[j];
            size_t sampleIdx     = sampleIdxReal < 0.0
                                ? 0
                                : sampleIdxReal >= nSamples
                                    ? nSamples - 1
                                    : size_t(sampleIdxReal);

            // build prefix‐sum of all prior blocks to get the true start offset
            size_t base = 0;
            for (int kk = 0; kk < j; ++kk) {
                base += c_forc_nT[kk] * size_t(num_systems);
            }

            // final flattened index into d_forc_data
            size_t idx = base + sampleIdx * size_t(num_systems) + sys;
            Fval_arr[j] = d_forc_data[idx];
        }

        

        // pass forcings into RHS
        Runoff5::rhs(t, y, k45[0], N_EQ, sys, d_sp, Fval_arr, nForc);
        



        rk45_step<Runoff5>(t, y, y_next, N_EQ, h, rtol, atol, &err, k45, sys, d_sp, Fval_arr, nForc);

    // int next_q = 0, reject_count = 0;
    // bool stiff = false;

    // // ─── 1) Explicit RK45 phase ───
    // while(t < tf && !stiff) {
    //     if (t + h > tf) h = tf - t;
    //     //Runoff5::rhs(t, y, k45[0], N_EQ, sys, d_sp);
    //     Runoff5::rhs(t, y, k45[0], N_EQ, sys, d_sp, Fval_arr, nForc);

    //     rk45_step<Runoff5>(t, y, y_next, N_EQ, h, rtol, atol, &err, k45, sys, d_sp);

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
                    rk45_dense<Runoff5>(y, k45, N_EQ, h, th, yd);
                    for (int c=0; c<N_EQ; ++c){
                        DBG_ASSERT(sys < num_systems, "Invalid system index sys=%d", sys);
                        DBG_ASSERT(next_q < num_queries, "Invalid query index next_q=%d", next_q);
                        DBG_ASSERT(dense_all != nullptr, "dense_all is null!");
                        // Store the dense output in the global array
                        long long idx = ((long long)sys * num_queries + next_q) * N_EQ + c;
                        DBG_ASSERT(idx < (long long)num_systems * num_queries * N_EQ, 
                                "OOB idx=%lld sys=%d q=%d c=%d", idx, sys, next_q, c);
                        // dense_all[idx] = yd[c];
                        dense_all[idx] = static_cast<float>(yd[c]); // convert to float


                     }
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

// ────────────────────────────────────────────────────────────────────────────
// Explicit instantiation for Runoff5
// ────────────────────────────────────────────────────────────────────────────
// All 12 params in order:
template __global__ void rk45_then_radau_multi<Runoff5>(
    double*,          // y0_all
    double*,          // y_final_all
    double*,          // query_times
    //double*,          // dense_all
    float*,          // dense_all
    int,              // num_systems
    int,              // num_queries
    double,           // t0
    double,           // tf
    const Runoff5::SP_TYPE*, // d_sp
    int*             // d_stiff
);


