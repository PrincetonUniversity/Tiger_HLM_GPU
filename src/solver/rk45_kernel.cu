//solvers/rk45_kernel.cu
#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"
#include "rk45_step_dense.cuh"
#include "event_detector.cuh"    // norm_inf, norm_inf_diff, SLOPE_JUMP_THRESH, MIN_STEP_FRACTION
#include "small_lu.cuh"          // small_matrix_LU_solve
#include "radau_step_dense.cuh"  // radau_step, radau_dense
#include "models/model_204.hpp"   // brings in Model204
// -----------------------------------------------------------------------------
// Single-kernel that does RK45, then (per-thread) falls back to Radau if needed.
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
    //const SpatialParams* __restrict__ d_sp
    const typename Model204::SP_TYPE* __restrict__ d_sp
) { 
    // figure out which system this thread is working on
    int sys = blockIdx.x*blockDim.x + threadIdx.x;
    if (blockIdx.x==0 && threadIdx.x==0) {
        printf("[rk45_solver] entered rk45_then_radau_multi on sys %d\n", sys);
    }
    constexpr int N_EQ = Model204::N_EQ;

    //int sys = blockIdx.x*blockDim.x + threadIdx.x;
    if (sys >= num_systems) return;

    // solver parameters
    double rtol = devParams.rtol;
    double atol = devParams.atol;
    double h    = devParams.initialStep;
    double t    = t0;

    // state buffers
    double y[N_EQ], y_next[N_EQ];
    double k45[7][N_EQ];
    double radau_k[3][N_EQ];   // Radau‐IIA has 3 stages → 3×N_EQ
    double err;

    // load initial y
    for(int i=0;i<N_EQ;++i) y[i] = y0_all[sys*N_EQ + i];

    int next_q = 0;
    int reject_count = 0;
    bool stiff = false;

    // ─── 1) Explicit RK45 phase ───
    while(t < tf && !stiff) {
        if (t + h > tf) h = tf - t;

        // compute k45[0]
        //Model204::rhs(t, y, k45[0], N_EQ); //without sys
        Model204::rhs(t, y, k45[0], N_EQ, sys, d_sp);
        // one RK45 step
        rk45_step<Model204>(t, y, y_next, N_EQ, h, rtol, atol, &err, k45, sys, d_sp);

        if (err <= 1.0) {
            // accepted
            reject_count = 0;

            // ─── Event detection / slope‐jump would go here ───
            // Compare k45-stage 0 vs. stage 1 to see if slope changes too abruptly !!!
            {
                double jump = norm_inf_diff(k45[0], k45[1], N_EQ);
                if (jump > SLOPE_JUMP_THRESH) {
                    // detect kink: halve the step and retry
                    h *= 0.5;
                    // also clamp h to a minimum
                    h = fmax(h, devParams.initialStep * MIN_STEP_FRACTION);
                    continue;  // go back and re-attempt the step
                }
            }

            // ─── Dense output for RK45 queries ───
            double t1 = t + h;
            while(next_q < num_queries && query_times[next_q] <= t1) {
                double tq = query_times[next_q];
                if (tq > t) {
                    double th = (tq - t)/h;
                    double yd[N_EQ];
                    rk45_dense<Model204>(y, k45, N_EQ, h, th, yd);
                    for(int c=0;c<N_EQ;++c){
                        int idx = sys*(N_EQ*num_queries) + c*num_queries + next_q;
                        dense_all[idx] = yd[c];
                    }
                }
                ++next_q;
            }

            // advance
            for(int i=0;i<N_EQ;++i) y[i] = y_next[i];
            t = t1;

            // rescale h
            double fac = devParams.safety * pow(1.0/(err + 1e-16), 0.2);
            h *= fmin(fmax(fac, devParams.minScale), devParams.maxScale);

        } else {
            // rejected
            ++reject_count;
            double fac = devParams.safety * pow(1.0/(err + 1e-16), 0.2);
            fac = fmin(fac, 1.0);
            fac = fmin(fmax(fac, devParams.minScale), devParams.maxScale);
            h *= fac;

            // detect stiffness
            if (reject_count > 5 || h < (tf - t0)*MIN_STEP_FRACTION) {
                stiff = true;
            }
        }
    }

    // ─── 2) Implicit Radau phase (if needed) ───
    if (stiff && t < tf) {
        // bring forward the same y[], next_q, h, t
        double y_r[N_EQ], y_rnext[N_EQ];
        for(int i=0;i<N_EQ;++i) y_r[i] = y[i];

        while(t < tf) {
            if (t + h > tf) h = tf - t;

            // one Radau IIA step
            radau_step<Model204>(t, y_r, y_rnext, N_EQ, h, rtol, atol, &err, radau_k, sys, d_sp);

            // ─── Dense‐output for leftover queries ───
            double t1 = t + h;
            while(next_q < num_queries && query_times[next_q] <= t1) {
                double tq = query_times[next_q];
                if (tq > t) {
                    double th = (tq - t)/h;
                    double yd[N_EQ];
                    // <<< The only change: no cast, just pass radau_k directly >>>
                    radau_dense<Model204>(y_r, radau_k, N_EQ, h, th, yd);
                    for(int c=0;c<N_EQ;++c) {
                        int idx = sys*(N_EQ*num_queries) + c*num_queries + next_q;
                        dense_all[idx] = yd[c];
                    }
                }
                ++next_q;
            }

            // accept
            for(int i=0;i<N_EQ;++i) y_r[i] = y_rnext[i];
            t = t1;

            // stiff step‐size control
            double fac = devParams.safety * pow(1.0/(err + 1e-16), 1.0/5.0);
            h *= fmin(fmax(fac, devParams.minScale), devParams.maxScale);
        }

        // copy final
        for(int i=0;i<N_EQ;++i){
            y_final_all[sys*N_EQ + i] = y_r[i];
        }
        return;
    }

    // ─── 3) If never stiff or after RK45 finishes ───
    for(int i=0;i<N_EQ;++i){
        y_final_all[sys*N_EQ + i] = y[i];
    }
}

// ----------------------------------------------------------------------------
// Explicit instantiation for the chosen Model204
// ----------------------------------------------------------------------------
//  template __global__ void rk45_then_radau_multi<Model204>(
//      double* y0_all,
//      double* y_final_all,
//      double* query_times,
//      double* dense_all,
//      int     num_systems,
//      int     num_queries,
//      double  t0,
//      double  tf,
//      const Model204::SP_TYPE* __restrict__ d_sp
//      //const SpatialParams* d_sp
//  );

 template __global__ void rk45_then_radau_multi<Model204>(
    double*,double*,double*,double*,
    int,int,double,double,
    const Model204::SP_TYPE* __restrict__ d_sp
);




