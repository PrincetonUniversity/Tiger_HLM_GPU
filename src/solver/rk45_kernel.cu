#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"
#include "rk45_step_dense.cuh"
#include "event_detector.cuh"
#include "small_lu.cuh"
#include "radau_step_dense.cuh"
#include "models/model_204.hpp"

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
    int*    d_stiff       // new: flag array
) {
    constexpr int N_EQ = Model204::N_EQ;
    int sys = blockIdx.x*blockDim.x + threadIdx.x;
    if (sys >= num_systems) return;

    // solver parameters
    double rtol = devParams.rtol;
    double atol = devParams.atol;
    double h    = devParams.initialStep;
    double t    = t0;

    // state buffers
    double y[N_EQ], y_next[N_EQ], k45[7][N_EQ], err;
    for(int i=0; i<N_EQ; ++i) y[i] = y0_all[sys*N_EQ + i];

    int next_q = 0, reject_count = 0;
    bool stiff = false;

    // ─── 1) Explicit RK45 phase ───
    while(t < tf && !stiff) {
        if (t + h > tf) h = tf - t;
        Model204::rhs(t, y, k45[0], N_EQ, sys, d_sp);
        rk45_step<Model204>(t, y, y_next, N_EQ, h, rtol, atol, &err, k45, sys, d_sp);

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
template __global__ void rk45_then_radau_multi<Model204>(
    double*, double*, double*, double*,
    int,int,double,double,
    const Model204::SP_TYPE*,
    int*
);

