#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"               // for query_times / y0_all types
#include "radau_kernel.cuh"     // declaration of radau_kernel_multi
#include "radau_step_dense.cuh"       // radau_step() and radau_dense()
#include "event_detector.cuh"   // reuse norm_inf, norm_inf_diff if needed
#include "../models/model_Runoff5.hpp" // Runoff5
#include "../I_O/forcing_data.h"
#include "rk45.h"

// ────────── DEBUG BOUNDS MACRO ────────────────────────────────────────
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
// ────────────────────────────────────────────────────────────────────

// Optional throttled logging
#ifndef RKDBG
#define RKDBG 0
#endif
#if RKDBG
  #define DBG_PRINTF(...) printf(__VA_ARGS__)
#else
  #define DBG_PRINTF(...) ((void)0)
#endif
__device__ __managed__ int g_radau_logs = 0;


// Declaration only (definition is in rk45_kernel.cu)
extern __constant__ DevParams rkDevParams;

// Optional status codes
enum { RADAU_OK = 0, RADAU_REJECT_LIMIT = 1, RADAU_TIMEOUT = 5 };

// ─────────────────────────────────────────────────────────────────────────────────
// Implementation of radau_kernel_multi defined in radau_kernel.cuh
// ─────────────────────────────────────────────────────────────────────────────────
// This kernel integrates multiple systems in parallel using the Radau IIA method.

template <class Runoff5>
__global__ void radau_kernel_multi(
    double* y0_all,
    double* y_final_all,
    double* query_times,
    float* dense_all,
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf,
    const typename Runoff5::SP_TYPE* d_sp,
    int*    stiff_system_indices,
    int     n_stiff
)
    
  {
    constexpr int N_EQ = Runoff5::N_EQ;

    // Map thread → index in the stiff list
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_stiff) return;
    const int sys = stiff_system_indices[idx];

    // // Optional one-time debug print
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("[RADAU] d_forc_data=%p  nForc=%d  dt0=%.6g nT0=%llu\n",
    //            (void*)d_forc_data, nForc, c_forc_dt[0],
    //            (unsigned long long)c_forc_nT[0]);
    // }

    // // Optional one-time debug print
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("[RADAU] d_forc_data=%p  nForc=%d  dt0=%.6g nT0=%llu\n",
    //            (void*)d_forc_data, nForc, c_forc_dt[0],
    //            (unsigned long long)c_forc_nT[0]);
    //     printf("[RADAU] DevParams{ rtol=%.3e atol=%.3e safety=%.2f minScale=%.2f maxScale=%.2f h0=%.3e }\n",
    //            rkDevParams.rtol, rkDevParams.atol, rkDevParams.safety,
    //            rkDevParams.minScale, rkDevParams.maxScale, rkDevParams.initialStep);
    // }

    if (blockIdx.x == 0 && threadIdx.x == 0 && atomicAdd(&g_radau_logs,1) < 5) {
        DBG_PRINTF("[RADAU] d_forc_data=%p  nForc=%d  dt0=%.6g nT0=%llu\n",
                   (void*)d_forc_data, nForc, c_forc_dt[0],
                   (unsigned long long)c_forc_nT[0]);
        DBG_PRINTF("[RADAU] DevParams{ rtol=%.3e atol=%.3e safety=%.2f minScale=%.2f maxScale=%.2f h0=%.3e }\n",
                   rkDevParams.rtol, rkDevParams.atol, rkDevParams.safety,
                   rkDevParams.minScale, rkDevParams.maxScale, rkDevParams.initialStep);
    }

    // Load tolerances and initial step from constant memory
    double my_rtol = rkDevParams.rtol;
    double my_atol = rkDevParams.atol;
    double h       = rkDevParams.initialStep;
    double t       = t0;

    // Load initial condition & set up state buffers
    double y[N_EQ], y_next[N_EQ], error_norm;
    for (int i = 0; i < N_EQ; ++i) y[i] = y0_all[sys * N_EQ + i];

    // Index of next dense‐output query
    int next_q = 0;

    // Precompute base offsets & meta for forcings (identical to RK45)
    size_t forc_base[MAX_FORCINGS];
    double dt_cache[MAX_FORCINGS];
    size_t nT_cache[MAX_FORCINGS];
    {
        size_t acc = 0;
        for (int j = 0; j < nForc; ++j) {
            forc_base[j] = acc;
            acc += c_forc_nT[j] * (size_t)num_systems;
            dt_cache[j] = c_forc_dt[j];
            nT_cache[j] = c_forc_nT[j];
            if (!(dt_cache[j] > 0.0) || nT_cache[j] == 0) {
                if (idx == 0) {
                    printf("[RADAU-ERR] invalid forcing meta: j=%d dt=%.6g nT=%llu (sys=%d)\n",
                           j, dt_cache[j], (unsigned long long)nT_cache[j], sys);
                }
                // Write current state and return to avoid infinite loop
                for (int c = 0; c < N_EQ; ++c) y_final_all[sys * N_EQ + c] = y[c];
                return;
            }
        }
    }

    // ─── Main implicit Radau loop ───
    // Guards copied from RK path to prevent infinite loops
    const int    REJECT_LIMIT = 5;
    const double EPS          = 1e-10;
    const double H_MIN        = 1e-8;
    const double H_MIN_ABS    = 1e-7;
    long long iter = 0;
    long long no_prog_iters = 0;
    double t_prev = t;
    if (!(h > 0.0)) h = H_MIN;
    if (h < H_MIN)  h = H_MIN;
    while (t < tf) {
        ++iter;
        if (!isfinite(h) || h < H_MIN) h = H_MIN;
        // adjust final step if needed
        if (t + h > tf) h = tf - t;

        // Do not cross forcing boundaries (match RK45 behavior)
        {
            double h_forc = 1e300;
            for (int j = 0; j < nForc; ++j) {
                const double dtj    = dt_cache[j];
                const double s      = (t - t0) / dtj;
                const double t_next = (floor(s) + 1.0) * dtj + t0;
                const double rem    = t_next - t;
                if (rem > 1e-12) h_forc = fmin(h_forc, rem);
            }
            if (h_forc < 1e300 && h > h_forc) h = h_forc;
        }
        

        // Take one Radau step (fills y_next and error_norm)
        double Z[3][N_EQ];
        radau_step<Runoff5>(t, y, y_next, N_EQ, h, my_rtol, my_atol, &error_norm,
                            sys, d_sp, d_forc_data, nForc, Z,
                            /* sampling ctx: */ t0, forc_base, dt_cache, nT_cache, num_systems);

        // Accept or reject based on embedded error
        if (!isfinite(error_norm)) {
            // Treat NaN/Inf error as a reject with huge error
            error_norm = 1e30;
        }
        if (error_norm <= 1.0) {
            // Accept step
            const double t1 = t + h;
            while (next_q < num_queries && query_times[next_q] <= t1) {
                const double tq = query_times[next_q];
                const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                const long long base   = flat_q * N_EQ;
                if (tq <= t + 1e-12) {
                    double yc[N_EQ]; for (int c=0;c<N_EQ;++c) yc[c] = y[c];
                    Runoff5::project_nonnegative(yc);
                    for (int c=0;c<N_EQ;++c) dense_all[base + c] = (float)yc[c];
                } else {
                    const double theta = (tq - t) / h;
                    double y_dense[N_EQ];
                    radau_dense<Runoff5>(y, Z, N_EQ, h, theta, y_dense);
                    Runoff5::project_nonnegative(y_dense);
                    for (int c=0;c<N_EQ;++c) dense_all[base + c] = (float)y_dense[c];
                }
                ++next_q;
            }


            // Advance state to the next step
            for (int i = 0; i < N_EQ; ++i) y[i] = y_next[i];
            t = t1;


            // Rescale h: stiff‐controller (e.g. simple power‐law)
            double fac = rkDevParams.safety * pow(1.0 / (error_norm + 1e-16), 0.2);
            fac = fmin(fmax(fac, rkDevParams.minScale), rkDevParams.maxScale);
            // h *= fac;
            // if (h < 1e-12) h = 1e-12;
            h *= fac;
            if (h < H_MIN) h = H_MIN;

            // progress watchdog
            if (fabs(t - t_prev) < EPS) {
                if (++no_prog_iters >= 50000) {
                    printf("[RADAU-STOP] sys=%d reason=no_progress iters=%lld t=%.9f h=%.3e\n",
                           sys, (long long)no_prog_iters, t, h);
                    break;
                }
            } else {
                no_prog_iters = 0;
                t_prev = t;
            }

        } else {
            // Reject: shrink h and retry
            static_assert(true, "No-op to keep patch context");
            int rejects_local = 0;  // start at 0 for this step attempt
            ++rejects_local;
            double fac = rkDevParams.safety * pow(1.0 / (error_norm + 1e-16), 0.2);
            fac = fmin(fac, 1.0);
            fac = fmin(fmax(fac, rkDevParams.minScale), rkDevParams.maxScale);
            h *= fac;
            if (h < H_MIN) h = H_MIN;
            if (rejects_local > REJECT_LIMIT || h < H_MIN_ABS) {
                if (atomicAdd(&g_radau_logs,1) < 50) {
                    printf("[RADAU-STOP] sys=%d reason=reject_limit=%d h=%.3e t=%.9f\n",
                           sys, REJECT_LIMIT, h, t);
                }
                break;
            }
        }
    }

    // Write final state
    for (int i = 0; i < N_EQ; ++i)
    {
        y_final_all[sys * N_EQ + i] = y[i];
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Explicit instantiation of radau_kernel_multi for Runoff5
// ───────────────────────────────────────────────────────────────────────────
template __global__ void radau_kernel_multi<Runoff5>(
    double*, double*, double*, float*,
    int, int, double, double,
    const Runoff5::SP_TYPE*,
    int*,     // stiff_system_indices
    int       // n_stiff
);
