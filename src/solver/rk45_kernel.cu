// RK45-only kernel (fallbacks commented out) with detailed diagnostics.
// All prints include sys and stream ID context.

#include <cstdio>
#include <cuda_runtime.h>
#include <math.h> 
#include "rk45.h"
#include "rk45_step_dense.cuh" 
#include "small_lu.cuh"        // for Radau fallback
#include "../models/model_Runoff5.hpp"
#include "../I_O/forcing_data.h"
#include "radau_step_dense.cuh"   // enable Radau fallback inline
#include <assert.h>

// Single definition of the shared solver parameters in constant memory
__constant__ DevParams rkDevParams;

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

// ─────────── Logging throttle ───────────
#ifndef RKDBG
#define RKDBG 0   // set to 1 to re-enable chatty prints
#endif
#if RKDBG
  #define DBG_PRINTF(...) printf(__VA_ARGS__)
#else
  #define DBG_PRINTF(...) ((void)0)
#endif
// Global managed counters to cap specific messages across the grid
__device__ __managed__ int g_stiff_logs   = 0;
__device__ __managed__ int g_reject_logs  = 0;
__device__ __managed__ int g_jump_logs    = 0;
__device__ __managed__ int g_nanlte_logs  = 0;
__device__ __managed__ int g_nanstate_logs= 0;
__device__ __managed__ int g_heartbeat    = 0;

// ────────────────────────────────────────────────────────────────────────────
// d_stiff status codes (per-system solver outcome)
//
//  0 : OK                  (normal RK45 finish)
//  1 : Stiff               (reject-limit or min h hit; host may run Radau)
//  2 : NaN LTE             (non-finite LTE estimate)
//  3 : NaN state           (non-finite state proposal)
//  5 : Fuse/timeout        (iteration/time-progress guard tripped)
//  7 : Config error        (invalid forcing meta)
// Notes: host inspects d_stiff[sys] after kernel.
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
/* Local fallbacks for helpers/thresholds that were defined in other headers
   in the original build. Keeping them here makes this file self-contained.
   If your project already defines them elsewhere, you can remove this block
   and include that header instead. */
// ────────────────────────────────────────────────────────────────────────────
#ifndef SLOPE_JUMP_THRESH
// Heuristic threshold on ||k1 - k2||_inf indicating a sharp slope change.
// Choose conservatively large by default; tune per model if needed.
#define SLOPE_JUMP_THRESH 1e6
#endif

#ifndef MIN_STEP_FRACTION
// Minimum fraction of the initial step allowed when cutting h due to
// slope jumps. Very small floor avoids h→0 underflow in stiff regions.
#define MIN_STEP_FRACTION 1e-6
#endif

// Infinity-norm difference helper for the slope-jump heuristic.
__device__ __forceinline__ double norm_inf_diff(const double* __restrict__ a,
                                                const double* __restrict__ b,
                                                int n)
{
    double m = 0.0;
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}


// ────────────────────────────────────────────────────────────────────────────
// Single‐kernel that does RK45 and flags stiffness. All in-kernel fallback
// (ROS2W / Radau / “pdf” fallback) is removed/commented out.
// ────────────────────────────────────────────────────────────────────────────
template <class Runoff5>
__global__ void rk45_then_radau_multi(
    double* y0_all,       // [num_systems × N_EQ]
    double* y_final_all,  // [num_systems × N_EQ]
    double* query_times,  // [num_queries]
    float*  dense_all,    // [num_systems × num_queries × N_EQ]
    int     num_systems,  // number of systems to solve
    int     num_queries,  // number of queries to answer
    double  t0,           // initial time (same units as c_forc_dt entries)
    double  tf,           // final time
    const typename Runoff5::SP_TYPE* d_sp, // per-system parameters
    int*    d_stiff,      // [num_systems] status codes
    const long* stream_ids// [num_systems] for diagnostics
) {
    // Device-side forcing descriptors (defined elsewhere in CU/CUH)
    extern __device__   float*   d_forc_data;   // flattened forcing values
    extern __constant__ double   c_forc_dt[];   // forcing cadence (same units used by original code)
    extern __constant__ size_t   c_forc_nT[];   // number of time samples per forcing
    extern __constant__ int      nForc;         // number of distinct forcings

    constexpr int N_EQ = Runoff5::N_EQ;
    const int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys >= num_systems) return;

    // Safety: ensure forcing slab pointer is valid
    if (!d_forc_data) {
        if (sys == 0) {
            printf("[KERNEL] d_forc_data is null; sys=%d streamID=%ld — skipping.\n",
                   sys, stream_ids[sys]);
        }
        return;
    }

    // Validate forcing count
    if (nForc <= 0 || nForc > MAX_FORCINGS) {
        if (sys == 0) {
            printf("[KERNEL] Invalid nForc=%d; sys=%d streamID=%ld\n",
                   nForc, sys, stream_ids[sys]);
        }
        d_stiff[sys] = 7;
        return;
    }

    // Solver params from constant/dev memory
    const double rtol = rkDevParams.rtol;
    const double atol = rkDevParams.atol;

    // Controller / guards constants
    const int    REJECT_LIMIT = 20;      // fine
    const double EPS          = 1e-10;   // ok
    const double H_MIN        = 1e-8;    // raise from 1e-12
    const double H_MIN_ABS    = 1e-7;    // ~10× H_MIN


    // Per-thread state
    double y[N_EQ], y_next[N_EQ], k45[7][N_EQ];
    double err = 0.0;

    for (int i = 0; i < N_EQ; ++i)
        y[i] = y0_all[sys * N_EQ + i];

    int  next_q = 0;
    int  reject_count = 0;
    bool stiff = false;

    // Time integration window 
    double t = t0;
    double h = rkDevParams.initialStep;
    if (h < H_MIN) h = H_MIN;

    // Precompute base offsets for each forcing slab and cache meta
    size_t forc_base[MAX_FORCINGS];
    double dt_cache[MAX_FORCINGS];   // same units as c_forc_dt
    size_t nT_cache[MAX_FORCINGS];
    {
        size_t acc = 0;
        for (int j = 0; j < nForc; ++j) {
            forc_base[j] = acc;
            acc += c_forc_nT[j] * size_t(num_systems);
            dt_cache[j] = c_forc_dt[j];
            nT_cache[j] = c_forc_nT[j];
            if (!(dt_cache[j] > 0.0) || nT_cache[j] == 0) {
                if (sys == 0) {
                    printf("[KERNEL] invalid forcing meta: j=%d dt=%.6g nT=%zu  sys=%d streamID=%ld\n",
                           j, dt_cache[j], nT_cache[j], sys, stream_ids[sys]);
                }
                d_stiff[sys] = 7;
                return;
            }
        }
    }

    // ───── Progress guards ─────
    // cache the smallest forcing cadence (in minutes) for adaptive thresholds
    double dt_min = 1e300;
    for (int j = 0; j < nForc; ++j) dt_min = fmin(dt_min, dt_cache[j]);
    if (!(dt_min > 0.0) || !isfinite(dt_min)) dt_min = 60.0;  // sane fallback (1 hr)

    // If time advances less than eps_t for many iterations, flag stiffness.
    const double eps_t = fmax(1e-4 /* minutes ≈ 0.006 s */, 1e-6 * dt_min);
    long long no_prog_iters = 0;
    double t_prev = t;   // t now exists

    // If accepted h is persistently tiny relative to forcing cadence, flag stiffness.
    const double tiny_h_thresh = 1e-6 * dt_min;
    int tiny_h_accepted = 0;
    const int TINY_H_LIMIT = 2000;   // ~2k accepted micro-steps ⇒ stiff  


    // Global loop guard to avoid GPU hangs
    long long iter = 0;
    const long long MAX_ITERS = 500000000LL; // 5e8; tune as needed

    if (sys == 0) {
        DBG_PRINTF("[SOLVER-START] sys=%d streamID=%ld t0=%.9f tf=%.9f init_h=%.3e rtol=%.1e atol=%.1e\n",
            sys, stream_ids[sys], t0, tf, h, rtol, atol);
               
        // ───── Debug: Verify constant-memory DevParams ─────
        DBG_PRINTF("[DevParams] rtol=%.3e atol=%.2e safety=%.2f minScale=%.2f maxScale=%.2f h0=%.3e\n",
               rkDevParams.rtol,
               rkDevParams.atol,
               rkDevParams.safety,
               rkDevParams.minScale,
               rkDevParams.maxScale,
               rkDevParams.initialStep);
    }

    while (t < tf && !stiff && iter < MAX_ITERS) {
        // Track whether this step has been clamped to a forcing boundary.
        // If true, the slope-jump heuristic is disabled because the RHS is
        // discontinuous and k1/k2 difference will be huge but expected.
        bool at_forcing_boundary = false;

        ++iter;
        if ((iter % 5000000LL) == 0 && sys == 0) {
            if (atomicAdd(&g_heartbeat, 1) < 10) {
                DBG_PRINTF("[HEARTBEAT] sys=%d streamID=%ld iter=%lld t=%.9f h=%.3e last_err=%.3e\n",
                           sys, stream_ids[sys], (long long)iter, t, h, err);
            }
        }

        // ── Stall heartbeat ───────────────────────────────────────────────────────────
        // Prints every 10 million iterations per thread to confirm progress.
        // Helpful for diagnosing GPU stalls or extremely stiff systems.
        if ((iter % 10000000LL) == 0) {
            printf("[STALL-CHECK] sys=%d streamID=%ld still alive at t=%.9f h=%.3e\n",
                sys, stream_ids[sys], t, h);
        }
        // ──────────────────────────────────────────────────────────────────────────────



        // Never step past tf
        if (t + h > tf) h = tf - t;

        // Do not cross forcing boundaries (piecewise-constant RHS)
        {
            double h_forc = 1e300;
            for (int j = 0; j < nForc; ++j) {
                const double dtj    = dt_cache[j];
                const double s      = (t - t0) / dtj;
                const double t_next = (floor(s) + 1.0) * dtj + t0;
                const double rem    = t_next - t;
                if (rem > EPS) h_forc = fmin(h_forc, rem);
            }
            if (h_forc < 1e300) {
                // if (h > h_forc) h = h_forc;
                // // reject_count reset across discontinuities (original behavior)
                // reject_count = 0;
                if (h > h_forc) {
                    h = h_forc;              // land *exactly* on forcing change
                    reject_count = 0;        // allow clean step acceptance
                    at_forcing_boundary = true;
                }
            }
        }

        // Gather forcings (sample-and-hold)
        float Fval_arr[MAX_FORCINGS];
        for (int j = 0; j < nForc; ++j) {
            const double dtj   = dt_cache[j];
            const double s_real= (t - t0) / dtj;
            const size_t nS    = nT_cache[j];
            size_t sampleIdx = (s_real < 0.0) ? size_t(0)
                                : (s_real >= double(nS)) ? (nS - 1)
                                : size_t(s_real);
            const size_t idx = forc_base[j] + sampleIdx * size_t(num_systems) + size_t(sys);
            Fval_arr[j] = d_forc_data[idx];
            if (!::isfinite(Fval_arr[j])) {
            if (atomicAdd(&g_nanlte_logs, 1) < 50) {
                printf("[FORCING-NAN] sys=%d streamID=%ld j=%d val=%f t=%.9f\n",
                       sys, stream_ids[sys], j, Fval_arr[j], t);
            }
                d_stiff[sys] = 7;
                return;
            }
        }

        // Stage-0 slope
        Runoff5::rhs(t, y, k45[0], N_EQ, sys, d_sp, Fval_arr, nForc);

        // Attempt one RK45 step (fills y_next, k45, err)
        rk45_step<Runoff5>(t, y, y_next, N_EQ, h, rtol, atol, &err,
                           k45, sys, d_sp, Fval_arr, nForc);

        // Check for NaN LTE estimate
        if (!::isfinite(err)) {
            if (atomicAdd(&g_nanlte_logs, 1) < 50) {
                printf("[NaN-LTE] sys=%d streamID=%ld t=%.9f h=%.3e\n", sys, stream_ids[sys], t, h);
            }
            stiff = true;           // force implicit fallback
            // shrink h and continue; Radau block will handle the step
            h = fmax(0.5 * h, H_MIN);
            continue;
        }
        // Check for NaN in proposed state
        for (int c = 0; c < N_EQ; ++c) {
            if (!::isfinite(y_next[c])) {
                Runoff5::project_nonnegative(y);
                for (int i = 0; i < N_EQ; ++i) y_final_all[sys * N_EQ + i] = y[i];
                d_stiff[sys] = 3; // NaN state
                // printf("[NaN-STATE] sys=%d streamID=%ld comp=%d t=%.9f h=%.3e\n",
                //        sys, stream_ids[sys], c, t, h);
                if (atomicAdd(&g_nanstate_logs, 1) < 50) {
                    printf("[NaN-STATE] sys=%d streamID=%ld comp=%d t=%.9f h=%.3e\n",
                           sys, stream_ids[sys], c, t, h);
                }
                return;
            }
        }

        if (err <= 1.0) {
            // ───────────── Accepted step ─────────────
            reject_count = 0;

            // Slope-jump heuristic: disable at forcing boundaries
            const double jump = norm_inf_diff(k45[0], k45[1], N_EQ);
            if (!at_forcing_boundary && jump > SLOPE_JUMP_THRESH) {
                double h_prev = h;
                h = fmax(0.5 * h, rkDevParams.initialStep * MIN_STEP_FRACTION);
                if (h < H_MIN) h = H_MIN;
                if ((iter % 1000000LL) == 0 && atomicAdd(&g_jump_logs,1) < 200) {
                    printf("[SLOPE-JUMP] sys=%d streamID=%ld t=%.9f new_h=%.3e jump=%.3e\n",
                           sys, stream_ids[sys], t, h, jump);
                }

                // If already at H_MIN: no room left to shrink → declare stiff
                if (fabs(h - h_prev) < 1e-20) {
                    stiff = true;   // go implicit, stop retry spam
                    break;
                }
                continue; // retry at same t with smaller h
            }

            // Emit dense output for queries in (t, t+h]
            const double t1 = t + h;
            while (next_q < num_queries && query_times[next_q] <= t1) {
                const double tq = query_times[next_q];
                const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                DBG_ASSERT(next_q >= 0 && next_q < num_queries, "next_q OOB %d/%d", next_q, num_queries);
                DBG_ASSERT(flat_q >= 0 &&
                           flat_q < (long long)num_systems * (long long)num_queries,
                           "dense idx OOB: flat_q=%lld ns*nq=%lld",
                           flat_q, (long long)num_systems * (long long)num_queries);
                const long long base_idx = flat_q * N_EQ;

                if (tq <= t + EPS) {
                    double y_clamped[N_EQ];
                    for (int c = 0; c < N_EQ; ++c) y_clamped[c] = y[c];
                    Runoff5::project_nonnegative(y_clamped);
                    for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = (float)y_clamped[c];
                } else {
                    const double th = (tq - t) / h;
                    double yd[N_EQ];
                    rk45_dense<Runoff5>(y, k45, N_EQ, h, th, yd);
                    Runoff5::project_nonnegative(yd);
                    for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = (float)yd[c];
                }
                ++next_q;
            }

            // Accept and advance
            Runoff5::project_nonnegative(y_next);
            for (int i = 0; i < N_EQ; ++i) y[i] = y_next[i];
            t = t1;

            // ───── Guard B: persistent micro-steps relative to forcing cadence ───────────────
            if (h < tiny_h_thresh) {
                if (++tiny_h_accepted >= TINY_H_LIMIT) {
                    printf("[STIFF-DETECTED] sys=%d streamID=%ld t=%.9f reason=micro_steps tiny_h=%d\n",
                           sys, stream_ids[sys], t, tiny_h_accepted);
                    stiff = true;
                    break;  // Exit RK45 loop immediately to enter Radau fallback
                 
                }
            } else {
                // reset if we escape the micro-step regime
                if (tiny_h_accepted) tiny_h_accepted = 0;
            }

            // Step-size update
            double fac = rkDevParams.safety * pow(1.0 / (err + 1e-16), 0.2);
            fac = fmin(fmax(fac, rkDevParams.minScale), rkDevParams.maxScale);
            h *= fac;
            if (h < H_MIN) h = H_MIN;

        } else {
            // ───────────── Rejected step ─────────────
            ++reject_count;
            if ((reject_count % 10) == 0 && atomicAdd(&g_reject_logs,1) < 200) {
                printf("[REJECT] sys=%d streamID=%ld rej=%d t=%.9f h=%.3e err=%.3e\n",
                       sys, stream_ids[sys], reject_count, t, h, err);
            }
            // Conservative shrink on reject
            double fac = rkDevParams.safety * pow(1.0 / (err + 1e-16), 0.2);
            fac = fmin(fac, 1.0);
            fac = fmin(fmax(fac, rkDevParams.minScale), rkDevParams.maxScale);
            h *= fac;
            if (h < H_MIN) h = H_MIN;

            if (reject_count > REJECT_LIMIT || h < H_MIN_ABS) {
                if (atomicAdd(&g_stiff_logs,1) < 200) {
                    printf("[STIFF-DETECTED] sys=%d streamID=%ld t=%.9f reason=reject_limit=%d h=%.3e\n",
                           sys, stream_ids[sys], t, reject_count, h);
                }
                stiff = true; // host may run Radau/implicit out-of-kernel
                break;  // Exit RK45 loop immediately to enter Radau fallback
            }
        }

        // ───── Guard A: time-progress watchdog (independent of accept/reject) ─────
        if (fabs(t - t_prev) < eps_t) {
            if (++no_prog_iters >= 50000) { // 50k iterations with < eps_t advance
                // printf("[STIFF-DETECTED] sys=%d streamID=%ld t=%.9f reason=no_progress iters=%lld\n",
                //        sys, stream_ids[sys], t, no_prog_iters);
                if (atomicAdd(&g_stiff_logs,1) < 50) {
                printf("[TIMEOUT] sys=%d streamID=%ld iters=%lld t=%.9f h=%.3e\n",
                   sys, stream_ids[sys], (long long)iter, t, h);
                }
                stiff = true;
                break;  // Exit RK45 loop immediately to enter Radau fallback
            }
        } else {
            no_prog_iters = 0;
            t_prev = t;
        }
    } // while

    // Timeout / fuse
    if (iter >= MAX_ITERS) {
        d_stiff[sys] = 5;
        printf("[TIMEOUT] sys=%d streamID=%ld iters=%lld t=%.9f h=%.3e\n",
               sys, stream_ids[sys], (long long)iter, t, h);
        return;
    }


    // Stiff flag: switch to in-kernel Radau until tf
    if (stiff && t < tf) {
        if (sys == 0 && atomicAdd(&g_stiff_logs,1) < 20) {
            DBG_PRINTF("[RADAU-SWITCH] sys=%d streamID=%ld t=%.9f h=%.3e  DevParams{ rtol=%.3e atol=%.3e safety=%.2f minScale=%.2f maxScale=%.2f h0=%.3e }\n",
                       sys, stream_ids[sys], t, h,
                       rkDevParams.rtol, rkDevParams.atol, rkDevParams.safety,
                       rkDevParams.minScale, rkDevParams.maxScale, rkDevParams.initialStep);
        }
        while (t < tf) {
            if (t + h > tf) h = tf - t;
            // clamp to forcing boundary like RK45
            double h_forc = 1e300;
            for (int j = 0; j < nForc; ++j) {
                const double dtj = dt_cache[j];
                const double s   = (t - t0) / dtj;
                const double tn  = (floor(s) + 1.0) * dtj + t0;
                const double rem = tn - t;
                if (rem > EPS) h_forc = fmin(h_forc, rem);
            }
            if (h_forc < 1e300 && h > h_forc) h = h_forc;

            // one implicit step (Radau does its own forcing sampling)
            double Z[3][N_EQ], y_next_r[N_EQ];
            double errn = 0.0;
            radau_step<Runoff5>(t, y, y_next_r, N_EQ, h, rtol, atol, &errn,
                                sys, d_sp, d_forc_data, nForc, Z,
                                /* sampling ctx */ t0, forc_base, dt_cache, nT_cache, num_systems);

            if (errn <= 1.0) {
                const double t1 = t + h;
                // dense output for (t, t+h]
                while (next_q < num_queries && query_times[next_q] <= t1) {
                    const double tq = query_times[next_q];
                    const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                    const long long base_idx = flat_q * N_EQ;
                    if (tq <= t + EPS) {
                        double yc[N_EQ]; for (int c=0;c<N_EQ;++c) yc[c] = y[c];
                        Runoff5::project_nonnegative(yc);
                        for (int c=0;c<N_EQ;++c) dense_all[base_idx + c] = (float)yc[c];
                    } else {
                        const double th = (tq - t) / h;
                        double yd[N_EQ];
                        radau_dense<Runoff5>(y, Z, N_EQ, h, th, yd);
                        Runoff5::project_nonnegative(yd);
                        for (int c=0;c<N_EQ;++c) dense_all[base_idx + c] = (float)yd[c];
                    }
                    ++next_q;
                }
                // accept & advance
                Runoff5::project_nonnegative(y_next_r);
                for (int i=0;i<N_EQ;++i) y[i] = y_next_r[i];
                t = t1;
                // step size update (PI controller)
                double fac = rkDevParams.safety * pow(1.0 / (errn + 1e-16), 0.2);
                fac = fmin(fmax(fac, rkDevParams.minScale), rkDevParams.maxScale);
                h *= fac; if (h < H_MIN) h = H_MIN;
            } else {
                // reject: shrink and retry
                double fac = rkDevParams.safety * pow(1.0 / (errn + 1e-16), 0.2);
                fac = fmin(fac, 1.0);
                fac = fmin(fmax(fac, rkDevParams.minScale), rkDevParams.maxScale);
                h *= fac; if (h < H_MIN) h = H_MIN;
            }
        }
        // success after Radau fallback
        Runoff5::project_nonnegative(y);
        for (int i = 0; i < N_EQ; ++i)
            y_final_all[sys * N_EQ + i] = y[i];
        d_stiff[sys] = 0;
        return;
    }




    // Success: write final state
    Runoff5::project_nonnegative(y);
    for (int i = 0; i < N_EQ; ++i)
        y_final_all[sys * N_EQ + i] = y[i];

    d_stiff[sys] = 0;
    if (sys == 0) {
        DBG_PRINTF("[SOLVER-END] sys=%d streamID=%ld done. iters=%lld queries=%d\n",
                              sys, stream_ids[sys], (long long)iter, next_q);
    }
}


// ────────────────────────────────────────────────────────────────────────────
// Explicit instantiation for Runoff5
// ────────────────────────────────────────────────────────────────────────────
template __global__ void rk45_then_radau_multi<Runoff5>(
    double*,          // y0_all
    double*,          // y_final_all
    double*,          // query_times
    float*,           // dense_all
    int,              // num_systems
    int,              // num_queries
    double,           // t0
    double,           // tf
    const Runoff5::SP_TYPE*, // d_sp
    int*,             // d_stiff
    const long*       // stream_ids
);
