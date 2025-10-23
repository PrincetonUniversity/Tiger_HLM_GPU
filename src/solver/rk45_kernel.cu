// src/solver/rk45_kernel.cu

#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>
#include "rk45.h"
#include "rk45_step_dense.cuh"
#include "event_detector.cuh"
#include "small_lu.cuh"
#include "radau_step_dense.cuh"
#include "../models/model_Runoff5.hpp"
#include "../I_O/forcing_data.h"
#include <assert.h>

#define STIFF_CODE_EARLY_TIMEOUT 66


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
// d_stiff status codes (per-system solver outcome)
//
//  0 : OK
//      Integration finished normally on RK45 without triggering any special path.
//
//  1 : Stiff
//      RK45 flagged stiffness (e.g., too many rejects or step collapsed) and
//      bailed early. (In this version, Radau is handled out-of-kernel.)
//
//  2 : NaN LTE
//      Local truncation error (LTE) estimate was non-finite (NaN/Inf).
//
//  3 : NaN state
//      Proposed next state y_next contained a non-finite value (NaN/Inf).
//
//  5 : Fuse
//      Iteration fuse tripped (e.g., exceeded max iterations or stalled progress).
//
//  6 : Fallback finished
//      A fixed-step fallback integrator completed to tf. (Only used if fallback
//      logic is enabled; otherwise this code will not appear.)
//
//  7 : Config error
//      Configuration or bounds issue detected (e.g., nForc > MAX_FORCINGS).
//
// Notes:
//  • d_stiff is written per system (index: sys) and read by the host for diagnostics.
//  • 0 is the implicit “success” default when nothing sets an error/flag.
// ────────────────────────────────────────────────────────────────────────────

// If you ever want to finish the remainder of the window with RK4 instead of
// “burst then resume RK45”, flip this to true. Default: false (burst + resume).
static constexpr bool FALLBACK_TO_TF = true; //!!


// ────────────────────────────────────────────────────────────────────────────
// Small helper: sample-and-hold forcings at time t for this system
// (keeps semantics in SECONDS exactly as in the original kernel)
// ────────────────────────────────────────────────────────────────────────────
template <class Runoff5>
__device__ inline void sample_forcings_SnH_seconds(
    double t, double t0, int sys, int num_systems,
    const size_t* __restrict__ forc_base,
    const double* __restrict__ dt_sec_cache,
    const size_t* __restrict__ nT_cache,
    float* __restrict__ Fout, int nForc)
{
    extern __device__ float* d_forc_data;
    for (int j = 0; j < nForc; ++j) {
        const double dt_sec = dt_sec_cache[j];
        const double s_real = (t - t0) / dt_sec;
        const size_t nS     = nT_cache[j];
        size_t sampleIdx = (s_real < 0.0) ? size_t(0)
                            : (s_real >= double(nS)) ? (nS - 1)
                            : size_t(s_real);
        const size_t idx = forc_base[j] + sampleIdx * size_t(num_systems) + size_t(sys);
        Fout[j] = d_forc_data[idx];
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Rosenbrock–W (ROS2, L-stable) implicit step (seconds)
//  • 2 linearly-implicit stages, each solves (I - γ h J) k = rhs
//  • No inner Newton; stable in stiff regions
//  • Jacobian is built by finite differences (device-safe) to avoid model edits
//  • Returns false on any numerical failure so caller can fall back/bail
// ────────────────────────────────────────────────────────────────────────────
template <class Runoff5>
__device__ inline bool ros2w_step_seconds(
    double& t, double* __restrict__ y, const int N_EQ,
    double h, int sys, double t0, int num_systems,
    const size_t* __restrict__ forc_base,
    const double* __restrict__ dt_sec_cache,
    const size_t* __restrict__ nT_cache,
    const typename Runoff5::SP_TYPE* d_sp, int nForc)
{
    if (!(h > 0.0)) return false;
    if (sys == 0){
        printf("[ROS2W] sys=%d entering implicit ROS2W step at t=%.4f, h=%.2e\n", sys, t, h);
    }

    // Coefficients (ROS2-W; L-stable)
    const double gamma = 1.0 - 1.0 / sqrt(2.0); // ≈ 0.292893218
    const double a21 = 1.0;
    const double c21 = -2.0 * gamma;
    const double b1  = 0.5, b2 = 0.5;

    // Scratch
    float F0[MAX_FORCINGS], F1[MAX_FORCINGS];
    double f0[Runoff5::N_EQ], f1[Runoff5::N_EQ];
    double J[Runoff5::N_EQ * Runoff5::N_EQ];
    double M[Runoff5::N_EQ * Runoff5::N_EQ];
    double rhs[Runoff5::N_EQ];
    double k1[Runoff5::N_EQ], k2[Runoff5::N_EQ];
    double y_a[Runoff5::N_EQ], y_new[Runoff5::N_EQ];
    double ytmp[Runoff5::N_EQ], ftmp[Runoff5::N_EQ];

    // Forcings and RHS at (t, y)
    sample_forcings_SnH_seconds<Runoff5>(t, t0, sys, num_systems,
        forc_base, dt_sec_cache, nT_cache, F0, nForc);
    Runoff5::rhs(t, y, f0, N_EQ, sys, d_sp, F0, nForc);

    // Finite-difference Jacobian J = df/dy at (t,y)
    // step size scaled to state magnitude and tolerances
    for (int j = 0; j < N_EQ; ++j) {
        for (int k = 0; k < N_EQ; ++k) ytmp[k] = y[k];
        const double scale = 1.0 + fabs(y[j]);
        const double eps = 1e-6 * scale;  // simple, robust choice on GPU
        ytmp[j] += eps;
        Runoff5::rhs(t, ytmp, ftmp, N_EQ, sys, d_sp, F0, nForc);
        for (int i = 0; i < N_EQ; ++i) {
            J[i * N_EQ + j] = (ftmp[i] - f0[i]) / eps;
        }
    }

    // Build M = I - γ h J and solve for k1:  M k1 = f0
    for (int i = 0; i < N_EQ; ++i) {
        for (int j = 0; j < N_EQ; ++j) {
            M[i * N_EQ + j] = (i == j ? 1.0 : 0.0) - gamma * h * J[i * N_EQ + j];
        }
    }
    for (int i = 0; i < N_EQ; ++i) rhs[i] = f0[i];
    small_matrix_LU_solve(N_EQ, M, rhs); // M is overwritten
    for (int i = 0; i < N_EQ; ++i) k1[i] = rhs[i];

    // Stage 2 state
    for (int i = 0; i < N_EQ; ++i) y_a[i] = y[i] + a21 * k1[i];
    Runoff5::project_nonnegative(y_a);

    // RHS at (t+h, y_a)
    sample_forcings_SnH_seconds<Runoff5>(t + h, t0, sys, num_systems,
        forc_base, dt_sec_cache, nT_cache, F1, nForc);
    Runoff5::rhs(t + h, y_a, f1, N_EQ, sys, d_sp, F1, nForc);

    // Rebuild M (since LU destroyed it) and solve M k2 = f1 + (c21/h) k1
    for (int i = 0; i < N_EQ; ++i)
        for (int j = 0; j < N_EQ; ++j)
            M[i * N_EQ + j] = (i == j ? 1.0 : 0.0) - gamma * h * J[i * N_EQ + j];
    for (int i = 0; i < N_EQ; ++i) rhs[i] = f1[i] + (c21 / h) * k1[i];
    small_matrix_LU_solve(N_EQ, M, rhs);
    for (int i = 0; i < N_EQ; ++i) k2[i] = rhs[i];

    // Combine
    for (int i = 0; i < N_EQ; ++i) y_new[i] = y[i] + b1 * k1[i] + b2 * k2[i];
    Runoff5::project_nonnegative(y_new);
    for (int i = 0; i < N_EQ; ++i) {
        if (!::isfinite(y_new[i])) return false;
        y[i] = y_new[i];
    }
    t += h;
    return true;
}


// ────────────────────────────────────────────────────────────────────────────
template <class Runoff5>
__device__ inline void rk4_fixed_step_seconds(
    double& t, double* __restrict__ y, const int N_EQ,
    double h, int sys, double t0, int num_systems,
    const size_t* __restrict__ forc_base,
    const double* __restrict__ dt_sec_cache,
    const size_t* __restrict__ nT_cache,
    const typename Runoff5::SP_TYPE* d_sp, int nForc)
{
    float F0[MAX_FORCINGS], Fm[MAX_FORCINGS], F1[MAX_FORCINGS];
    double k1[Runoff5::N_EQ], k2[Runoff5::N_EQ], k3[Runoff5::N_EQ], k4[Runoff5::N_EQ], yt[Runoff5::N_EQ];

    sample_forcings_SnH_seconds<Runoff5>(t,          t0, sys, num_systems, forc_base, dt_sec_cache, nT_cache, F0, nForc);
    Runoff5::rhs(t, y, k1, N_EQ, sys, d_sp, F0, nForc);

    for (int i=0;i<N_EQ;++i) yt[i] = y[i] + 0.5*h*k1[i];
    Runoff5::project_nonnegative(yt);
    sample_forcings_SnH_seconds<Runoff5>(t + 0.5*h, t0, sys, num_systems, forc_base, dt_sec_cache, nT_cache, Fm, nForc);
    Runoff5::rhs(t + 0.5*h, yt, k2, N_EQ, sys, d_sp, Fm, nForc);

    for (int i=0;i<N_EQ;++i) yt[i] = y[i] + 0.5*h*k2[i];
    Runoff5::project_nonnegative(yt);
    Runoff5::rhs(t + 0.5*h, yt, k3, N_EQ, sys, d_sp, Fm, nForc);

    for (int i=0;i<N_EQ;++i) yt[i] = y[i] + h*k3[i];
    Runoff5::project_nonnegative(yt);
    sample_forcings_SnH_seconds<Runoff5>(t + h,     t0, sys, num_systems, forc_base, dt_sec_cache, nT_cache, F1, nForc);
    Runoff5::rhs(t + h, yt, k4, N_EQ, sys, d_sp, F1, nForc);

    for (int i=0;i<N_EQ;++i)
        y[i] += (h/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    Runoff5::project_nonnegative(y);
    t += h;
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers to write consistent outputs on early exit
// ────────────────────────────────────────────────────────────────────────────
template<int N_EQ>
__device__ inline void write_final_zero(int sys, double* __restrict__ y_final_all) {
    #pragma unroll
    for (int i = 0; i < N_EQ; ++i)
        y_final_all[sys * N_EQ + i] = 0.0;
}


// ────────────────────────────────────────────────────────────────────────────
// Single‐kernel that does RK45 and flags stiffness, but no in‐kernel Radau.
// (Base: original code. Added constraints below without changing sec/min logic.)
// ────────────────────────────────────────────────────────────────────────────
template <class Runoff5>
__global__ void rk45_then_radau_multi(
    double* y0_all,       // [num_systems × N_EQ]
    double* y_final_all,  // [num_systems × N_EQ]
    double* query_times,  // [num_queries]
    float*  dense_all,    // [num_systems × num_queries × N_EQ]
    int     num_systems,  // number of systems to solve
    int     num_queries,  // number of queries to answer
    double  t0,           // initial time (seconds)
    double  tf,           // final time   (seconds)
    const typename Runoff5::SP_TYPE* d_sp, // per-system parameters
    int*    d_stiff,       // [num_systems] stiffness flags
    const long* stream_ids // [num_systems] stream identifiers 
) {
    // Re-declare device symbols (defined elsewhere)
    extern __device__   float*   d_forc_data;   // flattened forcing values
    extern __constant__ double   c_forc_dt[];   // forcing cadence in MINUTES
    extern __constant__ size_t   c_forc_nT[];   // number of time samples per forcing
    extern __constant__ int      nForc;         // number of distinct forcings

    constexpr int N_EQ = Runoff5::N_EQ;
    const int sys = blockIdx.x * blockDim.x + threadIdx.x;

    bool used_ros2w = false;
    bool finished_by_fallback = false;

    __shared__ int stiff_kill_count;
    if (threadIdx.x == 0) stiff_kill_count = 0;
    __syncthreads();


    // (keep original limits in SECONDS)
    const int    REJECT_LIMIT = 20;     // was 5
    const double H_MIN_ABS    = 1e-7;  // seconds; tiny absolute floor (unchanged)

    // int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys >= num_systems) return;  // safety check
    assert(sys < num_systems);       // detect out-of-bounds early


    // if (sys >= num_systems) return;

    // Safety: ensure global pointer to forcing slab is valid
    if (!d_forc_data) {
        if (sys == 0) {
            printf("[KERNEL] d_forc_data is null; skipping computation.\n");
        }
        return;
    }

    // Validate forcing count early (constraint)
    if (nForc <= 0 || nForc > MAX_FORCINGS) {
        if (sys == 0) printf("[KERNEL] Invalid nForc=%d\n", nForc);
        d_stiff[sys] = 7; // Config error
        return;
    }

    // Solver parameters from constant/dev memory
    const double rtol = devParams.rtol;
    const double atol = devParams.atol;

    // ────────────────────────
    // Per-thread scratch state
    // ────────────────────────
    double y[N_EQ], y_next[N_EQ], k45[7][N_EQ], err;
    for (int i = 0; i < N_EQ; ++i) y[i] = y0_all[sys * N_EQ + i];

    int  next_q = 0, reject_count = 0;
    bool stiff  = false;

    // ─────────────────────────────────────────────────────────
    // Precompute per-forcing base offsets into d_forc_data once
    // Layout: [forc0: nT0 × num_systems] [forc1: nT1 × num_systems] ...
    // Index = base[j] + sampleIdx * num_systems + sys
    // ─────────────────────────────────────────────────────────
    size_t forc_base[MAX_FORCINGS];
    {
        size_t acc = 0;
        for (int j = 0; j < nForc; ++j) {
            forc_base[j] = acc;
            acc += c_forc_nT[j] * size_t(num_systems);
        }
    }

    // (Optional but cheap) Cache dt_sec and nT locally to reduce constant-memory traffic
    // IMPORTANT: Do not change minute/second handling from original!
    double dt_sec_cache[MAX_FORCINGS]; // forcing cadences in SECONDS (same naming as original)
    size_t nT_cache[MAX_FORCINGS];
    for (int j = 0; j < nForc; ++j) {
        dt_sec_cache[j] = c_forc_dt[j]*60; // original kept as-is; do NOT modify units here
        nT_cache[j]     = c_forc_nT[j];
        if (!(dt_sec_cache[j] > 0.0) || nT_cache[j] == 0) {
            if (sys == 0) {
                printf("[KERNEL] invalid forcing meta: j=%d dt=%.6g nT=%zu\n",
                       j, dt_sec_cache[j], nT_cache[j]);
            }
            d_stiff[sys] = 7; // Config error
            return;
        }
    }

    // ──────────────────────────────────────────────
    // RK45 integration over [t0, tf] with event caps
    // ──────────────────────────────────────────────
    double t = t0;

    // If initialStep==0, let the controller grow but keep a tiny floor to avoid underflow.
    double h = devParams.initialStep;
    const double H_MIN = 1e-12; // seconds; tiny, just avoids 0 *anything = 0 traps
    if (h < H_MIN) h = H_MIN;

    // EPS for boundary comparisons to avoid FP “just negative” remainders
    const double EPS = 1e-12;

    // ── Stall fuse bookkeeping (constraint)
    double last_t_progress = t;
    const double T_PROG_EPS = 1e-12;  // in SECONDS: require strict increase to count as progress
    int    stall_iters = 0;
    // const int STALL_LIMIT = 5000;     // much tighter fuse

    // Cap how many consecutive slope-jump halvings we allow at the same t
    int jump_halves = 0;
    
    // ── Tiny-h accepted-steps fuse bookkeeping (fires even if err<=1)
    int    tiny_h_iters       = 0;
    const int    TINY_H_LIMIT = 150000;    // tunable: #iters with tiny h before bursting
    const double H_TINY_THRESHOLD = 1e-5;  // seconds; consider h "tiny" below this

    // // Global loop-iteration fuse (constraint)
    // int total_iters = 0;
    // const int MAX_TOTAL_ITERS = 100000;
    // ────────────────────────────────────────────────────────────────────────────────
    // SMART FUSE UPGRADE  (Approach 1 + 2)
    // -------------------------------------------------------------------------------
    //  • Budget iterations based on the smallest forcing cadence (dt_sec)
    //    → avoids false timeouts for long windows with many tiny steps.
    //  • Keep a stall fuse that fires only when time stops advancing
    //    → catches true infinite-loop conditions without penalizing progress.
    // -------------------------------------------------------------------------------
    long long total_iters = 0;                                         // ★ 64-bit counter

    // Find smallest forcing cadence (seconds)
    double min_dt = 1e300;
    for (int j = 0; j < nForc; ++j) min_dt = fmin(min_dt, dt_sec_cache[j]);  // ★
    // --- PATCH: Allow solver to shrink below 1 hour if needed to prevent timeouts
    if (min_dt > 60.0) {
        min_dt = 60.0;
    }

    // Estimate number of forcing "slabs" in this window
    const double window = fmax(tf - t0, 0.0);
    const long long n_slabs = (long long)ceil(window / fmax(min_dt, 1e-30)); // ★

    // Allow up to this many RK steps per forcing slab (tunable)
    // const long long STEPS_PER_SLAB_BUDGET = 4096;                       // ★
    const long long STEPS_PER_SLAB_BUDGET = 262144; //131072;  // 128k per forcing slab

    // // Compute data-dependent iteration budget with generous global cap
    // const long long MAX_TOTAL_ITERS =
    //     llmin((long long)5e8,
    //           llmax((long long)1e6, n_slabs * STEPS_PER_SLAB_BUDGET));   // ★

    // Compute data-dependent iteration budget with generous global cap (no llmin/llmax)  // ★
    long long tmp_iters = (long long)(n_slabs * STEPS_PER_SLAB_BUDGET);                                               // ★
    // if (tmp_iters < 1000000LL)   tmp_iters = 1000000LL;   // floor at 1e6     
    if (tmp_iters < 5000000LL)   tmp_iters = 5000000LL;   // floor at 5e6                                  // ★
    if (tmp_iters > 500000000LL) tmp_iters = 500000000LL; // cap at 5e8                                         // ★
    const long long MAX_TOTAL_ITERS = tmp_iters; 

    // Progress-based stall fuse
    const int HARD_STALL_LIMIT = 200000;   // ★ only fires on true spin


    while (t < tf && !stiff && total_iters < MAX_TOTAL_ITERS) {
        // Early bailout for systems stuck at t ≈ 0, h ≈ 0, iters > huge
        const long long MAX_EARLY_STALL_ITERS = 1000000000LL;  // 1e9
        if (t < 1e-8 && h <= 1e-12 && total_iters > MAX_EARLY_STALL_ITERS) {
            for (int i = 0; i < N_EQ; ++i) {
                y_final_all[sys * N_EQ + i] = 0.0f;
            }
            d_stiff[sys] = STIFF_CODE_EARLY_TIMEOUT;  // special code: early stall at t≈0
            atomicAdd(&stiff_kill_count, 1);
            if (sys % 100000 == 0)
                printf("[EARLY-STIFF] sys=%d stuck at t=%.3e h=%.1e → exiting early\n", sys, t, h);
            return;
        }

        ++total_iters;

        // Never step past tf
        if (t + h > tf) h = tf - t;

        // Near-final-time escape to prevent end-of-window bounce/stall (constraint)
        const double rem_tf = tf - t;       // seconds remaining
        const double REM_EPS = 1e-15;       // seconds
        if (rem_tf <= REM_EPS) {
            break; // effectively at tf
        }
        if (h > rem_tf) h = rem_tf;

        // ────────────────────────────────────────────────────────────────
        // Do not step across any forcing boundary (piecewise-constant RHS)
        // For each forcing j with cadence dt_sec[j], compute the next
        // change time t_next ≥ t, then cap h to the smallest positive gap.
        // (Keep original seconds-based logic; add snap-on-boundary constraint.)
        // ────────────────────────────────────────────────────────────────
        {
            const double INF = 1e300;
            double h_forc = INF;

            for (int j = 0; j < nForc; ++j) {
                const double dt_sec = dt_sec_cache[j];
                const double s      = (t - t0) / dt_sec;             // fractional sample index since epoch t0
                const double t_next = (floor(s) + 1.0) * dt_sec + t0; // time of next boundary for forcing j
                const double rem    = t_next - t;                     // gap until that boundary
                if (rem > EPS) {                                      // strictly in the future
                    h_forc = fmin(h_forc, rem);
                }
            }
            if (h_forc < INF) {
                // If we are essentially on the boundary, snap and continue (seconds)
                // const double BOUNDARY_SNAP_EPS_S = 3.6e-3; // ≈ 0.0036 s (tight but nonzero)
                const double BOUNDARY_SNAP_EPS_S = 1e-3 * min_dt; // 3.6 s for 1-hour cadence
                if (h_forc <= BOUNDARY_SNAP_EPS_S) {
                    t += h_forc;
                    Runoff5::project_nonnegative(y);
                    // Emit any queries now ≤ t
                    while (next_q < num_queries && query_times[next_q] <= t + EPS) {
                        const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                        DBG_ASSERT(next_q >= 0 && next_q < num_queries, "next_q OOB %d/%d", next_q, num_queries);
                        DBG_ASSERT(flat_q >= 0 &&
                                   flat_q < (long long)num_systems * (long long)num_queries,
                                   "dense idx OOB: flat_q=%lld ns*nq=%lld",
                                   flat_q, (long long)num_systems * (long long)num_queries);
                        const long long base_idx = flat_q * N_EQ;
                        double y_clamped[N_EQ];
                        for (int c = 0; c < N_EQ; ++c) y_clamped[c] = y[c];
                        Runoff5::project_nonnegative(y_clamped);
                        for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = static_cast<float>(y_clamped[c]);
                        ++next_q;
                    }
                    reject_count = 0; // safe to clear when we land on a boundary
                    continue; // start a fresh step from the boundary
                }
                // Otherwise cap h to stay within the current forcing slab
                if (h > h_forc) h = h_forc;

                // IMPORTANT: don’t carry reject penalties across a discontinuity (original behavior kept)
                reject_count = 0;
            }
        }

         // ────────────────────────────────────────────────────────────────
         // Tiny-h fuse: if h stays microscopic for a long time (even on accepts),
         // force a short RK4 burst to jump out of the stall region, or mark stiff.
         // ────────────────────────────────────────────────────────────────
         if (h < H_TINY_THRESHOLD) {
             if (++tiny_h_iters > TINY_H_LIMIT) {
                 // Scale burst to forcing cadence so it actually moves
                 const double HFIX     = fmin(0.001 * min_dt, 30.0);   // ≤0.1% cadence, ≤30 s
                 const double MIN_ADV  = fmin(0.02  * min_dt, 120.0);  // 2% cadence, ≤2 min
                 const int    NFIX_MAX = 500;
 
                 double t_start = t;
                 int steps = 0;
                 while (t < tf && (t - t_start) + 1e-12 < MIN_ADV && steps < NFIX_MAX) {
                     double gap_local = 1e300;
                     for (int j = 0; j < nForc; ++j) {
                         const double ss     = (t - t0) / dt_sec_cache[j];
                         const double t_next = (floor(ss) + 1.0) * dt_sec_cache[j] + t0;
                         const double g      = t_next - t;
                         if (g > 0.0 && g < gap_local) gap_local = g;
                     }
                     double hs = (gap_local < HFIX ? gap_local : HFIX);
                     if (hs <= 0.0) break;
 
                    //  rk4_fixed_step_seconds<Runoff5>(
                    //      t, y, N_EQ, hs, sys, t0, num_systems,
                    //      forc_base, dt_sec_cache, nT_cache, d_sp, nForc);
    if (!used_ros2w && sys == 0)
        printf("[SOLVER] sys=%d switching to ROS2W (tiny-h fallback) at t=%.4f\n", sys, t);
    used_ros2w = true;

                        if (!ros2w_step_seconds<Runoff5>(
                                t, y, N_EQ, hs, sys, t0, num_systems,
                                forc_base, dt_sec_cache, nT_cache, d_sp, nForc)) {
                            hs *= 0.5;
                            if (hs <= 0.0 || !ros2w_step_seconds<Runoff5>(
                                    t, y, N_EQ, hs, sys, t0, num_systems,
                                    forc_base, dt_sec_cache, nT_cache, d_sp, nForc)) {
                                stiff = true;
                                break;
                            }
                        }
                     ++steps;
 
                     // Emit any queries we passed (no interpolation inside micro-step)
                     while (next_q < num_queries && query_times[next_q] <= t + EPS) {
                         const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                         DBG_ASSERT(next_q >= 0 && next_q < num_queries, "next_q OOB %d/%d", next_q, num_queries);
                         const long long base_idx = flat_q * N_EQ;
                         double y_clamped[N_EQ];
                         for (int c = 0; c < N_EQ; ++c) y_clamped[c] = y[c];
                         Runoff5::project_nonnegative(y_clamped);
                         for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = (float)y_clamped[c];
                         ++next_q;
                     }
                 }
 
                 if (t - t_start >= 1e-7) {
                     // We advanced → resume RK45 with a sane step and reset counters
                     h = fmax(0.02, devParams.initialStep); // ~0.02 s
                     reject_count = 0;
                     stall_iters  = 0;
                     tiny_h_iters = 0;
                     continue; // restart loop from (t,y)
                 } else {
                     // Couldn’t move even with burst → bail to stiff and let host handle fallback
                     Runoff5::project_nonnegative(y);
                     for (int i = 0; i < N_EQ; ++i) y_final_all[sys * N_EQ + i] = y[i];
                     d_stiff[sys] = 1; // stiff
                     printf("[STIFF-FLAG] sys=%d stiff=%d (tiny-h crawl)\n", sys, d_stiff[sys]);
                     return;
                 }
             }
         } else {
             tiny_h_iters = 0; // reset when h isn’t tiny
         }

        // ────────────────────────────────────────────────────────────────
        // Gather forcings at current time t (sample-and-hold semantics)
        // (seconds, identical to original semantics; add NaN guard constraint)
        // ────────────────────────────────────────────────────────────────
        float Fval_arr[MAX_FORCINGS];
        for (int j = 0; j < nForc; ++j) {
            const double dt_sec  = dt_sec_cache[j];
            const double s_real  = (t - t0) / dt_sec;   // fractional index
            const size_t nS      = nT_cache[j];
            // Clamp to valid sample range [0, nS-1]
            size_t sampleIdx = (s_real < 0.0) ? size_t(0)
                                : (s_real >= double(nS)) ? (nS - 1)
                                : size_t(s_real); // truncates toward zero = floor for s_real >= 0

            const size_t idx = forc_base[j] + sampleIdx * size_t(num_systems) + size_t(sys);
            Fval_arr[j] = d_forc_data[idx];
        }
        for (int j = 0; j < nForc; ++j) {
            if (!::isfinite(Fval_arr[j])) {
                d_stiff[sys] = 7; // config error
                printf("[FORCING-NAN] sys=%d forc=%d val=%f at t=%.9f\n",
                       sys, j, Fval_arr[j], t);
                return;
            }
        }

        // RHS at stage 0 (k1) with current forcings
        Runoff5::rhs(t, y, k45[0], N_EQ, sys, d_sp, Fval_arr, nForc);

        // One adaptive RK45 step attempt (fills y_next, k45, err)
        rk45_step<Runoff5>(t, y, y_next, N_EQ, h, rtol, atol, &err,
                           k45, sys, d_sp, Fval_arr, nForc);

        // Add this right after the rk45_step call
        // if (total_iters > MAX_TOTAL_ITERS || isnan(h)) {
        //     d_stiff[sys] = 5;
        //     printf("[TIMEOUT] sys=%d streamID=%ld t=%.6f h=%.3e iters=%lld\n",
        //         sys, stream_ids[sys], t, h, total_iters);
        //     return;
        // }
        // Timeout → zero outputs for this system, flag, fence, and return
        if (total_iters > MAX_TOTAL_ITERS || isnan(h)) {
            write_final_zero<N_EQ>(sys, y_final_all);
            d_stiff[sys] = 5;
            __threadfence();
            printf("[TIMEOUT] sys=%d streamID=%ld t=%.6f h=%.3e iters=%lld\n",
                sys, stream_ids[sys], t, h, total_iters);
            return;
        }

        // [A] ──NaN/Inf checks (constraint)
        if (!::isfinite(err)) {
            Runoff5::project_nonnegative(y);
            for (int i=0;i<N_EQ;++i) y_final_all[sys*N_EQ+i] = y[i];
            d_stiff[sys] = 2; // NaN LTE
            printf("[STIFF-FLAG] sys=%d stiff=%d (NaN LTE)\n", sys, d_stiff[sys]);
            return;
        }
        // runaway error guard (prevents super-long reject loops)
        if (err > 1e50) {
            stiff = true;
            break;
        }
        for (int c=0;c<N_EQ;++c) {
            if (!::isfinite(y_next[c])) {
                Runoff5::project_nonnegative(y);
                for (int i=0;i<N_EQ;++i) y_final_all[sys*N_EQ+i] = y[i];
                d_stiff[sys] = 3; // NaN state
                printf("[STIFF-FLAG] sys=%d stiff=%d (NaN state)\n", sys, d_stiff[sys]);
                return;
            }
        }

        // ---- Stall fuse: detect lack of time progress over many iterations (constraint)
        // if (t > last_t_progress + T_PROG_EPS) {
        //     last_t_progress = t;
        //     stall_iters = 0;
        // } else {
        //     if (++stall_iters > STALL_LIMIT) {
        //         Runoff5::project_nonnegative(y);
        //         for (int i = 0; i < N_EQ; ++i) {
        //             y_final_all[sys * N_EQ + i] = y[i];
        //         }
        //         d_stiff[sys] = 5; // Fuse: exceeded stall iteration limit
        //         printf("[STIFF-FLAG] sys=%d code=5 (stall) t=%.15g h=%.15g iters=%d\n",
        //                sys, t, h, stall_iters);
        //         return;
        //     }
        // }

        // ★ Stall fuse now tied to *lack of progress* instead of iteration count.
        if (t > last_t_progress + T_PROG_EPS) {
            last_t_progress = t;
            stall_iters = 0;
         } else if (++stall_iters > HARD_STALL_LIMIT) {
            Runoff5::project_nonnegative(y);
            // for (int i = 0; i < N_EQ; ++i)
            //     y_final_all[sys * N_EQ + i] = y[i];
            // d_stiff[sys] = 5;
            // printf("[TIMEOUT] sys=%d streamID=%ld stall_iters=%d t=%.9f h=%.3e\n",
            //        sys,stream_ids[sys],stall_iters, t, h);
            // Hard stall → zero outputs, flag, fence, and return
            write_final_zero<N_EQ>(sys, y_final_all);
            d_stiff[sys] = 5;
            __threadfence();
            printf("[TIMEOUT] sys=%d streamID=%ld stall_iters=%d t=%.9f h=%.3e\n",
                sys,stream_ids[sys],stall_iters, t, h);
            return;
        }

        if (err <= 1.0) {
            // ───────────── Accepted step ─────────────
            reject_count = 0;
            jump_halves  = 0;

            // Slope-jump heuristic (constraint)
            const double jump = norm_inf_diff(k45[0], k45[1], N_EQ);
            if (jump > SLOPE_JUMP_THRESH) {
                ++jump_halves;
                h = fmax(0.5 * h, devParams.initialStep * MIN_STEP_FRACTION);
                if (h < H_MIN) h = H_MIN;
                if (jump_halves > 8 || h < H_MIN_ABS) { stiff = true; break; }
                ++reject_count; // treat as controlled reject to allow bail if pathological
                continue; // retry from same t with smaller h
            }

            // Dense output and query emission for (t, t+h]
            const double t1 = t + h;
            while (next_q < num_queries && query_times[next_q] <= t1) {
                const double tq = query_times[next_q];

                // If tq <= t (+EPS), emit the current state without interpolation.
                if (tq <= t + EPS) {
                    // Clamp for physical validity (matches later behavior)
                    double y_clamped[N_EQ];
                    for (int c = 0; c < N_EQ; ++c) y_clamped[c] = y[c];
                    Runoff5::project_nonnegative(y_clamped);

                    // [B1] ─── bounds before writing ────────────────────
                    const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                    DBG_ASSERT(next_q >= 0 && next_q < num_queries, "next_q OOB %d/%d", next_q, num_queries);
                    DBG_ASSERT(flat_q >= 0 &&
                            flat_q < (long long)num_systems * (long long)num_queries,
                            "dense idx OOB: flat_q=%lld ns*nq=%lld",
                            flat_q, (long long)num_systems * (long long)num_queries);
                    const long long base_idx = flat_q * N_EQ;
                    // ────────────────────────────────────────────────────

                    for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = static_cast<float>(y_clamped[c]);
                } else {
                    // Interpolate within the step at th = (tq - t)/h
                    const double th = (tq - t) / h;
                    double yd[N_EQ];
                    rk45_dense<Runoff5>(y, k45, N_EQ, h, th, yd);
                    Runoff5::project_nonnegative(yd);

                    const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                    DBG_ASSERT(next_q >= 0 && next_q < num_queries, "next_q OOB %d/%d", next_q, num_queries);
                    DBG_ASSERT(flat_q >= 0 &&
                            flat_q < (long long)num_systems * (long long)num_queries,
                            "dense idx OOB: flat_q=%lld ns*nq=%lld",
                            flat_q, (long long)num_systems * (long long)num_queries);
                    const long long base_idx = flat_q * N_EQ;
                    for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = static_cast<float>(yd[c]);
                }

                ++next_q;
            }

            // (optional belt-and-suspenders)
            DBG_ASSERT(next_q <= num_queries, "next_q overflow: %d/%d", next_q, num_queries);

            // Accept and advance time/state
            Runoff5::project_nonnegative(y_next); // safety clamp for carried state
            for (int i = 0; i < N_EQ; ++i) y[i] = y_next[i];
            t = t1;

            // Standard PI-like step-size update with growth/decay clamps
            double fac = devParams.safety * pow(1.0 / (err + 1e-16), 0.2);
            fac = fmin(fmax(fac, devParams.minScale), devParams.maxScale);
            h *= fac;
            if (h < H_MIN) h = H_MIN; // keep nonzero

             // Relative floor tied to forcing cadence: prevents controller pinning at μs scale
             const double H_REL_FLOOR = 1e-6 * min_dt;  // 1 ppm of smallest cadence
             if (h < H_REL_FLOOR) h = H_REL_FLOOR;

        } else {
            // ───────────── Rejected step ─────────────
            ++reject_count;

            // On reject we only allow shrink (common practice)
            double fac = devParams.safety * pow(1.0 / (err + 1e-16), 0.2);
            fac = fmin(fac, 1.0);
            fac = fmin(fmax(fac, devParams.minScale), devParams.maxScale);
            h *= fac;
            if (h < H_MIN) h = H_MIN;

            // ── Trigger “micro-fallback burst” if pathological (constraint)
            const int STALL_TRIGGER = 2000;  // ~#iters with no time progress
            if (reject_count > REJECT_LIMIT || h < H_MIN_ABS || stall_iters > STALL_TRIGGER) {
                if (sys == 0) {
                    printf("[TRIP] rej=%d h=%.3e stall=%d t=%.9f\n",
                           reject_count, h, stall_iters, t);
                }

                // seconds (keep units consistent with original code)
                // const double HFIX     = 0.06; // 0.06 s per micro-step
                const double HFIX  = fmin(0.10 * min_dt, 300.0); // ≤10% cadence, ≤5 min !!!
                const double MIN_ADV  = 1.2;  // 1.2 s net progress
                const int    NFIX_MAX = 200;

                double t_start = t;

                if (FALLBACK_TO_TF) {
                    // Finish the rest of the window with small RK4 steps,
                    // respecting forcing boundaries and emitting queries.
                    while (t < tf) {
                        // Cap by next forcing boundary
                        double gap_local = 1e300;
                        for (int j = 0; j < nForc; ++j) {
                            const double ss     = (t - t0) / dt_sec_cache[j];
                            const double t_next = (floor(ss) + 1.0) * dt_sec_cache[j] + t0;
                            const double g      = t_next - t;
                            if (g > 0.0 && g < gap_local) gap_local = g;
                        }
                        double hs = (gap_local < HFIX ? gap_local : HFIX);
                        if (t + hs > tf) hs = tf - t;
                        if (hs <= 0.0) break;

                        // rk4_fixed_step_seconds<Runoff5>(
                        //     t, y, N_EQ, hs, sys, t0, num_systems,
                        //     forc_base, dt_sec_cache, nT_cache, d_sp, nForc);

                    // Logging when we switch to fallback
                    if (!used_ros2w && sys == 0){
                        printf("[SOLVER] sys=%d switching to ROS2W (reject-limit fallback) at t=%.4f\n", sys, t);
                    }
                    used_ros2w = true;


                     if (!ros2w_step_seconds<Runoff5>(
                             t, y, N_EQ, hs, sys, t0, num_systems,
                             forc_base, dt_sec_cache, nT_cache, d_sp, nForc)) {
                        // // Try once with half step; if still bad, bail as stiff
                        //  hs *= 0.5;
                        //  if (hs <= 0.0 || !ros2w_step_seconds<Runoff5>(
                        //          t, y, N_EQ, hs, sys, t0, num_systems,
                        //          forc_base, dt_sec_cache, nT_cache, d_sp, nForc)) {
                        //      return;
                        //  }
                        // Try once with half step; if still bad, write what we have,
                        // flag the system (distinct code) and return so the host can
                        // see the failure instead of losing the result.
                        hs *= 0.5;
                        if (hs <= 0.0 || !ros2w_step_seconds<Runoff5>(
                                t, y, N_EQ, hs, sys, t0, num_systems,
                                forc_base, dt_sec_cache, nT_cache, d_sp, nForc)) {
                            // Ensure we write a safe final state and set d_stiff so host
                            // can diagnose that fallback failed for this system.
                            Runoff5::project_nonnegative(y);
                            for (int _i = 0; _i < N_EQ; ++_i) {
                                y_final_all[sys * N_EQ + _i] = y[_i];
                            }
                            d_stiff[sys] = 99; // distinct code for "fallback failed"
                            if (sys == 0) {
                                printf("[SOLVER] sys=%d fallback FAILED at t=%.9f hs=%.3e -> d_stiff=99\n",
                                    sys, t, hs);
                            }
                            return;
                        }
                    }

                        // Emit any queries that are now ≤ t (no interpolation inside micro-step)
                        while (next_q < num_queries && query_times[next_q] <= t + EPS) {
                            const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                            DBG_ASSERT(next_q >= 0 && next_q < num_queries, "next_q OOB %d/%d", next_q, num_queries);
                            const long long base_idx = flat_q * N_EQ;
                            double y_clamped[N_EQ];
                            for (int c = 0; c < N_EQ; ++c) y_clamped[c] = y[c];
                            Runoff5::project_nonnegative(y_clamped);
                            for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = (float)y_clamped[c];
                            ++next_q;
                        }
                    }
                    // Write final and exit “successfully finished by fallback”
                    Runoff5::project_nonnegative(y);
                    for (int i = 0; i < N_EQ; ++i) y_final_all[sys * N_EQ + i] = y[i];
                    if (sys == 0) {
                        printf("[SOLVER] sys=%d completed with fixed-step fallback to tf = %.2f\n", sys, tf);
                        }
                    finished_by_fallback = true;
                    d_stiff[sys] = 6; // Fallback finished (not an error)

                    return;
                } else {
                    // Burst mode: advance by MIN_ADV (but not past next boundary),
                    // then resume RK45 with a reset controller.
                    double adv_target = MIN_ADV;
                    double min_gap = 1e300;
                    for (int j = 0; j < nForc; ++j) {
                        const double ss     = (t - t0) / dt_sec_cache[j];
                        const double t_next = (floor(ss) + 1.0) * dt_sec_cache[j] + t0;
                        const double gap    = t_next - t;
                        if (gap > 0.0 && gap < min_gap) min_gap = gap;
                    }
                    if (min_gap < adv_target) adv_target = min_gap;

                    int steps = 0;
                    while (t < tf && (t - t_start) + 1e-12 < adv_target && steps < NFIX_MAX) {
                        double gap_local = 1e300;
                        for (int j = 0; j < nForc; ++j) {
                            const double ss     = (t - t0) / dt_sec_cache[j];
                            const double t_next = (floor(ss) + 1.0) * dt_sec_cache[j] + t0;
                            const double g      = t_next - t;
                            if (g > 0.0 && g < gap_local) gap_local = g;
                        }
                        double hs = (gap_local < HFIX ? gap_local : HFIX);
                        if (hs <= 0.0) break;

                        // rk4_fixed_step_seconds<Runoff5>(
                        //     t, y, N_EQ, hs, sys, t0, num_systems,
                        //     forc_base, dt_sec_cache, nT_cache, d_sp, nForc);
                        if (!ros2w_step_seconds<Runoff5>(
                                t, y, N_EQ, hs, sys, t0, num_systems,
                                forc_base, dt_sec_cache, nT_cache, d_sp, nForc)) {
                            // halve once and retry; if still failing, mark stiff and exit
                            hs *= 0.5;
                            if (hs <= 0.0 || !ros2w_step_seconds<Runoff5>(
                                    t, y, N_EQ, hs, sys, t0, num_systems,
                                    forc_base, dt_sec_cache, nT_cache, d_sp, nForc)) {
                                d_stiff[sys] = 1;
                                return;
                            }
                        }


                        ++steps;
                    }

                    if (t - t_start >= 1e-9) {
                        // Emit any queries that we “passed” during burst (no interpolation)
                        while (next_q < num_queries && query_times[next_q] <= t + EPS) {
                            const long long flat_q = ((long long)sys * (long long)num_queries + (long long)next_q);
                            DBG_ASSERT(next_q >= 0 && next_q < num_queries, "next_q OOB %d/%d", next_q, num_queries);
                            const long long base_idx = flat_q * N_EQ;
                            double y_clamped[N_EQ];
                            for (int c = 0; c < N_EQ; ++c) y_clamped[c] = y[c];
                            Runoff5::project_nonnegative(y_clamped);
                            for (int c = 0; c < N_EQ; ++c) dense_all[base_idx + c] = (float)y_clamped[c];
                            ++next_q;
                        }
                        // Resume RK45 with a modest step
                        h = fmax(0.02, devParams.initialStep); // ~0.02 s minimum (seconds)
                        reject_count = 0;
                        stall_iters  = 0;  // reset stagnation counter
                        continue;
                    } else {
                        stiff = true; // fallback couldn’t move → bail as before
                    }
                }
            } // end trip
        } // end reject/accept

    } // ← end while (t < tf && !stiff && total_iters < MAX_TOTAL_ITERS)

    // if (total_iters >= MAX_TOTAL_ITERS) {
    //     // d_stiff[sys] = 5;  // fuse
    //     // printf("[TIMEOUT] sys=%d hit iteration limit at t=%.9f\n", sys, t);
    //     // return;
    //     // ★ Iteration budget exhausted → likely pathological, but print context
    //     d_stiff[sys] = 5;
    //     printf("[TIMEOUT] sys=%d streamID=%ld iters=%lld window=%.6g s min_dt=%.6g s t=%.9f h=%.3e\n",
    //            sys,stream_ids[sys],total_iters, window, min_dt, t, h);
    //     return;
    // }
    // if (total_iters >= MAX_TOTAL_ITERS) {
    //     d_stiff[sys] = 5;
    //     printf("[SOLVER] sys=%d streamID=%ld hit iteration limit → TIMEOUT (iters=%lld, t=%.6f, h=%.3e)\n",
    //            sys, total_iters, t, h);
    //     return;
    // }

    if (total_iters >= MAX_TOTAL_ITERS) {
        // Loop-tail timeout → zero outputs, flag, fence, and return
        write_final_zero<N_EQ>(sys, y_final_all);
        d_stiff[sys] = 5;
        __threadfence();
        printf("[SOLVER] sys=%d streamID=%ld hit iteration limit → TIMEOUT (iters=%lld, t=%.6f, h=%.3e)\n",
            sys, stream_ids[sys], (long long)total_iters, t, h);
        return;
        }

    // ─── 2) Flag stiff and bail ───
    if (stiff && t < tf) {
        // write the *current* y to y_final_all so host doesn’t get zeros
        Runoff5::project_nonnegative(y);
        for (int i = 0; i < N_EQ; ++i) {
            y_final_all[sys * N_EQ + i] = y[i];
        }
        d_stiff[sys] = 1; // stiff flag
        // printf("[STIFF-FLAG] sys=%d streamID=%ld stiff=%d (t=%.9f h=%.3e)\n", sys, d_stiff[sys], t, h);
        printf("[STIFF-FLAG] sys=%d streamID=%llu stiff=%d (t=%.9f h=%.3e)\n",sys, (unsigned long long) stream_ids[sys], d_stiff[sys], t, h);
       return; // Radau would be handled out-of-kernel in this design
    }

    // ─── 2.9) Success marker ───
    // if (!used_ros2w && !finished_by_fallback && d_stiff[sys] == 0) {
    //     printf("[SOLVER] sys=%d used RK45 successfully (no fallback)\n", sys);
    // }


    // ─── 3) Never stiff: write final state ───
    Runoff5::project_nonnegative(y);
    for (int i = 0; i < N_EQ; ++i) {
        // Checking if an overwrite occurs
        int offset = sys * N_EQ + i;
        if (fabs(y_final_all[offset] + 999.0) > 1e-6) {
            printf("[OVERWRITE WARNING] sys=%d i=%d y_final_all[%d] already = %.5f\n",
                sys, i, offset, y_final_all[offset]);
        }
        y_final_all[offset] = y[i];

        y_final_all[sys * N_EQ + i] = y[i];
    }

    // if (sys == 0) {
    //     printf("[SOLVER-END] sys=%d done → d_stiff=%d (used_ros2w=%d, fallback=%d)\n",
    //            sys, d_stiff[sys], used_ros2w, finished_by_fallback);
    // }

    // Mark as successful RK45 completion
    d_stiff[sys] = 0;

        // --- [BLOCK SUMMARY PRINT] ---
    __syncthreads();
    if (threadIdx.x == 0 && stiff_kill_count > 0) {
        printf("[BLOCK-STIFF] block %d: %d early-stiff systems exited early (code=66)\n",
               blockIdx.x, stiff_kill_count);
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
    float*,           // dense_all
    int,              // num_systems
    int,              // num_queries
    double,           // t0
    double,           // tf
    const Runoff5::SP_TYPE*, // d_sp
    int*,              // d_stiff
    const long*        // stream_ids
);


