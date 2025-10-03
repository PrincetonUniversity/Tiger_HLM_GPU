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
    float*  dense_all,    // [num_systems × num_queries × N_EQ]
    int     num_systems,  // number of systems to solve
    int     num_queries,  // number of queries to answer
    double  t0,           // initial time (seconds)
    double  tf,           // final time   (seconds)
    const typename Runoff5::SP_TYPE* d_sp, // per-system parameters
    int*    d_stiff       // [num_systems] stiffness flags
) {
    // Re-declare device symbols (defined elsewhere)
    extern __device__   float*   d_forc_data;   // flattened forcing values
    extern __constant__ double   c_forc_dt[];   // forcing cadence in MINUTES
    extern __constant__ size_t   c_forc_nT[];   // number of time samples per forcing
    extern __constant__ int      nForc;         // number of distinct forcings

    constexpr int N_EQ = Runoff5::N_EQ;
    const int sys = blockIdx.x * blockDim.x + threadIdx.x;
    const int REJECT_LIMIT = 20;            // was 5
    const double H_MIN_ABS = 1e-10;         // minutes (convert if you store seconds)
    if (sys >= num_systems) return;

    // Safety: ensure global pointer to forcing slab is valid
    if (!d_forc_data) {
        if (sys == 0) {
            printf("[KERNEL] d_forc_data is null; skipping computation.\n");
        }
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
    double dt_sec_cache[MAX_FORCINGS]; // forcing cadences in SECONDS
    size_t nT_cache[MAX_FORCINGS];
    for (int j = 0; j < nForc; ++j) {
        dt_sec_cache[j] = c_forc_dt[j];// * 60.0; // c_forc_dt is minutes → convert to seconds !!! changed to mins
        nT_cache[j]     = c_forc_nT[j];
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

    while (t < tf && !stiff) {
        // Never step past tf
        if (t + h > tf) h = tf - t;

        // ────────────────────────────────────────────────────────────────
        // Do not step across any forcing boundary (piecewise-constant RHS)
        // For each forcing j with cadence dt_sec[j], compute the next
        // change time t_next ≥ t, then cap h to the smallest positive gap.
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
            // if (h_forc < INF) h = fmin(h, h_forc);
            if (h_forc < INF) {
                // cap
                double rem = h_forc;
                if (rem < h) h = rem;

                // snap when extremely close to boundary to avoid tiny leftover
                if (fabs(rem - h) < 1e-10) h = rem;

                // IMPORTANT: don’t carry reject penalties across a discontinuity
                reject_count = 0;
            }
        }

        // ────────────────────────────────────────────────────────────────
        // Gather forcings at current time t (sample-and-hold semantics)
        // We select the sample index floor((t - t0)/dt_sec) clamped to [0, nT-1]
        // and read the value for this system.
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
        

        // RHS at stage 0 (k1) with current forcings
        Runoff5::rhs(t, y, k45[0], N_EQ, sys, d_sp, Fval_arr, nForc);

        // One adaptive RK45 step attempt (fills y_next, k45, err)
        rk45_step<Runoff5>(t, y, y_next, N_EQ, h, rtol, atol, &err,
                           k45, sys, d_sp, Fval_arr, nForc);

        // [A] ──NaN/Inf checks go HERE ──────────────────────────────────────
        if (!::isfinite(err)) {
            Runoff5::project_nonnegative(y);
            for (int i=0;i<N_EQ;++i) y_final_all[sys*N_EQ+i] = y[i];
            d_stiff[sys] = 2; // NaN LTE
            printf("[STIFF-FLAG] sys=%d stiff=%d (NaN LTE)\n", sys, d_stiff[sys]);
            return;
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
        // -─────────────────────────────────────────────────────────────────

        if (err <= 1.0) {
            // ───────────── Accepted step ─────────────
            reject_count = 0;

            // Slope-jump heuristic: if the first two stage slopes differ a lot,
            // reduce step proactively to better resolve fast changes.
            const double jump = norm_inf_diff(k45[0], k45[1], N_EQ);
            if (jump > SLOPE_JUMP_THRESH) {
                h = fmax(h * 0.5, devParams.initialStep * MIN_STEP_FRACTION);
                if (h < H_MIN) h = H_MIN;
                continue; // retry from same t with smaller h
            }

            // Dense output and query emission for (t, t+h] — include left endpoint if a query equals t
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

                    // const long long base_idx = ((long long)sys * num_queries + next_q) * N_EQ;
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

        } else {
            // ───────────── Rejected step ─────────────
            ++reject_count;

            // On reject we only allow shrink (common practice)
            double fac = devParams.safety * pow(1.0 / (err + 1e-16), 0.2);
            fac = fmin(fac, 1.0);
            fac = fmin(fmax(fac, devParams.minScale), devParams.maxScale);
            h *= fac;
            if (h < H_MIN) h = H_MIN;

            // Bail to implicit if too many rejects or step is ridiculously small
            // if (reject_count > 5 || h < (tf - t0) * MIN_STEP_FRACTION) {
            //     stiff = true;
            // }
            if (reject_count > REJECT_LIMIT || h < H_MIN_ABS) {
                stiff = true;
            }

        }
    }

    // ─── 2) Flag stiff and bail ───
    if (stiff && t < tf) {
        // write the *current* y to y_final_all so host doesn’t get zeros
        Runoff5::project_nonnegative(y);
        for (int i = 0; i < N_EQ; ++i) {
            y_final_all[sys * N_EQ + i] = y[i];
        }
        d_stiff[sys] = 1; // stiff flag
        printf("[STIFF-FLAG] sys=%d stiff=%d (t=%.3f h=%.3e)\n", sys, d_stiff[sys], t, h);
        return; // Radau would be handled out-of-kernel in this design
    }

    // ─── 3) Never stiff: write final state ───
    Runoff5::project_nonnegative(y);
    for (int i = 0; i < N_EQ; ++i) {
        y_final_all[sys * N_EQ + i] = y[i];
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


