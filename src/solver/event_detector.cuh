// src/solver/event_detector.cuh
#pragma once

#include <cmath>

// -----------------------------------------------------------------------------
// Thresholds for detecting events and stiffness
// -----------------------------------------------------------------------------

// How big a jump in the slope signals a kink?
static constexpr double SLOPE_JUMP_THRESH = 100.0;

// When the adaptive step size h falls below this fraction of the total
// interval, declare the problem “stiff” and switch to an implicit solver.
static constexpr double MIN_STEP_FRACTION = 1e-6;

// -----------------------------------------------------------------------------
// Helpers: ∞-norm and ∞-norm of difference (device-only, inlined & static)
// -----------------------------------------------------------------------------

/**
 * Compute the infinity-norm of a length-N vector `v[]`.
 *
 * __forceinline__ __device__:
 *   - __device__   : makes this function callable from other __device__/
 *                    __global__ code on the GPU.
 *   - __forceinline__: hints the compiler to inline aggressively,
 *                      avoiding a standalone symbol.
 *   - static       : internal linkage, so each .cu translation unit
 *                    gets its own copy and no global symbol is exported.
 */
__forceinline__ __device__ static double norm_inf(const double* v, int N) {
    double m = 0.0;
    for (int i = 0; i < N; ++i) {
        double f = fabs(v[i]);
        if (f > m) m = f;
    }
    return m;
}

/**
 * Compute the infinity-norm of the difference between two length-N arrays.
 *
 * Returns max_i |a[i] - b[i]|.
 */
__forceinline__ __device__ static double norm_inf_diff(const double* a, const double* b, int N) {
    double m = 0.0;
    for (int i = 0; i < N; ++i) {
        double f = fabs(a[i] - b[i]);
        if (f > m) m = f;
    }
    return m;
}
