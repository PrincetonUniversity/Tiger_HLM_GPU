// src/solver/rk45_step_dense.cuh
#pragma once

#include <math.h>
#include <cuda_runtime.h>

/**
 * Single‐step Dormand–Prince RK45 (no dense output) for a Model.
 * This function is templated on Model so it knows Model::N_EQ at compile time.
 *
 * The caller (rk45_kernel_multi) must fill k[0] by calling Model::rhs(t, y, k[0], N_EQ),
 * then call this to compute k[1..6], y_out, and error_norm.
 *
 * Template parameters:
 *   Model::N_EQ  → number of equations
 *   Model::rhs   → the device RHS function
 *   Model::devParams → the Parameters struct in constant memory
 */
template <class Model>
__device__ void rk45_step(
    double t,                            // current time
    const double* y,                     // current state [n]
    double* y_out,                       // next state [n]
    int n,                               // must equal Model::N_EQ
    double h,                            // timestep
    double* error_norm,                  // L₂ error estimate
    double k[7][Model::N_EQ]             // stage slopes (output)
) {
    constexpr int N_EQ = Model::N_EQ;

    // Dormand–Prince coefficients
    // Coefficients for the Butcher tableau (c vector)
    static constexpr double c[7] = {
        0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
    };

    // Coefficients for the Butcher tableau (a matrix)
    static constexpr double a[7][6] = {
        {},
        {1.0/5.0},
        {3.0/40.0, 9.0/40.0},
        {44.0/45.0, -56.0/15.0, 32.0/9.0},
        {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0},
        {9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0},
        {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0}
    };

    // Coefficients for the 5th‐order solution
    static constexpr double b[7] = {
        35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0,
        -2187.0/6784.0, 11.0/84.0, 0.0
    };

    // Coefficients for the 4th‐order embedded solution
    static constexpr double b_alt[7] = {
        5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
        -92097.0/339200.0, 187.0/2100.0, 1.0/40.0
    };

    double y_temp[N_EQ];

    // ─── Stage 1: Compute k[0] ───
    Model::rhs(t, y, k[0], N_EQ);

    // ─── Stage 2: Compute k[1..6] ───
    for (int s = 1; s < 7; ++s) {
        // Build y_temp = y + h * Σ_{j< s} a[s][j] * k[j][i]
        for (int i = 0; i < N_EQ; ++i) {
            double acc = y[i];
            for (int j = 0; j < s; ++j) {
                acc += h * a[s][j] * k[j][i];
            }
            y_temp[i] = acc;
        }
        // Call the model's RHS at t + c[s] * h:
        Model::rhs(t + c[s] * h, y_temp, k[s], N_EQ);
    }

    // ─── Stage 3: Build y_out (5th‐order) ───
    for (int i = 0; i < N_EQ; ++i) {
        double acc = y[i];
        for (int s = 0; s < 7; ++s) {
            acc += h * b[s] * k[s][i];
        }
        y_out[i] = acc;
    }

    // // ─── Stage 4: Compute error_norm (L₂ norm) ───
    // double err_sum = 0.0;
    // for (int i = 0; i < N_EQ; ++i) {
    //     double y_err = 0.0;
    //     for (int s = 0; s < 7; ++s) {
    //         y_err += h * (b[s] - b_alt[s]) * k[s][i];
    //     }
    //     double tol = 1e-9 + 1e-6 * fmax(fabs(y[i]), fabs(y_out[i]));
    //     err_sum += (y_err / tol) * (y_err / tol);
    // }
    // *error_norm = sqrt(err_sum / N_EQ);

    // ─── Stage 4: compute error_norm (∞–norm) ───
    double max_ratio = 0.0;
    for (int i = 0; i < N_EQ; ++i) {
        double y_err = 0.0;
        for (int s = 0; s < 7; ++s) {
            y_err += h * (b[s] - b_alt[s]) * k[s][i];
        }
        double tol = 1e-9 + 1e-6 * fmax(fabs(y[i]), fabs(y_out[i]));
        double ratio = fabs(y_err) / tol;
        if (ratio > max_ratio) {
            max_ratio = ratio;
        }
    }
    *error_norm = max_ratio;
}

/**
 * Single‐step Dormand–Prince RK45 dense‐output interpolation for a Model.
 *
 * Given the slopes k[0..6] from an already‐computed RK45 step, this computes:
 *   y_dense = y + h * Σ_{j=0..6} [b[j] + θ·d[j]] * k[j][i],
 * which is 5th‐order accurate at t + θ·h.
 *
 * Template parameters:
 *   Model::N_EQ  → number of equations
 */
template <class Model>
__device__ void rk45_dense(
    const double* y,                   // state at time t
    double k[7][Model::N_EQ],          // stage slopes from the step
    int n,                             // must equal Model::N_EQ
    double h,                          // timestep used in that step
    double theta,                      // normalized interpolation point in [0,1]
    double* dense_out                  // output: interpolated state [n]
) {
    // Coefficients for the 5th‐order solution (b[j])
    static constexpr double b[7] = {
        35.0/384.0, 0.0, 500.0/1113.0,
        125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0
    };

    // Dense‐output correction coefficients (d[j])
    static constexpr double d[7] = {
        -12715105075.0 / 11282082432.0,
        0.0,
        87487479700.0 / 32700410799.0,
        -10690763975.0 / 1880347072.0,
        701980252875.0 / 199316789632.0,
        -1453857185.0 / 822651844.0,
        69997945.0 / 29380423.0
    };

    // ─── Compute y_dense = y + h * Σ_{j=0..6} [b[j] + θ·d[j]] * k[j][i] ───
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = 0; j < 7; ++j) {
            // b[j]: 5th‐order weight
            // d[j]: dense‐output correction
            acc += (b[j] + theta * d[j]) * k[j][i];
        }
        dense_out[i] = y[i] + h * acc;
    }
}
