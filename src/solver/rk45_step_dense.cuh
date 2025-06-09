//src/solver/rk45_step_dense.cuh
#pragma once

#include <math.h>
#include <cuda_runtime.h>

/**
 * Single‐step Dormand–Prince RK45 integrator (no dense‐output) for a Model.
 * 
 * This device function takes the current time t and state y[0..N_EQ−1], computes
 * the six additional stage slopes k[1..6], forms the 5th‐order update y_out, and
 * returns the ∞‐norm error estimate in error_norm.  It does not perform any
 * interpolation between tₙ and tₙ₊₁.
 *
 * Template parameters:
 *   Model::N_EQ      → number of equations (dimension of y and y_out)
 *   Model::rhs       → device RHS function, used internally to compute slopes
 *   Model::devParams → Parameters struct (rtol, atol, initial step, etc.) in constant memory
 *
 * Arguments:
 *   t           : current time tₙ
 *   y           : pointer to the current state array of length N_EQ at time tₙ
 *   y_out       : pointer to an array of length N_EQ, to be filled with the 5th‐order solution at tₙ + h
 *   n           : integer that must equal Model::N_EQ
 *   h           : current step size (timestep) to advance from tₙ to tₙ₊₁ = tₙ + h
 *   rtol        : relative tolerance (scalar) for error control
 *   atol        : absolute tolerance (scalar) for error control
 *   error_norm  : address of a double to receive the ∞‐norm of the local error estimate
 *   k           : 2D array k[0..6][0..N_EQ−1], where:
 *                   • k[0] is already filled by the caller via Model::rhs(t, y, k[0], N_EQ, sys_id, d_sp)
 *                   • this function will compute k[1]..k[6] internally
 */
template <class Model>
__device__ void rk45_step(
    double t,                            // current time
    const double* y,                     // current state [n]
    double* y_out,                       // next state [n]
    int n,                               // must equal Model::N_EQ
    double h,                            // timestep
    double rtol,                         // relative tolerance (scalar)
    double atol,                         // absolute tolerance (scalar)
    double* error_norm,                  // ∞-norm error estimate
    double k[7][Model::N_EQ],             // stage slopes (output)
    int sys_id                           // passing stream ID here
) {
    constexpr int N_EQ = Model::N_EQ;

    // Coefficients for the Butcher tableau (c vector):
    // c[i] is the fractional stage time; in particular, c[6] = 1.0 denotes
    // that the 7th slope k[6] is evaluated at t + h (the end of the step)
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
        // The last row here lists exactly the 5th‐order solution weights b[j],
        // because at stage 6 (s = 6) c[6] = 1.0 corresponds to evaluating at t + h
        {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0}
    };

    // Coefficients for the 5th‐order solution (b)
    // Used to form y_out = y + h * Σ b[j]*k[j]
    static constexpr double b[7] = {
        35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0,
        -2187.0/6784.0, 11.0/84.0, 0.0
    };

    // Coefficients for the 4th‐order embedded solution (b_alt)
    // Used to compute the local error estimate: y_err = h * Σ (b[j] - b_alt[j]) * k[j]
    static constexpr double b_alt[7] = {
        5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
        -92097.0/339200.0, 187.0/2100.0, 1.0/40.0
    };
    
    // Temporary array for building y_temp = y + h*Σ a[s][j]*k[j]
    double y_temp[N_EQ];

    // // ─── Stage 1: Compute k[0] ───
    // In rk45_kernel_multi, the caller does:
    //     Model::rhs(t, y, k[0], N_EQ);
    // before invoking rk45_step, so that k[0] is already available here

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
        // Evaluate the RHS at t + c[s]*h, storing the slope into k[s]
        Model::rhs(t + c[s] * h, y_temp, k[s], N_EQ, sys_id, d_sp);
    }

    // ─── Stage 3: Build y_out (5th‐order) ───
    // y_out[i] = y[i] + h * Σ_{j=0..6} b[j] * k[j][i]
    for (int i = 0; i < N_EQ; ++i) {
        double acc = y[i];
        for (int s = 0; s < 7; ++s) {
            acc += h * b[s] * k[s][i];
        }
        y_out[i] = acc;
    }

    // ─── Stage 4: Compute the ∞‐norm error estimate ───
    // error_norm = max_i |y_err_i / tol_i|, where
    // y_err_i = h * Σ (b[j] - b_alt[j]) * k[j][i]
    // tol_i = atol + rtol * max(|y[i]|, |y_out[i]|)
    double max_ratio = 0.0;
    for (int i = 0; i < N_EQ; ++i) {
        // (a) assemble the 5th–4th order local error for component i
        double y_err_i = 0.0;
        for (int s = 0; s < 7; ++s) {
            y_err_i += h * (b[s] - b_alt[s]) * k[s][i];
        }
        // (b) compute tol_i = atol + rtol * max(|y_i|, |y_out_i|)
        double ymax  = fmax(fabs(y[i]), fabs(y_out[i]));
        double tol_i = atol + rtol * ymax;

        // (c) absolute‐value ratio for this component
        double ratio = fabs(y_err_i / tol_i);

        // (d) track the maximum over all i
        if (ratio > max_ratio) {
            max_ratio = ratio;
        }
    }
    *error_norm = max_ratio;   


}

/**
 * Dense‐output interpolator for a single Dormand–Prince RK45 step.
 *
 * After an RK45 step has produced the stage slopes k[0..6] and advanced
 * the solution from tₙ to tₙ₊₁ = tₙ + h, this routine builds the 5th‐order
 * interpolant
 *
 *   y_dense(tₙ + θ·h) = yₙ + h·[ (∑_{j=0..6} b[j]·k[j])·θ  +  ∑_{m=1..4} (∑_{j=0..6} Pmat[j][m–1]·k[j])·θ^m ],
 *
 * where θ ∈ [0,1].  The first term (using weights b[j]) is the base 5th‐order
 * solution, and the four additional powers of θ use the P‐matrix coefficients
 * below to produce a continuous interpolant on [tₙ, tₙ₊₁].
 *
 * Template parameters:
 *   Model::N_EQ → number of equations (dimension of y_n and y_dense)
 *
 * Arguments:
 *   y_n     : state vector at tₙ (length = N_EQ)
 *   k       : stage slopes from the last accepted RK45 step, k[j][i] = slope j for component i
 *   n       : must equal Model::N_EQ
 *   h       : step size h = tₙ₊₁ – tₙ
 *   theta   : normalized interpolation parameter in [0,1], so t = tₙ + θ·h
 *   y_dense : output array to be filled with the interpolated state at tₙ + θ·h
 */
template <class Model>
__device__ void rk45_dense(
    const double* y_n,                // State at time t_n, size = N_EQ
    double        k[7][Model::N_EQ],  // Slopes from the last accepted step
    int           n,                  // Must equal Model::N_EQ
    double        h,                  // Step size used in that step
    double        theta,              // Normalized time in (0,1], theta = (t - t_n)/h
    double*       y_dense             // Output: y(t_n + theta·h), size = N_EQ
) {
    constexpr int N_EQ = Model::N_EQ;

    
    //  Dormand–Prince RK45 dense‐output coefficients (P‐matrix)
    //  For each stage j = 0..6 and each power m = 1..4, Pmat[j][m–1] is the
    //  coefficient that multiplies k[j]·θ^m in the quartic portion of the
    //  interpolant.  In practice, one computes
    //      Q[m–1][i] = ∑_{j=0..6} Pmat[j][m–1] * k[j][i],    for m = 1..4,
    //  and then forms the sum Q[0][i]*θ + Q[1][i]*θ² + Q[2][i]*θ³ + Q[3][i]*θ⁴
    //  as the extra correction beyond the base 5th‐order solution
     
    static constexpr double Pmat[7][4] = {
        {  1.0,
           -8048581381.0/2820520608.0,
            8663915743.0/2820520608.0,
           -12715105075.0/11282082432.0 },
        {  0.0, 0.0, 0.0, 0.0 },
        {  0.0,
           131558114200.0/32700410799.0,
           -68118460800.0/10900136933.0,
            87487479700.0/32700410799.0 },
        {  0.0,
           -1754552775.0/470086768.0,
            14199869525.0/1410260304.0,
           -10690763975.0/1880347072.0 },
        {  0.0,
            127303824393.0/49829197408.0,
           -318862633887.0/49829197408.0,
            701980252875.0/199316789632.0 },
        {  0.0,
           -282668133.0/205662961.0,
            2019193451.0/616988883.0,
           -1453857185.0/822651844.0 },
        {  0.0,
             40617522.0/29380423.0,
           -110615467.0/29380423.0,
             69997945.0/29380423.0 }
    };

    // Precompute Q[m][i] = sum_j Pmat[j][m] * k[j][i], for m=0..3, i=0..N_EQ-1
    double Q[4][N_EQ];
    for (int m = 0; m < 4; ++m) {
        for (int i = 0; i < N_EQ; ++i) {
            double sum = 0.0;
            for (int j = 0; j < 7; ++j) {
                sum += Pmat[j][m] * k[j][i];
            }
            Q[m][i] = sum;
        }
    }

    // Evaluate the quartic interpolation polynomial at theta:
    // y_dense[i] = y_n[i] + h * [ Q[0][i]*θ + Q[1][i]*θ² + Q[2][i]*θ³ + Q[3][i]*θ⁴ ]
    for (int i = 0; i < N_EQ; ++i) {
        double poly = 0.0;
        double thp = theta; // θ^1
        for (int m = 0; m < 4; ++m) {
            poly += Q[m][i] * thp;
            thp *= theta;   // θ^(m+2)
        }
        y_dense[i] = y_n[i] + h * poly;
    }
}



