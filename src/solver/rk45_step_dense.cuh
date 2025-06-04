// src/solver/rk45_step_dense.cuh
#pragma once

#include <math.h>
#include <cuda_runtime.h>

/**
 * Single‐step Dormand–Prince RK45 (with dense output) for a Model.  
 * This function is templated on Model so it knows Model::N_EQ at compile time.
 *
 * The caller (rk45_kernel_multi) must fill k[0] by calling Model::rhs(t, y, k[0], N_EQ),
 * then call this to compute k[1..6], y_out, error_norm, and optionally dense_out.
 *
 * Template parameters:
 *   Model::N_EQ  → number of equations
 *   Model::rhs   → the device RHS function
 *   Model::devParams → the Parameters struct in constant memory
 */
template <class Model>
__device__ void rk45_step_dense(
    double t,                            // current time
    const double* y,                     // current state [n]
    double* y_out,                       // next state [n]
    int n,                               // must equal Model::N_EQ
    double h,                            // timestep
    double* error_norm,                  // L2 error estimate
    double k[7][Model::N_EQ],            // stage slopes 
    double* dense_out = nullptr,         // if non‐null, output dense interpolation at θ
    double theta = 0.5                   // normalized interpolation point
) {
    constexpr int N_EQ = Model::N_EQ;

    // Need to work on the following part
    // Compute the step size adjustment factor
    // The factor is calculated using the error estimate (*error_norm) from the previous step.
    // A safety factor (Model::devParams.safety) is applied to ensure the step size is conservative
    // and avoids overshooting the desired accuracy. The power of 0.2 corresponds to the 
    // inverse of the order of the RK45 method (5th order), which determines how the step size
    // should scale with the error.
    // Note: The addition of 1e-16 prevents division by zero in case *error_norm is extremely small.
    // double factor = Model::devParams.safety * pow(1.0 / (*error_norm + 1e-16), 0.2);

    // Step size adjustment logic
    // The step size (h) is adjusted by multiplying it with the computed factor. To ensure
    // stability, the factor is clamped between a minimum scale (Model::devParams.minScale)
    // and a maximum scale (Model::devParams.maxScale). This prevents the step size from
    // shrinking too much (causing excessive computation) or growing too large (causing
    // instability or loss of accuracy).
    // Need to check this with the existing asynch explicit.c solver
    // h *= fmin(fmax(factor, Model::devParams.minScale), Model::devParams.maxScale);

    // Debug print for step size, limited to the first thread of the first block
    // This debug statement prints the current time (t) and the adjusted step size (h) to the
    // console. It is limited to the first thread of the first block to avoid excessive output
    // from multiple threads. This is useful for debugging and understanding how the step size
    // evolves during the integration process.if (threadIdx.x == 0 && blockIdx.x == 0) 
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("Time t: %f, Step size h: %f\n", t, h);
    // }
    
    // Dormand–Prince coefficients
    static constexpr double c[7] = {
        0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
    };
    static constexpr double a[7][6] = {
        {},
        {1.0/5.0},
        {3.0/40.0, 9.0/40.0},
        {44.0/45.0, -56.0/15.0, 32.0/9.0},
        {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0},
        {9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0},
        {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0}
    };
    static constexpr double b[7] = {
        35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0,
        -2187.0/6784.0, 11.0/84.0, 0.0
    };
    static constexpr double b_alt[7] = {
        5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
        -92097.0/339200.0, 187.0/2100.0, 1.0/40.0
    };
    static constexpr double d[7] = {
        -12715105075.0 / 11282082432.0,
        0.0,
        87487479700.0 / 32700410799.0,
        -10690763975.0 / 1880347072.0,
        701980252875.0 / 199316789632.0,
        -1453857185.0 / 822651844.0,
        69997945.0 / 29380423.0
    };

    double y_temp[N_EQ];

    // ─── Stage 2: compute k[1..6] ───
    // We assume k[0] was filled by Model::rhs(t, y, k[0], N_EQ) in the caller.
    for (int s = 1; s < 7; ++s) {
        for (int i = 0; i < N_EQ; ++i) {
            y_temp[i] = y[i];
            for (int j = 0; j < s; ++j) {
                y_temp[i] += h * a[s][j] * k[j][i];
            }
        }
        // Call the model’s RHS at t + c[s]*h:
        Model::rhs(t + c[s] * h, y_temp, k[s], N_EQ);
    }

    // ─── Stage 3: build y_out ───
    for (int i = 0; i < N_EQ; ++i) {
        y_out[i] = y[i];
        for (int s = 0; s < 7; ++s) {
            y_out[i] += h * b[s] * k[s][i];
        }
    }

    // ─── Stage 4: compute error_norm (L2 norm) ───
    double err_sum = 0.0;
    for (int i = 0; i < N_EQ; ++i) {
        double y_err = 0.0;
        for (int s = 0; s < 7; ++s) {
            y_err += h * (b[s] - b_alt[s]) * k[s][i];
        }
        double tol = 1e-9 + 1e-6 * fmax(fabs(y[i]), fabs(y_out[i]));
        err_sum += (y_err / tol) * (y_err / tol);
    }
    *error_norm = sqrt(err_sum / N_EQ);

    // ─── Stage 5: dense output if requested ───
    if (dense_out != nullptr) {
        for (int i = 0; i < N_EQ; ++i) {
            dense_out[i] = y[i];
            for (int s = 0; s < 7; ++s) {
                dense_out[i] += h * d[s] * k[s][i] * theta;
            }
        }
    }
}
