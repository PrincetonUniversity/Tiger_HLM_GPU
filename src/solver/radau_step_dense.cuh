// src/solver/radau_step_dense.cuh

#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include "small_lu.cuh"    // for small_matrix_LU_solve

static constexpr double SQRT6 = 2.449489742783178;

/**
 * If Model::jacobian isn't available, approximate ∂f/∂y by finite differences.
 */
template<class Model>
__device__ void approx_jacobian(double t, const double* y, double J[Model::N_EQ][Model::N_EQ], int sys_id, const typename Model::SP_TYPE* d_sp) {
    constexpr int N = Model::N_EQ;
    double f0[N], f1[N];
    Model::rhs(t, y, f0, N, sys_id, d_sp);
    const double eps = sqrt(1e-16);
    for (int j = 0; j < N; ++j) {
        double yj = y[j];
        double h_eps = eps * fmax(1.0, fabs(yj));
        double ytmp[N];
        for (int i = 0; i < N; ++i) ytmp[i] = y[i];
        ytmp[j] += h_eps;
        Model::rhs(t, ytmp, f1, N, sys_id, d_sp);
        for (int i = 0; i < N; ++i) {
            J[i][j] = (f1[i] - f0[i]) / h_eps;
        }
    }
}

/**
 * Single‐step 3‐stage Radau IIA integrator (implicit) for a Model.
 *
 * Solves for the 3 implicit stage increments Z[s][i], builds y_out,
 * and returns the ∞‐norm error estimate in *error_norm.
 */
template <class Model>
__device__ void radau_step(
    double t,
    const double* y,
    double*       y_out,
    int           n,           // == Model::N_EQ
    double        h,           // timestep
    double        rtol,
    double        atol,
    double*       error_norm,
    double        /*k_unused*/[7][Model::N_EQ],  // dummy
    int           sys_id,
    const typename Model::SP_TYPE* d_sp
) {
    constexpr int N = Model::N_EQ;

    // Radau IIA 3‐stage coefficients (order 5)
    static constexpr double c[3] = {
        (4.0 - SQRT6)/10.0,
        (4.0 + SQRT6)/10.0,
        1.0
    };
    static constexpr double A[3][3] = {
        { (88.0-7.0*SQRT6)/360.0, (296.0-169.0*SQRT6)/1800.0, (-2.0+3.0*SQRT6)/225.0 },
        { (296.0+169.0*SQRT6)/1800.0, (88.0+7.0*SQRT6)/360.0,  (-2.0-3.0*SQRT6)/225.0 },
        { (16.0-SQRT6)/36.0,        (16.0+SQRT6)/36.0,         1.0/9.0 }
    };
    static constexpr double b[3] = {
        (16.0-SQRT6)/36.0,
        (16.0+SQRT6)/36.0,
        1.0/9.0
    };
    static constexpr double b_alt[3] = {
        (226.0-60.0*SQRT6)/720.0,
        (226.0+60.0*SQRT6)/720.0,
        1.0/12.0
    };

    // 1) Allocate and initialize stage increments Z[s][i]
    double Z[3][N];
    double f0[N];
    Model::rhs(t, y, f0, N, sys_id, d_sp);
    for (int s = 0; s < 3; ++s)
        for (int i = 0; i < N; ++i)
            Z[s][i] = f0[i];

    // 2) Simplified Newton iteration to solve for Z
    const int max_iter = 10;
    double J[N][N], Mmat[3*N][3*N], rhs[3*N];
    for (int iter = 0; iter < max_iter; ++iter) {
        // Build the big system and RHS
        for (int s = 0; s < 3; ++s) {
            // Compute Yi = y + h Σ_j A[s][j] Z[j]
            double Yi[N];
            for (int i = 0; i < N; ++i) {
                double acc = y[i];
                for (int j = 0; j < 3; ++j) acc += h * A[s][j] * Z[j][i];
                Yi[i] = acc;
            }
            // Evaluate Jacobian ∂f/∂y at (t + c[s]h, Yi)
            #ifdef HAS_MODEL_JACOBIAN
                Model::jacobian(t + c[s]*h, Yi, J);
            #else
                approx_jacobian<Model>(t + c[s]*h, Yi, J, sys_id, d_sp);
            #endif

            // Fill block row s of Mmat and rhs
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    // diagonal block
                    Mmat[s*N + i][s*N + j] = (i==j ? 1.0 : 0.0)
                                              - h * A[s][s] * J[i][j];
                }
            }
            for (int sp = 0; sp < 3; ++sp) if (sp != s) {
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        Mmat[s*N + i][sp*N + j] = - h * A[s][sp] * J[i][j];
                    }
                }
            }
            double fs[N];
            Model::rhs(t + c[s]*h, Yi, fs, N, sys_id, d_sp);
            for (int i = 0; i < N; ++i) {
                rhs[s*N + i] = -Z[s][i] + fs[i];
            }
        }

        // Solve Mmat · delta = rhs
        small_matrix_LU_solve(3*N, &Mmat[0][0], rhs);

        // Update Z and test convergence
        double maxd = 0.0;
        for (int s = 0; s < 3; ++s) {
            for (int i = 0; i < N; ++i) {
                Z[s][i] += rhs[s*N + i];
                maxd = fmax(maxd, fabs(rhs[s*N + i]));
            }
        }
        if (maxd < 1e-8) break;
    }

    // 3) Build y_out = y + h Σ b[s] Z[s]
    for (int i = 0; i < N; ++i) {
        double acc = y[i];
        for (int s = 0; s < 3; ++s) {
            acc += h * b[s] * Z[s][i];
        }
        y_out[i] = acc;
    }

    // 4) Error estimate with embedded b_alt
    double max_ratio = 0.0;
    for (int i = 0; i < N; ++i) {
        double err_i = 0.0;
        for (int s = 0; s < 3; ++s) {
            err_i += h * (b[s] - b_alt[s]) * Z[s][i];
        }
        double tol = atol + rtol * fmax(fabs(y[i]), fabs(y_out[i]));
        max_ratio = fmax(max_ratio, fabs(err_i / tol));
    }
    *error_norm = max_ratio;
}

/**
 * Dense‐output for Radau IIA (order 5) over a single step.
 *
 *   y_dense(t_n + θ·h) = y_n
 *     + h·[ Σ_{s=0..2} b[s]·Z[s]·θ
 *           + Σ_{m=1..4} ( Σ_{s=0..2} P[s][m−1]·Z[s] )·θ^m ]
 */
template <class Model>
__device__ void radau_dense(
    const double* y_n,                // state at t_n
    double        Z[3][Model::N_EQ],  // stage increments from last step
    int           n,                  // == Model::N_EQ
    double        h,                  // step size used
    double        theta,              // fraction in [0,1]
    double*       y_dense             // output array, length n
) {
    constexpr int N = Model::N_EQ;
    static constexpr double P[3][4] = {
        { 1.0/6.0, -1.0/15.0,  1.0/20.0, -1.0/60.0 },
        { 2.0/3.0, -4.0/15.0,  1.0/ 5.0, -1.0/20.0 },
        { 1.0/6.0, -1.0/15.0,  1.0/20.0, -1.0/60.0 }
    };

    // Compute Q[m][i] = Σ_s P[s][m] * Z[s][i]
    double Q[4][N] = {{0}};
    for (int m = 0; m < 4; ++m) {
        for (int s = 0; s < 3; ++s) {
            for (int i = 0; i < N; ++i) {
                Q[m][i] += P[s][m] * Z[s][i];
            }
        }
    }

    // Evaluate the quartic polynomial in θ
    for (int i = 0; i < N; ++i) {
        double poly = 0.0, th = theta;
        for (int m = 0; m < 4; ++m) {
            poly += Q[m][i] * th;
            th   *= theta;
        }
        // Q[0] already includes Σ b[s]·Z[s], so:
        y_dense[i] = y_n[i] + h * ( Q[0][i]*theta + poly );
    }
}
