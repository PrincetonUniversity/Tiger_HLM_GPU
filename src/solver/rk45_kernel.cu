#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"
#include "rk45_step_dense.cuh"
#include "models/model_dummy.hpp"  // For DummyModel::N_EQ and devParams

// -----------------------------------------------------------------------------
// Implementation of rk45_kernel_multi defined in rk45.h
// -----------------------------------------------------------------------------
template <class Model>
__global__ void rk45_kernel_multi(
    double* y0_all,        // [num_systems × Model::N_EQ]: initial states
    double* y_final_all,   // [num_systems × Model::N_EQ]: final states
    double* query_times,   // [num_queries]: sorted dense-output times
    double* dense_all,     // [num_systems × Model::N_EQ × num_queries]: dense outputs
    int     num_systems,   // total number of systems
    int     num_queries,   // total number of query times
    double  t0,            // start time
    double  tf             // end time
) {
    constexpr int N_EQ = Model::N_EQ;

    // Identify which system this thread integrates
    int sys_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys_id >= num_systems) return;

    // ─── DEBUG CHECK: only print once from sys_id == 0 ───
    // if (sys_id == 0) {
    //     printf("KERNEL: num_systems = %d, num_queries = %d\n", 
    //            num_systems, num_queries);
    //     if (num_queries > 0) {
    //         // Print the first and last query_time to confirm we copied the vector correctly:
    //         printf("KERNEL: query_times[0] = %f, query_times[%d] = %f\n",
    //                query_times[0], num_queries-1, query_times[num_queries-1]);
    //     }
    // }
    // // ────────────────────────────────────────────────────────

    // Load solver tolerances and initial step from constant memory
    double my_rtol = devParams.rtol; // default is 1e-6
    double my_atol = devParams.atol; // default is 1e-9
    double h = devParams.initialStep; // default is 0.01

    // ─── Integration state ───
    double t = t0;
    
    // Allocate registers for the current state, next state, and stage slopes
    double y[N_EQ], y_next[N_EQ];
    double k[7][N_EQ];
    double error_norm = 0.0;

    // Load initial condition y0 for this system
    for (int i = 0; i < N_EQ; ++i) {
        y[i] = y0_all[sys_id * N_EQ + i];
    }

    // ─── Trial step to compute SciPy-style first step size ───
    {
        // Compute initial slope k_trial[0] = f(t0, y0)
        double y_trial[N_EQ];
        double k_trial[7][N_EQ];
        Model::rhs(t, y, k_trial[0], N_EQ);

        // Take one RK45 step using the current h to get y_trial and trial_error_norm
        double trial_error_norm = 0.0;
        rk45_step<Model>(t, y, y_trial, N_EQ, h, my_rtol, my_atol, &trial_error_norm, k_trial);

        // Compute the infinity-norm of the scaled error exactly as SciPy:
        //     y_err_i = h * Σ_{s=0..6} (b[s] - b_alt[s]) * k_trial[s][i]
        //     tol_i   = my_atol + my_rtol * max( |y[i]|, |y_trial[i]| )
        //     ratio_i = |y_err_i| / tol_i, then max over i
        double max_ratio = 0.0;
        for (int i = 0; i < N_EQ; ++i) {
            // Assemble the 5th–4th order local error for component i
            double y_err_i = 0.0;
            for (int s = 0; s < 7; ++s) {
                // Coefficients must exactly match those in rk45_step
                static constexpr double b[7] = {
                    35.0/384.0,      0.0,
                    500.0/1113.0, 125.0/192.0,
                    -2187.0/6784.0, 11.0/84.0,
                    0.0
                };
                static constexpr double b_alt[7] = {
                    5179.0/57600.0, 0.0,
                    7571.0/16695.0, 393.0/640.0,
                    -92097.0/339200.0, 187.0/2100.0, 1.0/40.0
                };
                y_err_i += h * ((b[s] - b_alt[s]) * k_trial[s][i]);
            }
            double ymax   = fmax(fabs(y[i]), fabs(y_trial[i]));
            double tol_i  = my_atol + my_rtol * ymax;
            double ratio  = fabs(y_err_i / tol_i);
            if (ratio > max_ratio) {
                max_ratio = ratio;
            }
        }

        // Rescale h exactly as SciPy: h₀ = safety * h * (max_ratio)^(-1/5)
        // then bound h₀ to lie within [minScale * h, maxScale * h]
        double factor = devParams.safety * pow(max_ratio + 1e-16, -1.0/5.0);
        h = h * fmin(fmax(factor, devParams.minScale), devParams.maxScale);

        // Discard trial results and reset t, y to (t0, y0)
        t = t0;
        for (int i = 0; i < N_EQ; ++i) {
            y[i] = y0_all[sys_id * N_EQ + i];
        }
    }
    

    // Index of the next query time to process
    int next_q_idx = 0;

    // ─── Main adaptive RK45 loop ───
    while (t < tf) {
        // If the next step would overshoot tf, shorten it
        if (t + h > tf) h = tf - t;

        // Compute the first slope k[0] = f(t,y) explicitly 
        Model::rhs(t, y, k[0], N_EQ);

        // Take exactly one RK45 step (fills k[0..6], y_next, error_norm)
        rk45_step<Model>(t, y, y_next, N_EQ, h, my_rtol, my_atol, &error_norm, k); 

        if (error_norm <= 1.0) {
            // Accept step
            double t_next = t + h;

            // Dense output for queries in (t, t_next]
            while (next_q_idx < num_queries && query_times[next_q_idx] <= t_next) {
            double tq = query_times[next_q_idx];
            if (tq > t) {
                // Compute normalized fraction θ = (tq - t)/h
                double theta = (tq - t)/h;
                double y_dense[N_EQ];

                // Evaluate the 5th-order interpolant at θ
                rk45_dense<Model>(y, k, N_EQ, h, theta, y_dense);

                // Store each component y_dense[comp] into dense_all
                for (int comp = 0; comp < N_EQ; ++comp) {
                    int flat_index = 
                        sys_id * (N_EQ * num_queries)   // offset to system sys_id
                    + comp   * (num_queries)            // offset to component comp
                    + next_q_idx;                       // offset to the qth query index
                    dense_all[flat_index] = y_dense[comp];
                }
            }
            next_q_idx++;
            }

             // Advance state: y ← y_next, t ← t_next
            for (int i = 0; i < N_EQ; ++i) {
                y[i] = y_next[i];
            }
            t = t_next;

            // Rescale h: h ← h * clamp( safety * (1 / error_norm)^(1/5) , minScale, maxScale )
            double factor = devParams.safety * pow(1.0 / (error_norm + 1e-16), 0.2);
            h *= fmin(fmax(factor, devParams.minScale), devParams.maxScale);
        } else {
            // Reject the step
            // shrink h by at most a factor of 1.0, then clamp, and retry ───────
            double factor = devParams.safety * pow(1.0 / (error_norm + 1e-16), 0.2);
            factor = fmin(factor, 1.0);
            factor = fmin(fmax(factor, devParams.minScale), devParams.maxScale);
            h *= factor;

            // Do not advance t or y in else; the loop will repeat with the smaller h
        }
    }

    // Write final state y(t_f) into y_final_all
    for (int i = 0; i < N_EQ; ++i) {
        y_final_all[sys_id * N_EQ + i] = y[i];
    }
}

// ----------------------------------------------------------------------------
// Explicit instantiation of rk45_kernel_multi for dummy model
// ----------------------------------------------------------------------------
template __global__ void rk45_kernel_multi<DummyModel>(
    double*, double*, double*, double*, int, int, double, double
);
