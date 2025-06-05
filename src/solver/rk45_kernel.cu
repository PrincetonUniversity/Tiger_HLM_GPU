#include <cstdio>
#include <cuda_runtime.h>
#include "rk45.h"
#include "rk45_step_dense.cuh"
#include "models/model_dummy.hpp"  // For DummyModel::N_EQ and devParams

// ─────────────────────────────────────────────────────────────────────────────
//  Templated GPU kernel: each thread integrates one N_EQ‐dimensional system.
//  Reads from the device‐global `devParams`.
// ─────────────────────────────────────────────────────────────────────────────
template <class Model>
__global__ void rk45_kernel_multi(
    double* y0_all,        // [num_systems × Model::N_EQ]
    double* y_final_all,   // [num_systems × Model::N_EQ]
    double* query_times,   // [num_queries]
    double* dense_all,     // [num_systems × Model::N_EQ × num_queries]
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf
) {
    constexpr int N_EQ = Model::N_EQ;

    int sys_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys_id >= num_systems) return;

    // ─── Integration state ───
    double t = t0;
    double h = devParams.initialStep;   // load from device global memory

    double y[N_EQ], y_next[N_EQ];
    double k[7][N_EQ];
    double error_norm = 0.0;

    // Load initial state
    for (int i = 0; i < N_EQ; ++i) {
        y[i] = y0_all[sys_id * N_EQ + i];
    }

    int next_q_idx = 0;
    while (t < tf) {
        if (t + h > tf) h = tf - t;

        // ── Stage 1: take exactly one RK45 step (fills k[0..6], y_next, error_norm)
        rk45_step<Model>(t, y, y_next, N_EQ, h, &error_norm, k); // step‐error for acceptance, slopes for dense

        if (error_norm <= 1.0) {
            // ── Accept step
            double t_next = t + h;

            // ── Dense output for queries in (t, t_next]
            while (next_q_idx < num_queries && query_times[next_q_idx] <= t_next+ 1e-14) {
                double tq = query_times[next_q_idx];
                if (tq > t) {
                    double theta = (tq - t) / h;
                    // 0 < theta ≤ 1  ensures dense‐output returns the correct point
                    double y_dense[N_EQ];
                    rk45_dense<Model>(y, k, N_EQ, h, theta, y_dense); // output: y(t + θ·h)       

                    for (int i = 0; i < N_EQ; ++i) {
                        int idx = sys_id * (N_EQ * num_queries)
                                  + i * num_queries
                                  + next_q_idx;
                        dense_all[idx] = y_dense[i];
                    }
                }
                next_q_idx++;
            }

            // Update state and time
            for (int i = 0; i < N_EQ; ++i) {
                y[i] = y_next[i];
            }
            t = t_next;

            // ── Adjust step size
            double factor = devParams.safety * pow(1.0 / (error_norm + 1e-16), 0.2);
            h *= fmin(fmax(factor, devParams.minScale), devParams.maxScale);
        } else {
            // ── Reject step: reduce step size and retry
            double factor = devParams.safety * pow(1.0 / (error_norm + 1e-16), 0.2);
            h *= fmin(fmax(factor, devParams.minScale), devParams.maxScale);
        }
    }

    // ─── Write final state
    for (int i = 0; i < N_EQ; ++i) {
        y_final_all[sys_id * N_EQ + i] = y[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Instantiate for DummyModel
// ─────────────────────────────────────────────────────────────────────────────
template __global__ void rk45_kernel_multi<DummyModel>(
    double*, double*, double*, double*, int, int, double, double
);
