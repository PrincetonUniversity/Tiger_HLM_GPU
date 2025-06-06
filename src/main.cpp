#include <cstdio>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "solver/rk45_api.hpp"   // run_rk45<Model>() and setModelParameters<Model>()
#include "models/model_dummy.hpp" // DummyModel and Parameters
#include "model_registry.hpp"

// Forward‐declared kernel hosted in model_dummy_global.cu:
extern __global__ void checkDevParamsKernel();

int main() {
    using namespace rk45_api;

    // ───────── Define problem size ─────────
    // Number of independent ODE systems to solve in parallel on the GPU.
    constexpr int num_systems = 4;

    // Number of dense‐output query points to request from the GPU solver.
    constexpr int num_queries = 10000;

    // Initial and final times for the integration interval.
    const double t0 = 0.0, tf = 5.0;

    // ───────── Build host initial states ─────────
    // Allocate a flat vector of length (num_systems × N_EQ)
    // Each system’s initial condition is the constant vector [1,1,…,1], 
    // following the python example
    std::vector<double> h_y0(num_systems * DummyModel::N_EQ);
    for (int s = 0; s < num_systems; ++s) {
        for (int i = 0; i < DummyModel::N_EQ; ++i) {
            h_y0[s * DummyModel::N_EQ + i] = 1.0;
        }
    }

    // ───────── Build & sort query times ─────────
    // Fill h_query_times with evenly spaced points in (t0, tf), then sort
    std::vector<double> h_query_times(num_queries);
    for (int q = 0; q < num_queries; ++q) {
        h_query_times[q] = tf * (q + 1.0) / (num_queries + 1.0);
    }
    std::sort(h_query_times.begin(), h_query_times.end());

    // ───────── Compute SciPy’s “auto‐scaled” first step on the CPU ─────────
    {
        // Build y₀ and f₀
        int N_EQ = DummyModel::N_EQ;
        std::vector<double> y0_cpu(N_EQ, 1.0);
        std::vector<double> f0_cpu(N_EQ);
        DummyModel::rhs(t0, y0_cpu.data(), f0_cpu.data(), N_EQ);

        // Choose the same rtol/atol as SciPy’s RK45 solver
        double rtol = 1e-6;
        double atol = 1e-9;

        // Compute the per‐component "tol_i" array
        std::vector<double> scale(N_EQ);
        for (int i = 0; i < N_EQ; ++i) {
            // tol_i = atol + rtol * |y₀[i]|
            scale[i] = atol + rtol * std::fabs(y0_cpu[i]);
        }

        // Form the 2‐norms of y₀/scale and f₀/scale
        double d0 = 0.0, d1 = 0.0;
        for (int i = 0; i < N_EQ; ++i) {
            double yi = y0_cpu[i] / scale[i];
            double fi = f0_cpu[i] / scale[i];
            d0 += yi * yi;
            d1 += fi * fi;
        }
        d0 = std::sqrt(d0);
        d1 = std::sqrt(d1);

        // SciPy’s initial‐step estimate
        //     h_guess = 0.01 * ‖y₀/scale‖₂ / (‖f₀/scale‖₂ + ε)
        // Note: this formula selects a reasonable starting step
        // For HLM models we will have to set initial conditions differently
        double h_guess = 0.01 * d0 / (d1 + 1e-16);

        // Copy into device‐Parameters exactly
        DummyModel::Parameters hp;
        hp.initialStep = h_guess;  // ← matches SciPy’s first‐step guess, 
        hp.rtol        = rtol;
        hp.atol        = atol;
        hp.safety      = 0.9;
        hp.minScale    = 0.2;
        hp.maxScale    = 10.0;
        setModelParameters<DummyModel>(hp);

        // Uncomment to confirm that the device parameters match the host parameters
        // checkDevParamsKernel<<<1,1>>>();
        // cudaDeviceSynchronize();

        // Uncomment to see the initial step size computed by the CPU
        // printf("HOST:  SciPy‐style first step h0 = %.12f\n", h_guess);
    }

    // ───────── Run the solver on GPU ─────────
    // Returns a std::pair: (final states, dense output)
    auto result = run_rk45<DummyModel>(h_y0, t0, tf, h_query_times);
    const auto& h_y_final = result.first;
    const auto& h_dense   = result.second;

    // ───────── Write final.csv ─────────
    {
        std::ofstream final_file("final.csv");
        final_file << "Var0";
        for (int i = 1; i < DummyModel::N_EQ; ++i) {
            final_file << ",Var" << i;
        }
        final_file << "\n";

        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < DummyModel::N_EQ; ++i) {
                final_file << h_y_final[s * DummyModel::N_EQ + i];
                if (i < DummyModel::N_EQ - 1) final_file << ",";
            }
            final_file << "\n";
        }
    }

    // ───────── Write dense.csv ─────────
    {
        std::ofstream dense_file("dense.csv");
        dense_file << "time";
        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < DummyModel::N_EQ; ++i) {
                dense_file << ",Var" << i << "_sys" << s;
            }
        }
        dense_file << "\n";

        for (int q = 0; q < num_queries; ++q) {
            dense_file << h_query_times[q];
            for (int s = 0; s < num_systems; ++s) {
                for (int i = 0; i < DummyModel::N_EQ; ++i) {
                    int idx = s * (DummyModel::N_EQ * num_queries)
                              + i * num_queries
                              + q;
                    dense_file << "," << h_dense[idx];
                }
            }
            dense_file << "\n";
        }
    }

    // ───────── Print a quick summary ─────────
    printf("Final states at t = %.1f:\n", tf);
    for (int s = 0; s < num_systems; ++s) {
        printf(" System %d:", s);
        for (int i = 0; i < DummyModel::N_EQ; ++i) {
            printf(" H%d=%.6f", i, h_y_final[s * DummyModel::N_EQ + i]);
        }
        printf("\n");
    }

    // ───────── Second GPU run: request outputs exactly at t = 1, 2, 3, 4 ─────────
    std::vector<double> query_exact = {1.0, 2.0, 3.0, 4.0};
    // Create a placeholder buffer (filled with zeros) to confirm the GPU writes into it
    std::vector<double> h_dummy(
        num_systems * DummyModel::N_EQ * static_cast<int>(query_exact.size()),
        0.0
    );

    auto result_exact     = run_rk45<DummyModel>(h_y0, t0, tf, query_exact);
    const auto& h_dense_exact = result_exact.second;
    // ───────── END SECOND GPU RUN ─────────

    // ───────── Write test.csv at t = 0, 1, 2, 3, 4, 5 ─────────
    // This file collects values at integer times: initial (0), query points (1–4),
    // and final (5). Each row begins with “time”, followed by Var<i>_sys<s> columns.
    {
        std::ofstream test_file("test.csv");
        // Header: time, Var0_sys0, Var1_sys0, …, Var(N_EQ-1)_sys0, Var0_sys1, …
        test_file << "time";
        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < DummyModel::N_EQ; ++i) {
                test_file << ",Var" << i << "_sys" << s;
            }
        }
        test_file << "\n";

        // For each integer time td = 0, 1, 2, 3, 4, 5
        for (double td = 0.0; td <= tf; td += 1.0) {
            test_file << std::fixed << std::setprecision(1) << td;

            for (int s = 0; s < num_systems; ++s) {
                for (int i = 0; i < DummyModel::N_EQ; ++i) {
                    double value = 0.0;

                    if (td == 0.0) {
                        // At t = 0, use the initial condition (all ones)
                        value = 1.0;
                    }
                    else if (td == tf) {
                        // At t = 5, use the final state from the first GPU run
                        value = h_y_final[s * DummyModel::N_EQ + i];
                    }
                    else {
                        // For t = 1, 2, 3, 4, use the result from h_dense_exact
                        int qidx = static_cast<int>(td) - 1;  // index into query_exact
                        int base = s * (DummyModel::N_EQ * static_cast<int>(query_exact.size()))
                                   + i * static_cast<int>(query_exact.size())
                                   + qidx;
                        value = h_dense_exact[base];
                    }
                    test_file << "," << std::setprecision(9) << value;
                }
            }
            test_file << "\n";
        }
    }
    return 0;
}
