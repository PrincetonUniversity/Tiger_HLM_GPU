// main.cpp

#include <cstdio>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "solver/rk45_api.hpp"   // setup_gpu_buffers, launch_rk45_kernel, retrieve_and_free, run_rk45
#include "models/model_dummy.hpp" // DummyModel and its rhs, Parameters
#include "model_registry.hpp"     // setModelParameters<DummyModel>()

// Forward‐declared kernel hosted in model_dummy_global.cu:
extern __global__ void checkDevParamsKernel();

int main() {
    using namespace rk45_api;

    // ───────── Define problem size ─────────
    constexpr int num_systems = 4;      // how many parallel ODE systems
    constexpr int num_queries = 10000;  // how many dense‐output times
    const double t0 = 0.0, tf = 5.0;    // integration interval

    // ───────── Build host initial states ─────────
    // Allocate a flat vector of length (num_systems × N_EQ)
    // Each system’s initial condition is the constant vector [1,1,…,1] for now !!!
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
        // need the user to set this otherwise use the following defaults
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
        // Note: this formula selects a reasonable starting step, need to change this based on input !!!
        double h_guess = 0.01 * d0 / (d1 + 1e-16);

        // Copy into device‐Parameters exactly
        DummyModel::Parameters hp;
        hp.initialStep = h_guess;  // ← matches SciPy’s first‐step guess
        hp.rtol        = rtol;
        hp.atol        = atol;
        hp.safety      = 0.9;
        hp.minScale    = 0.2;
        hp.maxScale    = 10.0;
        setModelParameters<DummyModel>(hp);

        // Uncomment to verify on the GPU:
        // checkDevParamsKernel<<<1,1>>>();
        // cudaDeviceSynchronize();
        // printf("HOST:  SciPy‐style first step h0 = %.12f\n", h_guess);
    }

    // ───────── Run the solver on GPU ─────────
    // 1) Allocate GPU buffers & copy h_y0, h_query_times
    double *d_y0_all, *d_y_final_all, *d_query_times, *d_dense_all;
    int ns, nq;
    std::tie(d_y0_all, d_y_final_all, d_query_times, d_dense_all, ns, nq)
        = setup_gpu_buffers<DummyModel>(h_y0, h_query_times);

    // 2) Launch the GPU RK45 solver
    launch_rk45_kernel<DummyModel>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq,
        t0, tf
    );

    // 3) Copy back results & free GPU memory
    std::vector<double> h_y_final, h_dense;
    std::tie(h_y_final, h_dense) = retrieve_and_free<DummyModel>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq
    );

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
                if (i + 1 < DummyModel::N_EQ) final_file << ",";
            }
            final_file << "\n";
        }
    }

    // ───────── Write dense.csv ─────────
    {
        std::ofstream dense_file("dense.csv");
        // Header: time,Var0_sys0,…,Var4_sys0,Var0_sys1,…,Var4_sys3
        dense_file << "time";
        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < DummyModel::N_EQ; ++i) {
                dense_file << ",Var" << i << "_sys" << s;
            }
        }
        dense_file << "\n";

        // One line per query time in (0,5)
        for (int q = 0; q < num_queries; ++q) {
            dense_file << std::fixed << std::setprecision(8)
                       << h_query_times[q];
            for (int s = 0; s < num_systems; ++s) {
                for (int i = 0; i < DummyModel::N_EQ; ++i) {
                    int idx = (s * num_queries + q) * DummyModel::N_EQ + i;
                    dense_file << "," << std::setprecision(9)
                               << h_dense[idx];
                }
            }
            dense_file << "\n";
        }
    }

    // ───────── Print a quick summary ─────────
    std::printf("Final states at t = %.1f:\n", tf);
    for (int s = 0; s < num_systems; ++s) {
        std::printf(" System %d:", s);
        for (int i = 0; i < DummyModel::N_EQ; ++i) {
            std::printf(" H%d=%.6f", i, h_y_final[s * DummyModel::N_EQ + i]);
        }
        std::printf("\n");
    }

    // ───────── Second GPU run: request outputs at t = 1,2,3,4 ─────────
    std::vector<double> query_exact = {1.0, 2.0, 3.0, 4.0};

    // STEP 1 (again) for the new query set:
    std::tie(d_y0_all, d_y_final_all, d_query_times, d_dense_all, ns, nq)
        = setup_gpu_buffers<DummyModel>(h_y0, query_exact);

    // STEP 2 (again):
    launch_rk45_kernel<DummyModel>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq,
        t0, tf
    );

    // STEP 3 (again):
    std::vector<double> temp_final, h_dense_exact;
    std::tie(temp_final, h_dense_exact) = retrieve_and_free<DummyModel>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq
    );

    // ───────── Write test.csv (values at t = 0,1,2,3,4,5) ─────────
    {
        std::ofstream test_file("test.csv");
        test_file << "time";
        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < DummyModel::N_EQ; ++i) {
                test_file << ",Var" << i << "_sys" << s;
            }
        }
        test_file << "\n";

        for (double td = 0.0; td <= tf; td += 1.0) {
        // int Nsteps = static_cast<int>(std::floor(tf)) + 1;  // here tf==5.0 → Nsteps==6
        // for (int step = 0; step < Nsteps; ++step) {
        //     double td = static_cast<double>(step);
            test_file << std::fixed << std::setprecision(1) << td;
            for (int s = 0; s < num_systems; ++s) {
                for (int i = 0; i < DummyModel::N_EQ; ++i) {
                    double value = 0.0;
                    if (td == 0.0) {
                        value = 1.0;  // initial condition
                    }
                    else if (td == tf) {
                        value = h_y_final[s * DummyModel::N_EQ + i];
                    }
                    else {
                        int qidx = static_cast<int>(td) - 1;
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
