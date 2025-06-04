#include <cstdio>
#include <algorithm>
#include <fstream>
#include <vector>

#include "solver/rk45_api.hpp"   // run_rk45<Model>() and setModelParameters<Model>()
#include "models/model_dummy.hpp" // DummyModel and Parameters
#include "model_registry.hpp"

int main() {
    using namespace rk45_api;

    // ───────── Define problem size ─────────
    constexpr int num_systems = 4;
    constexpr int num_queries = 1000;
    const double t0 = 0.0, tf = 5.0;

    // ───────── Build host initial states ─────────
    std::vector<double> h_y0(num_systems * DummyModel::N_EQ);
    for (int s = 0; s < num_systems; ++s) {
        for (int i = 0; i < DummyModel::N_EQ; ++i) {
            h_y0[s * DummyModel::N_EQ + i] = 1.0;
        }
    }

    // ───────── Build & sort query times ─────────
    std::vector<double> h_query_times(num_queries);
    for (int q = 0; q < num_queries; ++q) {
        h_query_times[q] = tf * (q + 1.0) / (num_queries + 1.0);
    }
    std::sort(h_query_times.begin(), h_query_times.end());

    // ───────── Set RK45 parameters on host and copy to device ─────────
    DummyModel::Parameters hp;
    hp.initialStep = 0.01;
    hp.safety      = 0.9;
    hp.minScale    = 0.2;
    hp.maxScale    = 5.0;
    setModelParameters<DummyModel>(hp);

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
            printf(" %.6f", h_y_final[s * DummyModel::N_EQ + i]);
        }
        printf("\n");
    }

    return 0;
}
