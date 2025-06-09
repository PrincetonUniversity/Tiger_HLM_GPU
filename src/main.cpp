#include <cstdio>
#include <cuda_runtime.h>

#include <vector>
#include <array>
#include <fstream>
#include <iomanip>
#include <utility>    // for std::tie when unpacking tuples
#include <cmath>      // for std::sqrt, std::pow, std::fabs
#include <algorithm>  // for std::max, std::fmin, std::fmax

#include "rk45.h"                    // core RK45 solver interface (host‐side API)
#include "model_registry.hpp"        // setModelParameters<ActiveModel>()
#include "rk45_step_dense.cuh"       // device kernels for one RK45 step + dense‐output
#include "event_detector.cuh"        // device code for slope‐jump/stiffness detection
#include "small_lu.cuh"              // small‐matrix LU solver used by implicit Radau
#include "solver/rk45_api.hpp"       // host‐side RK45 API: setup_gpu_buffers, launch_rk45_kernel, etc.
#include "radau_step_dense.cuh"      // device kernels for Radau‐IIA step + dense‐output
#include "models/active_model.hpp"   // defines ActiveModel alias & __constant__ devParams
//#include "models/model_204.hpp"    // ActiveModel struct and rhs declaration

#include "parameters_loader.hpp"     // CSV loader for SpatialParams
#include "models/model_204.hpp"      // brings in SpatialParams
#include "stream.hpp"                // Stream<Model> wrapper (id, next_id, SpatialParams, y0)

int main() {
    using namespace rk45_api;

    // ───────── 0) load per‐stream spatial parameters ─────────
    auto spatialParams = loadSpatialParams("../data/small_test.csv");

    // build a vector of Stream<ActiveModel>, using a common y0
    std::array<double, ActiveModel::N_EQ> y0_common = {0.01, 3.0, 0.0, 5.0, 0.2};
    std::vector< Stream<ActiveModel> > streams;
    streams.reserve(spatialParams.size());
    for (auto const &sp : spatialParams) {
        streams.emplace_back(sp, y0_common);
    }
    int num_systems = int(streams.size());

    // flatten just the SpatialParams into a host array
    std::vector<SpatialParams> hostSP;
    hostSP.reserve(num_systems);
    for (auto const &st : streams) {
        hostSP.push_back(st.sp);
    }
    // // copy into the device constant memory array declared in model_204_global.cu
    // size_t byteCount = num_systems * sizeof(SpatialParams);
    // cudaMemcpyToSymbol(devSpatialParams, hostSP.data(), byteCount);
    // ────────── new: copy into a device-side global array ──────────
    size_t byteCount = num_systems * sizeof(SpatialParams);
    SpatialParams* d_sp = nullptr;
    cudaMalloc(&d_sp, byteCount);
    cudaMemcpy(d_sp, hostSP.data(), byteCount, cudaMemcpyHostToDevice);
    // populate devSpatialParamsPtr in constant memory:// use global memory here
    //cudaMemcpyToSymbol(devSpatialParamsPtr, &d_sp, sizeof(d_sp));

    // <<< DEBUG: verify on-device pointer >>>
    checkDevParamsKernel204<<<1,1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "checkDevParamsKernel204 failed: %s\n",
                    cudaGetErrorString(err));
        return 1;
    }

    // ───────── Define integration interval and initial y ─────────
    const double t0 = 0.0;
    const double tf = 2 * 24.0 * 60.0;  // 2 days

    // flatten the initial y for *all* systems
    std::vector<double> h_y0(num_systems * ActiveModel::N_EQ);
    for (int s = 0; s < num_systems; ++s) {
        for (int i = 0; i < ActiveModel::N_EQ; ++i) {
            h_y0[s * ActiveModel::N_EQ + i] = streams[s].y0[i];
        }
    }

    // ───────── Build query times ─────────
    std::vector<double> h_query_times;
    for (double t = t0; t <= tf; t += 1.0) {
        h_query_times.push_back(t);
    }
    int num_queries = int(h_query_times.size());

    // ───────── Compute SciPy‐style initial step and upload devParams ─────────
    {
        int N_EQ = ActiveModel::N_EQ;
        // Host‐side pointer into the loaded spatialParams vector:
        const SpatialParams* host_sp_ptr = spatialParams.data();
        std::vector<double> y0_cpu(N_EQ, 0.0), f0_cpu(N_EQ), scale(N_EQ);
        ActiveModel::rhs(t0, y0_cpu.data(), f0_cpu.data(), N_EQ, /*sys=*/0, /*host copy of d_sp*/ host_sp_ptr);

        double rtol = 1e-6, atol = 1e-9;
        for (int i = 0; i < N_EQ; ++i)
            scale[i] = atol + rtol * std::fabs(y0_cpu[i]);
        double d0 = 0, d1 = 0;
        for (int i = 0; i < N_EQ; ++i) {
            d0 += std::pow(y0_cpu[i]/scale[i],2);
            d1 += std::pow(f0_cpu[i]/scale[i],2);
        }
        d0 = std::sqrt(d0);
        d1 = std::sqrt(d1);
        double h_guess = std::max(1e-6, 0.01 * d0 / (d1 + 1e-16));

        ActiveModel::Parameters hp;
        hp.initialStep = h_guess;
        hp.rtol        = rtol;
        hp.atol        = atol;
        hp.safety      = 0.9;
        hp.minScale    = 0.2;
        hp.maxScale    = 10.0;
        setModelParameters<ActiveModel>(hp);
    }

    // ───────── Allocate GPU buffers & launch solver ─────────
    double *d_y0_all, *d_y_final_all, *d_query_times, *d_dense_all;
    int ns, nq;
    std::tie(d_y0_all, d_y_final_all, d_query_times, d_dense_all, ns, nq) =
        setup_gpu_buffers<ActiveModel>(h_y0, h_query_times);

    // launch_rk45_kernel<ActiveModel>(
    //     d_y0_all, d_y_final_all,
    //     d_query_times, d_dense_all,
    //     ns, nq,
    //     t0, tf
    // );

    launch_rk45_kernel<ActiveModel>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq,
        t0, tf, 
        d_sp
    );

    // ───────── Error‐check the kernel ─────────
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Kernel launch failed: %s\n",
                         cudaGetErrorString(err));
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Kernel execution failed: %s\n",
                         cudaGetErrorString(err));
            return 1;
        }
    }
    
    // ───────── Retrieve results & free buffers ─────────
    auto [h_y_final, h_dense] = retrieve_and_free<ActiveModel>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq
    );

    // ───────── Write final.csv ─────────
    {
        std::ofstream final_file("final_204_a.csv");
        final_file << "h_snow";
        for (int i = 1; i < ActiveModel::N_EQ; ++i) {
            final_file << ",var" << i;
        }
        final_file << "\n";

        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < ActiveModel::N_EQ; ++i) {
                final_file << h_y_final[s * ActiveModel::N_EQ + i];
                if (i + 1 < ActiveModel::N_EQ) final_file << ",";
            }
            final_file << "\n";
        }
    }

    // ───────── Write dense.csv ─────────
    {
        std::ofstream dense_file("dense_204_a.csv");
        dense_file << "time";
        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < ActiveModel::N_EQ; ++i) {
                dense_file << ",var" << i << "_sys" << s;
            }
        }
        dense_file << "\n";

        for (int q = 0; q < num_queries; ++q) {
            dense_file << std::fixed << std::setprecision(8)
                       << h_query_times[q];
            for (int s = 0; s < num_systems; ++s) {
                for (int i = 0; i < ActiveModel::N_EQ; ++i) {
                    int idx = (s * num_queries + q) * ActiveModel::N_EQ + i;
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
        for (int i = 0; i < ActiveModel::N_EQ; ++i) {
            std::printf(" y%d=%.6f", i, h_y_final[s * ActiveModel::N_EQ + i]);
        }
        std::printf("\n");
    }

    return 0;
}
