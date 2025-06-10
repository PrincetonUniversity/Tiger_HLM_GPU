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
#include "model_registry.hpp"        // setModelParameters<Model204>()
#include "rk45_step_dense.cuh"       // device kernels for one RK45 step + dense‐output
#include "event_detector.cuh"        // device code for slope‐jump/stiffness detection
#include "small_lu.cuh"              // small‐matrix LU solver used by implicit Radau
#include "solver/rk45_api.hpp"       // host‐side RK45 API: setup_gpu_buffers, launch_rk45_kernel, etc.
#include "radau_step_dense.cuh"      // device kernels for Radau‐IIA step + dense‐output
//#include "models/active_model.hpp"   // defines Model204 alias & __constant__ devParams
#include "parameters_loader.hpp"     // CSV loader for SpatialParams
#include "models/model_204.hpp"      // brings in SpatialParams
#include "stream.hpp"                // Stream<Model> wrapper (id, next_id, SpatialParams, y0)
# include "radau_kernel.cuh"       // Radau‐only kernel for Model204
//# include "rk45_kernel.cuh"
//#include "solver/rk45_kernel.cu" // RK45+Radau kernel for Model204
// ───────── Minimal test kernel ─────────
__global__ void testKernel() { /* nothing */ }

// ───────── Tiny kernel to verify one SpatialParams via printf ─────────
__global__ void debugParams(const SpatialParams* sp) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU sees: stream=%ld, Hu=%g, infil=%g, perco=%g\n",
               sp[0].stream, sp[0].Hu, sp[0].infil, sp[0].perco);
    }
}

// ----------------------------------------------------------------------------
// ----- Debugging device-side full print of spatial params -----
// Print every stream's SpatialParams from the GPU.
// ----------------------------------------------------------------------------
__global__ void debugAllParams(const SpatialParams* sp, int N) {
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys < N) {
        printf("sys=%3d → stream=%10ld, Hu=%6.3f, infil=%6.3f, perco=%6.3f, L=%6.3f, A_h=%6.3f\n",
               sys,
               sp[sys].stream,
               sp[sys].Hu,
               sp[sys].infil,
               sp[sys].perco,
               sp[sys].L,
               sp[sys].A_h
        );
    }
}
// ----- ended debugging -----


// --------------------------------------------------
// Debug‐kernel: call your RHS once for one stream
// --------------------------------------------------
__global__ void debugRHS(const SpatialParams* sp, int sys_id) {
    // Only one thread actually prints
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Header so you can spot this block in the log
        printf("[DebugRHS] ==== BEGIN RHS debug for stream index %d ====\n",
               sys_id);
        // Print the stream ID and raw parameters
        printf("[DebugRHS]   Stream ID = %ld\n", sp[sys_id].stream);
        printf("[DebugRHS]   Parameters: Hu=%g, infil=%g, perco=%g, L=%g, A_h=%g\n",
               sp[sys_id].Hu,
               sp[sys_id].infil,
               sp[sys_id].perco,
               sp[sys_id].L,
               sp[sys_id].A_h);

        // Prepare a trivial y-vector and compute dydt
        double y[Model204::N_EQ]   = {1.0, 1.0, 1.0, 1.0, 1.0};
        double dydt[Model204::N_EQ];
        Model204::rhs(0.0, y, dydt, Model204::N_EQ, sys_id, sp);

        // Print the resulting derivatives cleanly
        printf("[DebugRHS]   dydt: ");
        for (int i = 0; i < Model204::N_EQ; ++i) {
            printf("%g ", dydt[i]);
        }
        printf("\n[DebugRHS] ====  END RHS debug  ====\n\n");
    }
}


// ----- ended debugging -----


// ----------------------------------------------------------------------------

int main() {
    // ───────── Print GPU properties ─────────
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::printf("Running on GPU %s (SM %d.%d)\n",
                prop.name, prop.major, prop.minor);
    // _____________ end checking GPU properties _______________

    // ───────── Test that even a trivial kernel will launch ─────────
    testKernel<<<1,1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "testKernel launch failed: %s\n",
                    cudaGetErrorString(err));
    } else {
        std::puts("testKernel launch: OK");
    }
    cudaDeviceSynchronize();

   

    using namespace rk45_api;

    // ───────── 0) load per‐stream spatial parameters ─────────
    auto spatialParams = loadSpatialParams("../data/small_test.csv");

    // build a vector of Stream<Model204>, using a common y0
    std::array<double, Model204::N_EQ> y0_common = {0.01, 3.0, 0.0, 5.0, 0.2};
    std::vector< Stream<Model204> > streams;
    streams.reserve(spatialParams.size());
    for (auto const &sp : spatialParams) {
        streams.emplace_back(sp, y0_common);
    }
    int num_systems = int(streams.size());

    // ────────── Debug: copy SpatialParams to device and verify ──────────
    std::vector<SpatialParams> hostSP;
    hostSP.reserve(num_systems);
    for (auto const &st : streams) {
        hostSP.push_back(st.sp);
    }
    size_t byteCount = num_systems * sizeof(SpatialParams);

    SpatialParams* d_sp = nullptr;
    //cudaError_t err = cudaMalloc(&d_sp, byteCount);
    err = cudaMalloc(&d_sp, byteCount);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_sp) failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_sp, hostSP.data(), byteCount, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy(d_sp) failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Host‐side round‐trip check (first element)
    SpatialParams check0;
    err = cudaMemcpy(&check0, d_sp, sizeof(SpatialParams), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy back failed: %s\n", cudaGetErrorString(err));
    } else {
        std::printf("HOST→DEVICE round-trip: stream=%ld, Hu=%g, infil=%g, perco=%g\n",
                    check0.stream, check0.Hu, check0.infil, check0.perco);
    }

    // ------ Debugging full host→device→host compare ------
    {
        std::vector<SpatialParams> hostCheck(num_systems);
        err = cudaMemcpy(hostCheck.data(), d_sp, byteCount, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Full round-trip cudaMemcpy back failed: %s\n", cudaGetErrorString(err));
        } else {
            for (int i = 0; i < num_systems; ++i) {
                const auto &in  = hostSP[i];
                const auto &out = hostCheck[i];
                if (in.stream != out.stream ||
                    fabs(in.Hu - out.Hu)       > 1e-12 ||
                    fabs(in.infil - out.infil) > 1e-12 ||
                    fabs(in.perco - out.perco) > 1e-12
                    /* add more fields if desired */
                ) {
                    std::fprintf(stderr,
                        "Mismatch at idx %d: CSV(stream=%ld,Hu=%g,...) vs GPU(stream=%ld,Hu=%g,...)\n",
                        i,
                        in.stream, in.Hu,
                        out.stream, out.Hu
                    );
                }
            }
        }
    }

    // ----- ended debugging -----

    // Populate the device‐constant pointer so kernels see it
    err = cudaMemcpyToSymbol(devSpatialParamsPtr, &d_sp, sizeof(d_sp));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyToSymbol(devSpatialParamsPtr) failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

        // ─────── DebugRHS: verify Model204::rhs uses the right parameters ───────
    {
        int target_sys = 0;                // choose the stream index to inspect
        debugRHS<<<1,1>>>(d_sp, target_sys);
        cudaDeviceSynchronize();
    }


    

    // // ------ Debugging: launch debugAllParams to print every stream ------
    // {
    //     int dbgThreads = 32;
    //     int dbgBlocks  = (num_systems + dbgThreads - 1) / dbgThreads;
    //     debugAllParams<<<dbgBlocks, dbgThreads>>>(d_sp, num_systems);
    //     err = cudaDeviceSynchronize();
    //     if (err != cudaSuccess) {
    //         std::fprintf(stderr, "debugAllParams kernel failed: %s\n", cudaGetErrorString(err));
    //     } else {
    //         std::puts("debugAllParams kernel ran successfully");
    //     }
    // }
    // ----- ended debugging -----

    // Verify via checkDevParamsKernel204
    checkDevParamsKernel204<<<1,1>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "checkDevParamsKernel204 failed: %s\n",
                    cudaGetErrorString(err));
        return 1;
    }

    // Launch our tiny debugParams kernel
    debugParams<<<1,32>>>(d_sp);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "debugParams kernel failed: %s\n", cudaGetErrorString(err));
    } else {
        std::puts("debugParams kernel ran successfully!!!");
    }
    // __________ end debugging spatial params __________

    // ───────── Define integration interval and initial y ─────────
    const double t0 = 0.0;
    const double tf = 2 * 24.0 * 60.0;  // 2 days

    // flatten the initial y for *all* systems
    std::vector<double> h_y0(num_systems * Model204::N_EQ);
    for (int s = 0; s < num_systems; ++s) {
        for (int i = 0; i < Model204::N_EQ; ++i) {
            h_y0[s * Model204::N_EQ + i] = streams[s].y0[i];
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
        int N_EQ = Model204::N_EQ;
        const SpatialParams* host_sp_ptr = spatialParams.data();
        std::vector<double> y0_cpu(N_EQ, 0.0), f0_cpu(N_EQ), scale(N_EQ);
        Model204::rhs(t0, y0_cpu.data(), f0_cpu.data(), N_EQ, 0, host_sp_ptr);

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

        Model204::Parameters hp;
        hp.initialStep = h_guess;
        hp.rtol        = rtol;
        hp.atol        = atol;
        hp.safety      = 0.9;
        hp.minScale    = 0.2;
        hp.maxScale    = 10.0;
        setModelParameters<Model204>(hp);
    }

    // ───────── Allocate GPU buffers & launch solver ─────────
    // double *d_y0_all, *d_y_final_all, *d_query_times, *d_dense_all;
    // int ns, nq;
    // std::tie(d_y0_all, d_y_final_all, d_query_times, d_dense_all, ns, nq) =
    //     setup_gpu_buffers<Model204>(h_y0, h_query_times);

    double *d_y0_all, *d_y_final_all, *d_query_times, *d_dense_all;
    int    *d_stiff;                // NEW: flags buffer
    int     ns, nq;

    std::tie(d_y0_all,
            d_y_final_all,
            d_query_times,
            d_dense_all,
            d_stiff,          // ← now unpack 7 items
            ns,
            nq)
        = setup_gpu_buffers<Model204>(h_y0, h_query_times);


    // ───────── Diagnostic launch + checks ─────────
    const int THREADS_PER_BLOCK = 128;
    int numBlocks = (ns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 blocks(numBlocks), threads(THREADS_PER_BLOCK);

    std::printf("Launching kernel with %d blocks, %d threads/block (ns=%d)\n",
                numBlocks, THREADS_PER_BLOCK, ns);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Pre-launch cudaGetLastError: %s\n",
                    cudaGetErrorString(err));
    }

    rk45_then_radau_multi<Model204><<<blocks,threads>>>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq,
        t0, tf,
        d_sp, d_stiff
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n",
                    cudaGetErrorString(err));
        return 1;
    } else {
        std::puts("Kernel launch: OK");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel execution failed: %s\n",
                    cudaGetErrorString(err));
        return 1;
    } else {
        std::puts("Kernel execution: OK");
    }
    // ————————————————————————————————

    // ───────── Retrieve results & free buffers ─────────
    auto [h_y_final, h_dense] = retrieve_and_free<Model204>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all, d_stiff,
        ns, nq,
        t0, tf, d_sp
    );

    // ───────── Write final.csv ─────────
    {
        std::ofstream final_file("final_204_a.csv");
        final_file << "h_snow";
        for (int i = 1; i < Model204::N_EQ; ++i) {
            final_file << ",var" << i;
        }
        final_file << "\n";
        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < Model204::N_EQ; ++i) {
                final_file << h_y_final[s * Model204::N_EQ + i];
                if (i + 1 < Model204::N_EQ) final_file << ",";
            }
            final_file << "\n";
        }
    }

    // ───────── Write dense.csv ─────────
    {
        std::ofstream dense_file("dense_204_a.csv");
        dense_file << "time";
        for (int s = 0; s < num_systems; ++s) {
            for (int i = 0; i < Model204::N_EQ; ++i) {
                dense_file << ",var" << i << "_sys" << s;
            }
        }
        dense_file << "\n";
        for (int q = 0; q < num_queries; ++q) {
            dense_file << std::fixed << std::setprecision(8)
                       << h_query_times[q];
            for (int s = 0; s < num_systems; ++s) {
                for (int i = 0; i < Model204::N_EQ; ++i) {
                    int idx = (s * num_queries + q) * Model204::N_EQ + i;
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
        for (int i = 0; i < Model204::N_EQ; ++i) {
            std::printf(" y%d=%.6f", i, h_y_final[s * Model204::N_EQ + i]);
        }
        std::printf("\n");
    }



    return 0;
}
