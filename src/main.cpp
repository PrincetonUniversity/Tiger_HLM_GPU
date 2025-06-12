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

#include <mpi.h>
#include "output_series.hpp"          // series output to netcdf (serial version). ALWAYS INCLUDE

#include "chrono" // for timing

#include "models/model_204.hpp"      // brings in SpatialParams
#include "stream.hpp"                // Stream<Model> wrapper (id, next_id, SpatialParams, y0)
# include "radau_kernel.cuh"       // Radau‐only kernel for Model204
//# include "rk45_kernel.cuh"
//#include "solver/rk45_kernel.cu" // RK45+Radau kernel for Model204

#include "I_O/forcing_loader.hpp"   // defines NetCDFLoader, LookupMapper
#include <iostream>                 // for std::cerr, std::cout
#include "I_O/forcing_data.h"

// ────────── Device‐side constant pointers, checking kernel──────────
__global__ void checkForcingPtr() {
  printf("d_forc_data = %p, nForc = %zu\n", d_forc_data, nForc);
}



// ────────── Debugging Forcings ──────────
__global__ void debugForcings(const float *forc, size_t nF, int ns) {
  // print the first time‐slice of each of the first 4 systems
  if (threadIdx.x < 4 && blockIdx.x == 0) {
    int t = 0;               // time‐step
    int s = threadIdx.x;     // stream index
    size_t idx = t*ns + s;
    printf(" for sys %d, t=%d → forc = %f\n", s, t, forc[idx]);
  }
}


__global__ void debugForcings2(const float *forc, int ns) {
    int s = threadIdx.x;
    if (blockIdx.x==0 && s < 4) {       // print for first 4 streams
        int t = 0;                      // time‐step 0
        // forcing #0 lives at offset 0
        size_t idx0 = /* offset of f=0 */ 0 + size_t(t)*ns + s;
        printf("forcing[0], sys=%d, t=0 → %f\n", s, forc[idx0]);

        // forcing #1 starts after all timesteps of forcing #0:
        size_t offset1 = c_forc_nT[0] * size_t(ns);
        size_t idx1    = offset1 + size_t(t)*ns + s;
        printf("forcing[1], sys=%d, t=0 → %f\n", s, forc[idx1]);
    }
}

// __global__ void debugForcingsMulti(const float *forc, int ns) {
//     int s = threadIdx.x;           // stream index
//     if (blockIdx.x==0 && s < 4) {  // only do streams 0..3
//         size_t offset1 = c_forc_nT[0] * size_t(ns);
//         for (int t = 0; t < 5; ++t) {
//             // forcing 0 at time t
//             size_t idx0 = size_t(t)*ns + s;
//             printf("f0, sys=%d, t=%d → %f\n", s, t, forc[idx0]);
//             // forcing 1 at time t
//             size_t idx1 = offset1 + size_t(t)*ns + s;
//             printf("f1, sys=%d, t=%d → %f\n", s, t, forc[idx1]);
//         }
//     }
// }

__global__ void debugForcingsMulti(const float *forc, int ns) {
    int s = threadIdx.x;
    if (blockIdx.x==0 && s < 4) {
        // block‐start of second forcing:
        size_t offset1 = c_forc_nT[0] * size_t(ns);
        size_t samples1 = c_forc_nT[1];    // e.g. 2 days
        // print every minute for the *first* daily‐step interval:
        for (int t = 0; t < min(samples1, 10UL); ++t) {
            size_t idx0 = /* first forcing */   size_t(t)*ns + s;
            size_t idx1 = offset1 + size_t(t)*ns + s;
            printf("t=%3d → pr=%7.3f, t2m=%7.3f\n", t, forc[idx0], forc[idx1]);
        }
        // then sample at the day boundary:
        int day1 = int(c_forc_nT[1] * c_forc_dt[1] * 60.0);  // dt=24h → 1440 min
        size_t idx1b = offset1 + size_t(day1/1)*ns + s;     // sampleIdx=1
        printf("at t=%d min → t2m=%7.3f\n", day1, forc[idx1b]);
    }
}

// Print pr and t2m for each minute up to first 2 hours and at daily boundary
__global__ void debugMinuteForcings(const float *forc, int ns) {
    int s = threadIdx.x;
    if (blockIdx.x == 0 && s < 1) {             // do first 2 systems, s=0..1
        size_t offset_pr  = 0;                   // pr is j=0 block
        size_t offset_t2m = c_forc_nT[0] * ns;   // t2m is j=1 block

        int max_min = 120;                       // first 120 minutes (2 h)
        for (int t = 0; t <= max_min; ++t) {
            // sample index in pr block:
            size_t idx_pr  = offset_pr  + size_t(t)*ns + s;
            // sample index in t2m block:
            // compute minute‐index into daily samples:
            double dt_t2m_min = c_forc_dt[1] * 60.0; // 24*60 = 1440
            // sampleIdx = floor(t / dt_t2m_min) → 0 for t<1440, 1 for t≥1440
            size_t sampleIdx_t2m = (t < (int)dt_t2m_min ? 0 : 1);
            size_t idx_t2m = offset_t2m + sampleIdx_t2m*ns + s;

            printf("sys=%d  t=%4d min → pr=%7.3f  t2m=%7.3f\n",
                   s, t,
                   forc[idx_pr],
                   forc[idx_t2m]);
        }

        // also show at exactly 1440 min (1 day) and at 2880 min (2 days)
        int day1 = int(c_forc_dt[1]*60.0);      // =1440
        int day2 = day1 * 2;                   // =2880
        for (int t : {day1, day2}) {
            size_t idx_pr  = offset_pr  + size_t(t)*ns + s; 
            size_t sampleIdx_t2m = (t < day1 ? 0 : (t < day2 ? 1 : 1));
            size_t idx_t2m = offset_t2m + sampleIdx_t2m*ns + s;
            printf("sys=%d  t=%4d min → pr=%7.3f  t2m=%7.3f\n",
                   s, t,
                   forc[idx_pr],
                   forc[idx_t2m]);
        }
    }
}

// ────────── In src/solver/rk45_kernel.cu, above testKernel ──────────

__global__ void debugHolding(const float *forc, int ns) {
    int sys = blockIdx.x * blockDim.x + threadIdx.x;
    if (sys != 0) return;      // only print for system 0

    // offsets into the big forcing array
    size_t offset_pr  = 0;
    size_t offset_t2m = size_t(c_forc_nT[0]) * ns;

    // sampling intervals in minutes
    // (c_forc_dt are in hours)
    double dt_pr_min   = c_forc_dt[0] * 60.0;   // e.g. 1 h→60 min
    double dt_t2m_min  = c_forc_dt[1] * 60.0;   // e.g. 24 h→1440 min

    printf(" t   →  pr[idx_pr]   |  t2m[idx_t2m]\n");
    for (int t = 0; t <= 180; ++t) {  // print first 3 hours = 180 min
        // integer sample index for each forcing:
        int idx_pr  = int(t / dt_pr_min);
        int idx_t2m = int(t / dt_t2m_min);

        // clamp to valid range
        if (idx_pr  >= int(c_forc_nT[0])) idx_pr  = int(c_forc_nT[0]) - 1;
        if (idx_t2m >= int(c_forc_nT[1])) idx_t2m = int(c_forc_nT[1]) - 1;

        float pr_val  = forc[offset_pr  + size_t(idx_pr)  * ns + sys];
        float t2m_val = forc[offset_t2m + size_t(idx_t2m) * ns + sys];

        printf("t=%3d → pr[%2d]=%7.3f   |   t2m[%2d]=%7.3f\n",
               t, idx_pr, pr_val,
               idx_t2m, t2m_val);
    }
}

// ────────── End debugging forcings ──────────





// ───────── Minimal test kernel ─────────
__global__ void testKernel() { /* nothing */ }

// ───────── Tiny kernel to verify one SpatialParams via printf ─────────
__global__ void debugParams(const SpatialParams* sp) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU sees: stream=%ld, Hu=%g, infil=%g, perco=%g\n",
               sp[0].stream, sp[0].Hu, sp[0].infil, sp[0].perco);
    }
}

// ─────────────────────────────────────────────────────────────

// ───────── Debugging device-side full print of spatial params ─────────
// Print every stream's SpatialParams from the GPU.
// ─────────────────────────────────────────────────────────────────────────
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
// ───────── ended debugging ──────────────────────────────────────────────────────


// ────────────────────────────────────────────────
// Debug‐kernel: call your RHS once for one stream
// ────────────────────────────────────────────────
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
        Model204::rhs(0.0, y, dydt, Model204::N_EQ, sys_id, sp, d_forc_data, nForc);

        // Print the resulting derivatives cleanly
        printf("[DebugRHS]   dydt: ");
        for (int i = 0; i < Model204::N_EQ; ++i) {
            printf("%g ", dydt[i]);
        }
        printf("\n[DebugRHS] ====  END RHS debug  ====\n\n");
    }
}


// ───────── ended debugging ─────────────────────────────────────────────



// ────────── Main function to run the RK45 solver on GPU ───────────────────

int main(int argc, char** argv) {

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
    auto spatialParams = loadSpatialParams("../data/small_test.csv");//10 links
    
    // Query one of the NetCDF files for its spatial dimensions
    NetCDFLoader prLoader("../data/pr_hourly_era5land_2019.nc", "pr");
    size_t lat_size = prLoader.getLatSize();
    size_t lon_size = prLoader.getLonSize();

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

    // ─── Debugging full host→device→host compare ───────────────────────────
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

    // ──── ended debugging ──────────────────────────────────────────────────────

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

    // // ────Debugging: launch debugAllParams to print every stream ─────────
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
    // ───────── ended debugging ────────────────────────────────────────────────────────

    // Verify via checkDevParamsKernel204
    checkDevParamsKernel204<<<1,1>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "checkDevParamsKernel204 failed: %s\n",
                    cudaGetErrorString(err));
        return 1;
    }

    // Launch the tiny debugParams kernel
    debugParams<<<1,32>>>(d_sp);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "debugParams kernel failed: %s\n", cudaGetErrorString(err));
    } else {
        std::puts("debugParams kernel ran successfully!!!");
    }
    // ───────── end debugging spatial params ────────────────────────────────────


    // ─── Build streamPoint lookup ─────────────────────────────────────────
    LookupMapper lm("../data/small_example_pr_lookup.csv");
    if (!lm.load()) {
        std::cerr << "Lookup load failed\n";
        return 1;
    }
    // one flat index per system
    std::vector<size_t> streamPoint(num_systems);
    for (int s = 0; s < num_systems; ++s) {
        auto [lat, lon] = lm.getLatLon(streams[s].id);
        streamPoint[s] = lat * lon_size + lon;
    }

    // ─── Adding forcings (first 2 days only) ─────────────────────────────────
    struct NCForcing {
        std::string path, var;
        double      dt;    // hours per time step
    };
    std::vector<NCForcing> ncForcings = {
        {"../data/pr_hourly_era5land_2019.nc",    "pr",  1.0},
        {"../data/t2m_daily_avg_era5land_2019.nc","t2m", 24.0}
    };
    int nForc = int(ncForcings.size());

    std::vector<float>  h_forc_data;
    std::vector<double> h_forc_dt;
    std::vector<size_t> h_forc_nT;
    h_forc_dt .reserve(nForc);
    h_forc_nT .reserve(nForc);
    h_forc_data.reserve(nForc * num_systems * 48);

    constexpr double daysToLoad = 2.0;  // grab only first 2 days

    for (auto &fm : ncForcings) {
        NetCDFLoader loader(fm.path, fm.var);
        size_t fullTime = loader.getTimeSize();

        // how many steps in 2 days?
        size_t steps2d = size_t(std::round(daysToLoad * 24.0 / fm.dt));
        steps2d = std::min(fullTime, steps2d);

        auto raw = loader.loadTimeChunk(0, steps2d);

        h_forc_dt.push_back(fm.dt);
        h_forc_nT.push_back(steps2d);

        size_t gridPts = loader.getLatSize() * loader.getLonSize();
        float *basePtr = raw.get();

        for (size_t t = 0; t < steps2d; ++t) {
            float *slice = basePtr + t * gridPts;
            for (int s = 0; s < num_systems; ++s) {
                h_forc_data.push_back(slice[ streamPoint[s] ]);
            }
        }
    }

    // C) Upload to device
    float* d_forc_ptr = nullptr;
    cudaMalloc(&d_forc_ptr, sizeof(float) * h_forc_data.size());
    cudaMemcpy(d_forc_ptr,
            h_forc_data.data(),
            sizeof(float) * h_forc_data.size(),
            cudaMemcpyHostToDevice);

    // D) Push dt, nT, pointer and count into device symbols
    {
        // copy forcing time‐step sizes (hours)
        cudaMemcpyToSymbol(c_forc_dt, h_forc_dt.data(),
                        sizeof(double) * nForc);
        // copy number of samples per forcing
        cudaMemcpyToSymbol(c_forc_nT, h_forc_nT.data(),
                        sizeof(size_t) * nForc);

        // copy pointer to big forcing array
        cudaMemcpyToSymbol(d_forc_data, &d_forc_ptr,
                        sizeof(d_forc_ptr));
        // copy the count
        cudaMemcpyToSymbol(nForc,       &nForc,
                        sizeof(nForc));
    }

    // ────────── End uploading forcings ──────────

    // ────────── Debugging forcings ──────────
    // Uncomment the following lines to debug forcings
    // cudaError_t err2 = cudaMemcpyToSymbol(d_forc_data, &d_forc_ptr, sizeof(d_forc_ptr));
    // printf("copy-to-symbol(forcing ptr) → %s\n", cudaGetErrorString(err2));

    // debugForcings<<<1,4>>>(d_forc_ptr, h_forc_data.size(), num_systems);
    // cudaDeviceSynchronize();

    // // check first time‐step of both forcings for the first few streams:
    // debugForcings2<<<1,4>>>(d_forc_ptr, num_systems);
    // cudaDeviceSynchronize();

    // debugForcingsMulti<<<1,4>>>(d_forc_ptr, num_systems);
    // cudaDeviceSynchronize();

    // // debug minute‐by‐minute stepping of pr (j=0) and t2m (j=1)
    // debugMinuteForcings<<<1,4>>>(d_forc_ptr, num_systems);
    // cudaDeviceSynchronize();

    // // debug first 3 hours holding behavior for sys=0
    // debugHolding<<<1,1>>>(d_forc_ptr, num_systems);
    // cudaDeviceSynchronize();

    // // debug the holding behavior for pr (hourly) and t2m (daily)
    // debugHolding<<<1, 1>>>(d_forc_ptr, num_systems);
    // cudaDeviceSynchronize();

    
    // ─────────End forcing─────────


    // ───────── Define time span (first‐2‐day test) ─────────
    const double t0 = 0.0;
    const double tf = 2 * 24.0 * 60.0;  // 2 days in minutes
    

    // ───────── Compute SciPy‐style initial step and upload devParams ─────────
    {
        int N_EQ = Model204::N_EQ;
        const SpatialParams* host_sp_ptr = spatialParams.data();
        std::vector<double> y0_cpu(N_EQ, 0.0), f0_cpu(N_EQ), scale(N_EQ);
        Model204::rhs(t0, y0_cpu.data(), f0_cpu.data(), N_EQ, 0, host_sp_ptr, h_forc_data.data(), nForc);

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

    // ───────── Flatten initial y ─────────
    std::vector<double> h_y0(num_systems * Model204::N_EQ);//use this
    for (int s = 0; s < num_systems; ++s) {
        for (int i = 0; i < Model204::N_EQ; ++i) {
            h_y0[s * Model204::N_EQ + i] = streams[s].y0[i];
        }
    }
  

    // ───────── Define query times (first 2 days, hourly) ─────────
    std::vector<double> h_query_times;
    for (double t = t0; t <= tf; t += 60.0) {
        h_query_times.push_back(t);
    }
    int num_queries = int(h_query_times.size());

    

    // ───────── Allocate GPU buffers & launch solver ─────────
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

    // —─────────────────────── launching error kernel
    cudaMemcpyToSymbol(d_forc_data, &d_forc_ptr, sizeof(d_forc_ptr));


    rk45_then_radau_multi<Model204><<<blocks,threads>>>(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        ns, nq,
        t0, tf,
        d_sp,      // spatial params
        d_stiff,   // stiffness flags
        d_forc_ptr,  // forcing data pointer
        nForc         // number of forcings
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

    // if(rank)
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

    // ———————————————————————————————— 
    // ───────── Write to netcdf  ─────────

    // !!!! NEED TO CHANGE TO ACCESS ACTUAL ID AND STATE INDEXES !!!.    
    int N_EQ = Model204::N_EQ;
    std::vector<int> linkid_vals(num_systems);
    std::vector<int> state_vals(N_EQ);
    for (int s = 0; s < num_systems; ++s) linkid_vals[s] = s;
    for (int v = 0; v < N_EQ; ++v) state_vals[v] = v;

    //Netcdf file attributes (will be defined in yaml)
    std::string dense_filename = "/scratch/gpfs/am2192/dense_example.nc";
    std::string final_filename = "/scratch/gpfs/am2192/final_example.nc";
    int compression_level = 0;

    // Write only the final time step (2D output)
    write_final_netcdf(final_filename,
                    h_y_final.data(),
                    linkid_vals.data(),
                    state_vals.data(),
                    num_systems,
                    N_EQ,
                    compression_level);

    auto start = std::chrono::high_resolution_clock::now();
    write_dense_netcdf(dense_filename,
                    h_dense.data(),
                    h_query_times.data(),
                    linkid_vals.data(),
                    state_vals.data(),
                    num_queries,
                    num_systems,
                    N_EQ,
                    compression_level);
    std::cout << "Write out finished" << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "write_dense_netcdf took " << elapsed.count() << " seconds.\n";

    return 0;
}
