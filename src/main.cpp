#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <fstream>
#include <iomanip>
#include <utility>    // for std::tie when unpacking tuples
#include <cmath>      // for std::sqrt, std::pow, std::fabs
#include <algorithm>  // for std::max, std::fmin, std::fmax
#include <numeric>    // for std::iota
#include <cstdint> // for std::uint64_t, std::int64_t

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
//#include "output_writer.hpp"

#include "chrono" // for timing

#include "models/model_204.hpp"      // brings in SpatialParams
#include "stream.hpp"                // Stream<Model> wrapper (id, next_id, SpatialParams, y0)
# include "radau_kernel.cuh"       // Radau‐only kernel for Model204
//# include "rk45_kernel.cuh"
//#include "solver/rk45_kernel.cu" // RK45+Radau kernel for Model204

#include "I_O/forcing_loader.hpp"   // defines NetCDFLoader, LookupMapper
#include <iostream>                 // for std::cerr, std::cout
#include "I_O/forcing_data.h"



// Helper macros for CUDA error checking
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr,                                                    \
        "[CUDA ERROR] %s:%d: “%s” failed → %s\n",                             \
        __FILE__, __LINE__, #call, cudaGetErrorString(err));                  \
      std::exit(1);                                                           \
    }                                                                         \
  } while(0)

// Helpers to format YYYYMMDD and advance a tm by N days
static std::string formatDate(int Y,int M,int D) {
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", Y, M, D);
    return buf;
}
static void advanceDate(int &Y,int &M,int &D,int days) {
    std::tm tm = {};
    tm.tm_year = Y - 1900;
    tm.tm_mon  = M - 1;
    tm.tm_mday = D;
    // convert to UTC time_t
    time_t t = timegm(&tm);
    t += days * 24*3600;
    gmtime_r(&t, &tm);
    Y = tm.tm_year + 1900;
    M = tm.tm_mon  + 1;
    D = tm.tm_mday;
}


// ────────── Device‐side constant pointers, checking kernel──────────
// __global__ void checkForcingPtr() {
//   // printf("d_forc_data = %p, nForc = %d\n", (void*)d_forc_data, nForc);
//   // Use d_forc_ptr and int nForc already on host side
//   printf("d_forc_ptr = %p, nForc = %d\n", (void*)d_forc_ptr, nForc);


// }

// __global__ void checkForcingSymbols() {
//     printf("[checkForcingSymbols] d_forc_data = %p, nForc = %d\n", (void*)d_forc_data, nForc);

//     // Prevent pruning (even if just this line)
//     if (nForc < 0) printf("nForc is negative?\n");
// }

// __global__ void checkForcingSymbols(const float* ptr, int nf) {
//   printf(
//     "d_forc_data = %p, nForc(arg) = %d\n"
//     "   dt[0]=%g, dt[1]=%g, nT[0]=%zu, nT[1]=%zu\n",
//     (void*)ptr, nf,
//     c_forc_dt[0], c_forc_dt[1],
//     c_forc_nT[0], c_forc_nT[1]
//   );
// }








// ────────── Debugging Forcings ──────────
// __global__ void debugForcings(const float *forc, size_t nF, int ns) {
//   // print the first time‐slice of each of the first 4 systems
//   if (threadIdx.x < 4 && blockIdx.x == 0) {
//     int t = 0;               // time‐step
//     int s = threadIdx.x;     // stream index
//     size_t idx = t*ns + s;
//     printf(" for sys %d, t=%d → forc = %f\n", s, t, forc[idx]);
//   }
// }


// __global__ void debugForcings2(const float *forc, int ns) {
//     int s = threadIdx.x;
//     if (blockIdx.x==0 && s < 4) {       // print for first 4 streams
//         int t = 0;                      // time‐step 0
//         // forcing #0 lives at offset 0
//         size_t idx0 = /* offset of f=0 */ 0 + size_t(t)*ns + s;
//         printf("forcing[0], sys=%d, t=0 → %f\n", s, forc[idx0]);

//         // forcing #1 starts after all timesteps of forcing #0:
//         size_t offset1 = c_forc_nT[0] * size_t(ns);
//         size_t idx1    = offset1 + size_t(t)*ns + s;
//         printf("forcing[1], sys=%d, t=0 → %f\n", s, forc[idx1]);
//     }
// }

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

// __global__ void debugForcingsMulti(const float *forc, int ns) {
//     int s = threadIdx.x;
//     if (blockIdx.x==0 && s < 4) {
//         // block‐start of second forcing:
//         size_t offset1 = c_forc_nT[0] * size_t(ns);
//         size_t samples1 = c_forc_nT[1];    // e.g. 2 days
//         // print every minute for the *first* daily‐step interval:
//         for (int t = 0; t < min(samples1, 10UL); ++t) {
//             size_t idx0 = /* first forcing */   size_t(t)*ns + s;
//             size_t idx1 = offset1 + size_t(t)*ns + s;
//             printf("t=%3d → pr=%7.3f, t2m=%7.3f\n", t, forc[idx0], forc[idx1]);
//         }
//         // then sample at the day boundary:
//         int day1 = int(c_forc_nT[1] * c_forc_dt[1] * 60.0);  // dt=24h → 1440 min
//         size_t idx1b = offset1 + size_t(day1/1)*ns + s;     // sampleIdx=1
//         printf("at t=%d min → t2m=%7.3f\n", day1, forc[idx1b]);
//     }
// }

// Print pr and t2m for each minute up to first 2 hours and at daily boundary
// __global__ void debugMinuteForcings(const float *forc, int ns) {
//     int s = threadIdx.x;
//     if (blockIdx.x == 0 && s < 1) {             // do first 2 systems, s=0..1
//         size_t offset_pr  = 0;                   // pr is j=0 block
//         size_t offset_t2m = c_forc_nT[0] * ns;   // t2m is j=1 block

//         int max_min = 100;                       // first 100 minutes (2 h)
//         for (int t = 0; t <= max_min; ++t) {
//             // sample index in pr block:
//             size_t idx_pr  = offset_pr  + size_t(t)*ns + s;
//             // sample index in t2m block:
//             // compute minute‐index into daily samples:
//             double dt_t2m_min = c_forc_dt[1] * 60.0; // 24*60 = 1440
//             // sampleIdx = floor(t / dt_t2m_min) → 0 for t<1440, 1 for t≥1440
//             size_t sampleIdx_t2m = (t < (int)dt_t2m_min ? 0 : 1);
//             size_t idx_t2m = offset_t2m + sampleIdx_t2m*ns + s;

//             printf("sys=%d  t=%4d min → pr=%7.3f  t2m=%7.3f\n",
//                    s, t,
//                    forc[idx_pr],
//                    forc[idx_t2m]);
//         }

//         // also show at exactly 1440 min (1 day) and at 2880 min (2 days)
//         int day1 = int(c_forc_dt[1]*60.0);      // =1440
//         int day2 = day1 * 2;                   // =2880
//         for (int t : {day1, day2}) {
//             size_t idx_pr  = offset_pr  + size_t(t)*ns + s; 
//             size_t sampleIdx_t2m = (t < day1 ? 0 : (t < day2 ? 1 : 1));
//             size_t idx_t2m = offset_t2m + sampleIdx_t2m*ns + s;
//             printf("sys=%d  t=%4d min → pr=%7.3f  t2m=%7.3f\n",
//                    s, t,
//                    forc[idx_pr],
//                    forc[idx_t2m]);
//         }
//     }
// }

// ────────── In src/solver/rk45_kernel.cu, above testKernel ──────────

// __global__ void debugHolding(const float *forc, int ns) {
//     int sys = blockIdx.x * blockDim.x + threadIdx.x;
//     if (sys != 0) return;      // only print for system 0

//     // offsets into the big forcing array
//     size_t offset_pr  = 0;
//     size_t offset_t2m = size_t(c_forc_nT[0]) * ns;

//     // sampling intervals in minutes
//     // (c_forc_dt are in hours)
//     double dt_pr_min   = c_forc_dt[0] * 60.0;   // e.g. 1 h→60 min
//     double dt_t2m_min  = c_forc_dt[1] * 60.0;   // e.g. 24 h→1440 min

//     printf(" t   →  pr[idx_pr]   |  t2m[idx_t2m]\n");
//     for (int t = 0; t <= 180; ++t) {  // print first 3 hours = 180 min
//         // integer sample index for each forcing:
//         int idx_pr  = int(t / dt_pr_min);
//         int idx_t2m = int(t / dt_t2m_min);

//         // clamp to valid range
//         if (idx_pr  >= int(c_forc_nT[0])) idx_pr  = int(c_forc_nT[0]) - 1;
//         if (idx_t2m >= int(c_forc_nT[1])) idx_t2m = int(c_forc_nT[1]) - 1;

//         float pr_val  = forc[offset_pr  + size_t(idx_pr)  * ns + sys];
//         float t2m_val = forc[offset_t2m + size_t(idx_t2m) * ns + sys];

//         printf("t=%3d → pr[%2d]=%7.3f   |   t2m[%2d]=%7.3f\n",
//                t, idx_pr, pr_val,
//                idx_t2m, t2m_val);
//     }
// }

// ────────── End debugging forcings ──────────





// ───────── Minimal test kernel ─────────
// __global__ void testKernel() { /* nothing */ }

// ───────── Tiny kernel to verify one SpatialParams via printf ─────────
// __global__ void debugParams(const SpatialParams* sp) {
//     if (blockIdx.x == 0 && threadIdx.x == 0) {
//         printf("GPU sees: stream=%ld, Hu=%g, infil=%g, perco=%g\n",
//                sp[0].stream, sp[0].Hu, sp[0].infil, sp[0].perco);
//     }
// }

// ─────────────────────────────────────────────────────────────

// ───────── Debugging device-side full print of spatial params ─────────
// Print every stream's SpatialParams from the GPU.
// ─────────────────────────────────────────────────────────────────────────
// __global__ void debugAllParams(const SpatialParams* sp, int N) {
//     int sys = blockIdx.x * blockDim.x + threadIdx.x;
//     if (sys < N) {
//         printf("sys=%3d → stream=%10ld, Hu=%6.3f, infil=%6.3f, perco=%6.3f, L=%6.3f, A_h=%6.3f\n",
//                sys,
//                sp[sys].stream,
//                sp[sys].Hu,
//                sp[sys].infil,
//                sp[sys].perco,
//                sp[sys].L,
//                sp[sys].A_h
//         );
//     }
// }
// ───────── ended debugging ──────────────────────────────────────────────────────


// ────────────────────────────────────────────────
// Debug‐kernel: call your RHS once for one stream
// ────────────────────────────────────────────────
// __global__ void debugRHS(const SpatialParams* sp, int sys_id) {
//     // Only one thread actually prints
//     if (blockIdx.x == 0 && threadIdx.x == 0) {
//         // Header so you can spot this block in the log
//         printf("[DebugRHS] ==== BEGIN RHS debug for stream index %d ====\n",
//                sys_id);
//         // Print the stream ID and raw parameters
//         printf("[DebugRHS]   Stream ID = %ld\n", sp[sys_id].stream);
//         printf("[DebugRHS]   Parameters: Hu=%g, infil=%g, perco=%g, L=%g, A_h=%g\n",
//                sp[sys_id].Hu,
//                sp[sys_id].infil,
//                sp[sys_id].perco,
//                sp[sys_id].L,
//                sp[sys_id].A_h);

//         // Prepare a trivial y-vector and compute dydt
//         double y[Model204::N_EQ]   = {1.0, 1.0, 1.0, 1.0, 1.0};
//         double dydt[Model204::N_EQ];
//         Model204::rhs(0.0, y, dydt, Model204::N_EQ, sys_id, sp, d_forc_data, nForc);

//         // Print the resulting derivatives cleanly
//         printf("[DebugRHS]   dydt: ");
//         for (int i = 0; i < Model204::N_EQ; ++i) {
//             printf("%g ", dydt[i]);
//         }
//         printf("\n[DebugRHS] ====  END RHS debug  ====\n\n");
//     }
// }


__global__ void debugMinuteForcings(const float *forc, int ns) {
    int s = 0;  // only look at system 0
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        size_t offset_pr  = 0;
        size_t offset_t2m = c_forc_nT[0] * size_t(ns);
        int max_min = 180;  // first 3 hours

        printf(" t   →  pr   |  t2m\n");
        for (int t = 0; t <= max_min; ++t) {
            int idx_pr  = min(int(c_forc_nT[0]) - 1, int(t / (c_forc_dt[0] * 60.0)));
            int idx_t2m = min(int(c_forc_nT[1]) - 1, int(t / (c_forc_dt[1] * 60.0)));
            float pr   = forc[offset_pr  + size_t(idx_pr)  * ns + s];
            float t2m  = forc[offset_t2m + size_t(idx_t2m) * ns + s];
            printf("t=%3d → pr[%2d]=%7.3f | t2m[%2d]=%7.3f\n",
                   t, idx_pr, pr, idx_t2m, t2m);
        }
    }
}

// ----------------------------------------------------------------------
// Kernel to debug the way rhs reads F[0] (rain) and F[1] (temp),
// now also printing sys index and stream ID
// ----------------------------------------------------------------------
__global__ void debugRhsForcings(const SpatialParams* sp_ptr,
                                 const float*         F,
                                 int                  ns,
                                 int                  nForc)
{
    if (blockIdx.x==0 && threadIdx.x==0) {
        // pick system 0 and time t=0
        int sys = 0;
        double t = 0.0;

        // grab that stream’s unique ID from your SpatialParams
        long stream_id = sp_ptr[sys].stream;

        printf(">>> debugRhsForcings <<<\n");
        printf("  sys index = %d, stream ID = %ld\n", sys, stream_id);
        printf("  nForc=%d, dt[0]=%g, dt[1]=%g, nT[0]=%llu, nT[1]=%llu\n",
               nForc,
               c_forc_dt[0], c_forc_dt[1],
               (unsigned long long)c_forc_nT[0],
               (unsigned long long)c_forc_nT[1]);

        // Compute the offsets into the big F array
        size_t pr_block   = 0;                    
        size_t t2m_block  = c_forc_nT[0] * size_t(ns);

        float rain0 = F[pr_block + sys];
        float temp0 = (nForc>1 ? F[t2m_block + sys] : 0.0f);

        printf("  GPU sees rainfall= %f, temperature= %f\n", rain0, temp0);

        // Now fire rhs and print dydt
        double y[Model204::N_EQ] = {0.01,0.1,0,0,0.01,1,1};   // use your real y0 here
        double dydt[Model204::N_EQ];
        Model204::rhs(t, y, dydt,
                      Model204::N_EQ,
                      sys,
                      sp_ptr,
                      F,
                      nForc);

        printf("  dydt =");
        for (int i = 0; i < Model204::N_EQ; ++i)
            printf(" %g", dydt[i]);
        printf("\n-----------------------------\n");
    }
}


// Debugging the initial y[0..6] for sys0
__global__ void debugDeviceY0(const double* d_y0_all, int ns) {
    // only need one thread
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf(">>> GPU initial y[0..6] for sys0 <<<\n");
        // d_y0_all is laid out [ y₀_sys0…y₆_sys0, y₀_sys1… , … ]
        for (int i = 0; i < Model204::N_EQ; ++i) {
            printf("  y[%d] = %g\n", i, d_y0_all[i]);
        }
        printf("-----------------------------\n");
    }
}



// above main():
// __global__ void checkForcingsOnDevice(const float* forc,
//                                       int ns,
//                                       int nf)    // <— pass nForc in!
// {
//     if (blockIdx.x==0 && threadIdx.x==0) {
//         printf(">>> DEVICE FORCINGS CHECK <<<\n");
//         printf("  nForc = %d\n", nf);
//         printf("  dt[0]=%g, dt[1]=%g\n", c_forc_dt[0], c_forc_dt[1]);
//         printf("  nT[0]=%llu, nT[1]=%llu\n",
//                (unsigned long long)c_forc_nT[0],
//                (unsigned long long)c_forc_nT[1]);

//         for (int f = 0; f < nf; ++f) {
//             // offset of forcing f at t=0
//             size_t offs = (f==0
//                            ? 0
//                            : c_forc_nT[0] * size_t(ns));
//             float v0 = forc[offs + /*sys=*/0];
//             printf("  forcing[%d], sys0, t=0 → %f\n", f, v0);
//         }
//         printf("-------------------------\n");
//     }
// }

__global__ void checkForcingsOnDevice(const float* forc,
                                      int ns,
                                      int nf)
{
    if (blockIdx.x==0 && threadIdx.x==0) {
        printf(">>> DEVICE FORCINGS MULTI‐STEP CHECK <<<\n");
        for (int f = 0; f < nf; ++f) {
            size_t base = (f==0? 0 : c_forc_nT[0] * size_t(ns));
            printf(" Forcing[%d] (dt=%g, nT=%llu):\n",
                   f, c_forc_dt[f],
                   (unsigned long long)c_forc_nT[f]);
            int maxPrint = min((int)c_forc_nT[f], 48);
            for (int t = 0; t < maxPrint; ++t) {
                size_t offs = base + size_t(t) * ns;
                printf("    t=%2d → %f\n", t, forc[offs + /*sys=*/0]);
            }
        }
        printf("----------------------------------------\n");
    }
}




// ───────── ended debugging ─────────────────────────────────────────────



// ────────── Main function to run the RK45 solver on GPU ───────────────────

int main(int argc, char** argv) {
    
    // Initialize MPI 
    MPI_Init(&argc, &argv);
    // Take the user’s time‐origin argument
    std::string time_origin = (argc > 1
                               ? argv[1]
                               : "2019-01-01T00:00:00Z");
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get gpu count
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);

    // Rank zero splits the spatial params into sections (!!! need to change to spatial chunking !!!)  
    if(rank==0){

        //load in all links
        // auto spatialParams = loadSpatialParams("../data/small_example_params.csv");        
        auto spatialParams = loadSpatialParams("../data/large_example_params.csv");  

        // Calculate chunk size for even splitting
        int totalRanks = size - 1; // Exclude rank 0
        int totalRows = spatialParams.size();
        int baseChunk = totalRows / totalRanks;
        int remainder = totalRows % totalRanks;

        int start = 0;

        std::cout << "Total rows: " << totalRows << std::endl;
        std::cout << "Total ranks: " << totalRanks << std::endl;

        for (int r = 1; r <= totalRanks; ++r) {
            // Calculate actual chunk size for this rank
            int chunkSize = baseChunk + (r <= remainder ? 1 : 0);
            int end = start + chunkSize;

            // Ensure we don't go out of bounds
            if (end > totalRows) end = totalRows;

            // Slice the vector
            std::vector<SpatialParams> subset(spatialParams.begin() + start, spatialParams.begin() + end);
            int count = subset.size();

            std::cout << "Sending " << count << " SpatialParams to rank " << r << std::endl;

            // Send count first
            MPI_Send(&count, 1, MPI_INT, r, 0, MPI_COMM_WORLD);

            // Send the raw data
            MPI_Send(subset.data(), count * sizeof(SpatialParams), MPI_BYTE, r, 1, MPI_COMM_WORLD);

            // Move to next chunk
            start = end;
        }
    
    }
    if(rank >= 1 && rank < size) { //!!!! NEED TO CHANGE size to nGPUs

        // ────────── Set the GPU device for this rank ──────────
        
        if(gpuCount < rank){
            cudaSetDevice(0);
        }else{
            // Assign GPU i as rank - 1
            cudaSetDevice(rank - 1);
        }

        // ───────── Print GPU properties ─────────
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::printf("Running on GPU %s (SM %d.%d)\n",
                    prop.name, prop.major, prop.minor);

        // Print UUID in standard format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        std::printf("GPU UUID: ");
        for (int i = 0; i < 16; ++i) {
            std::printf("%02x", (unsigned char)prop.uuid.bytes[i]);
            if (i == 3 || i == 5 || i == 7 || i == 9)
                std::printf("-");
        }
        std::printf("\n");
        // _____________ end checking GPU properties _______________

        // ───────── Test that even a trivial kernel will launch ─────────
        // testKernel<<<1,1>>>();
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     std::fprintf(stderr, "testKernel launch failed: %s\n",
        //                 cudaGetErrorString(err));
        // } else {
        //     std::puts("testKernel launch: OK");
        // }
        // cudaDeviceSynchronize();



        using namespace rk45_api;

        // ───────── 0) load per‐stream spatial parameters ─────────
        // auto spatialParams = loadSpatialParams("../data/small_test.csv");//10 links

        int count;

        // Receive the count first
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Allocate buffer
        std::vector<SpatialParams> receivedSubset(count);

        // Receive the actual data
        MPI_Recv(receivedSubset.data(), count * sizeof(SpatialParams), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto spatialParams = receivedSubset;
        
        // Query one of the NetCDF files for its spatial dimensions
        NetCDFLoader prLoader("../data/pr_hourly_era5land_2019.nc", "pr");
        size_t lat_size = prLoader.getLatSize();
        size_t lon_size = prLoader.getLonSize();

        // build a vector of Stream<Model204>, using a common y0
        // std::array<double, Model204::N_EQ> y0_common = {0.01, 0.1, 0.0, 0.0, 0.01, 1, 1};// 7 equation system
        std::array<double, Model204::N_EQ> y0_common = {0.01, 0.1, 0.0, 0.0, 0.01, 1, 1, 0, 0}; // 9 equation system
        // add an 2d array with link ID and state variables , final csv 
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
        cudaError_t err = cudaMalloc(&d_sp, byteCount);
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

        // // ─────── DebugRHS: verify Model204::rhs uses the right parameters ───────
        // {
        //     int target_sys = 0;                // choose the stream index to inspect
        //     debugRHS<<<1,1>>>(d_sp, target_sys);
        //     cudaDeviceSynchronize();
        // }



        // Verify via checkDevParamsKernel204
        // checkDevParamsKernel204<<<1,1>>>();
        // err = cudaDeviceSynchronize();
        // if (err != cudaSuccess) {
        //     std::fprintf(stderr, "checkDevParamsKernel204 failed: %s\n",
        //                 cudaGetErrorString(err));
        //     return 1;
        // }

        // Launch the tiny debugParams kernel
        // debugParams<<<1,32>>>(d_sp);
        // err = cudaDeviceSynchronize();
        // if (err != cudaSuccess) {
        //     std::fprintf(stderr, "debugParams kernel failed: %s\n", cudaGetErrorString(err));
        // } else {
        //     std::puts("debugParams kernel ran successfully!!!");
        // }
        // ───────── end debugging spatial params ────────────────────────────────────


        // ─── Build streamPoint lookup ─────────────────────────────────────────
        // LookupMapper lm("../data/small_example_pr_lookup.csv");
        // // LookupMapper lm("../data/large_example_pr_lookup.csv");
        // if (!lm.load()) {
        //     std::cerr << "Lookup load failed\n";
        //     return 1;
        // }
        // // one flat index per system
        // std::vector<size_t> streamPoint(num_systems);
        // for (int s = 0; s < num_systems; ++s) {
        //     auto [lat, lon] = lm.getLatLon(streams[s].id);
        //     streamPoint[s] = lat * lon_size + lon;
        // }



            // ─── Loop over 2019 in n-day chunks ───
            constexpr int DAYS_PER_CHUNK = 10; //automate this !!!
            constexpr int TOTAL_DAYS     = 365;
            // starting date: 2019-01-01
            int chunkStartYear = 2019, chunkStartMon = 1, chunkStartDay = 1;

            for (int dayOffset = 0; dayOffset < TOTAL_DAYS; dayOffset += DAYS_PER_CHUNK) {
                // clamp last chunk
                int daysThisChunk = std::min(DAYS_PER_CHUNK, TOTAL_DAYS - dayOffset);

                // compute chunkStart = YYYYMMDD
                std::string sDate = formatDate(chunkStartYear, chunkStartMon, chunkStartDay);

                // compute chunk end date (advance by daysThisChunk-1)
                int endY = chunkStartYear, endM = chunkStartMon, endD = chunkStartDay;
                advanceDate(endY, endM, endD, daysThisChunk - 1);
                std::string eDate = formatDate(endY, endM, endD); //might need to tab here *

                // <<< ADD HERE >>> debug‐print chunk header
                std::cout 
                << "[DEBUG] chunk #" << (dayOffset / DAYS_PER_CHUNK + 1)
                << "  dayOffset=" << dayOffset
                << "  daysThisChunk=" << daysThisChunk
                << "  date range=" << sDate << "→" << eDate
                << "\n";


                // ─── Adding forcings  ─────────────────────────────────
                struct NCForcing {
                    std::string path, var;
                    double      dt;    // hours per time step
                };
                std::vector<NCForcing> ncForcings = {
                    {"../data/pr_hourly_era5land_2019.nc",    "pr",  1.0},
                    {"../data/t2m_daily_avg_era5land_2019.nc","t2m", 24.0}
                };
                int nForc = int(ncForcings.size());
                int host_nForc = nForc;  

                // ── Dynamic lookup & per‐forcing index vectors ──→
                // lookup CSVs path should match what the user gives in .yaml
                // Comment out when testing for small example
                // std::vector<std::string> lookup_csv = {
                //     "../data/small_example_pr_lookup.csv",
                //     "../data/small_example_t2m_lookup.csv"
                // };

                std::vector<std::string> lookup_csv = {
                    "../data/large_example_pr_lookup.csv",
                    "../data/large_example_t2m_lookup.csv"
                };

                std::vector<LookupMapper> lookup_mappers;
                lookup_mappers.reserve(nForc);
                std::vector<size_t> lat_sizes(nForc), lon_sizes(nForc);
                for (int j = 0; j < nForc; ++j) {
                    lookup_mappers.emplace_back(lookup_csv[j]);
                    if (!lookup_mappers.back().load()) {
                        std::cerr << "Lookup load failed for " << lookup_csv[j] << "\n";
                        return 1;
                    }
                    // still grab lat/lon sizes from NetCDF file
                    NetCDFLoader tmp(ncForcings[j].path, ncForcings[j].var);
                    lat_sizes[j] = tmp.getLatSize();
                    lon_sizes[j] = tmp.getLonSize();
                }

                // now build your streamPoint_per_forcing as before:
                std::vector<std::vector<size_t>> streamPoint_per_forcing(
                    nForc, std::vector<size_t>(num_systems)
                );
                for (int j = 0; j < nForc; ++j) {
                    for (int s = 0; s < num_systems; ++s) {
                        auto [lat,lon] = lookup_mappers[j].getLatLon(streams[s].id);
                        streamPoint_per_forcing[j][s] = lat * lon_sizes[j] + lon;
                    }
                }

                // ── INSERT END ──



                std::vector<float>  h_forc_data;
                std::vector<double> h_forc_dt;
                std::vector<size_t> h_forc_nT;
                h_forc_dt .reserve(nForc);
                h_forc_nT .reserve(nForc);
                // h_forc_data.reserve(nForc * num_systems * daysThisChunk * 24); // reserve enough space for all forcings

                // ── new 64-bit safe reservation: estimate exact number of floats to push ── 
                size_t estimated = 0;
                for (int j = 0; j < nForc; ++j) {
                    // how many steps this forcing in the chunk?
                    double dt_hr = ncForcings[j].dt;
                    size_t steps2d = static_cast<size_t>(
                        std::round(daysThisChunk * 24.0 / dt_hr)
                    );
                    // account for the fact we clamp to fullTime inside the real loop,
                    // but this estimate is always ≥ actual, so it’s safe
                    estimated += steps2d * size_t(num_systems);
                }
                h_forc_data.reserve(estimated);
                // ── ended new reservation ──
            


                // for (auto &fm : ncForcings) { //old code
                for (int j = 0; j < nForc; ++j) {
                    auto &fm = ncForcings[j];
                    NetCDFLoader loader(fm.path, fm.var);
                    size_t fullTime = loader.getTimeSize();

                    // how many steps in 2 days?
                    // size_t steps2d = size_t(std::round(daysToLoad * 24.0 / fm.dt));
                    // steps2d = std::min(fullTime, steps2d); !!!


                    // auto raw = loader.loadTimeChunk(0, steps2d);
                    // // store dt in hrs?
                    // h_forc_dt.push_back(fm.dt);
                    // h_forc_nT.push_back(steps2d); !!!

                    // how many samples we really want (e.g. 100 days × 24 h / dt)
                    //size_t fullTime   = loader.getTimeSize();
                    size_t steps2d    = static_cast<size_t>(
                        std::round(daysThisChunk * 24.0 / fm.dt)
                    );

                    if (steps2d > fullTime) {
                        std::cerr
                            << "WARNING: requested " << daysThisChunk
                            << " days (" << steps2d << " steps) but only "
                            << fullTime << " available in “" << fm.path << "”; "
                            << "clamping to " << fullTime << " steps.\n";
                        steps2d = fullTime;  // clamp to available time steps
                    }


                    // — diagnostic print —
                    size_t gridPts = loader.getLatSize() * loader.getLonSize();
                    size_t totalFloats = steps2d * gridPts;
                    std::cout 
                        << "[MEMCHK] Loading forcing chunk for “" << fm.var << "”: \n"
                        << "   steps2d = "  << steps2d      << "\n"
                        << "   gridPts = "  << gridPts      << "\n"
                        << "   totalFloats = " << totalFloats << "  (" 
                            << (totalFloats * sizeof(float) / (1024.0*1024.0)) 
                            << " MiB)\n";
                    

                    // now load exactly what you asked for
                    // auto raw = loader.loadTimeChunk(0, steps2d);
                    double dt_hr = fm.dt; 
                    // how many time‐steps to skip at the front?
                    size_t startIdx = size_t(std::round(dayOffset * 24.0 / dt_hr));

                    // clamp if you run off the end
                    startIdx = std::min(startIdx, fullTime);

                    // DEBUG: print each forcing’s offsets
                    std::cout 
                    << "[DEBUG]   [" << fm.var << "] dt=" << dt_hr
                    << "  startIdx=" << startIdx
                    << "  steps2d="  << steps2d
                    << "  fullTime=" << fullTime
                    << "\n";
                    
                    auto raw = loader.loadTimeChunk(startIdx, steps2d);

                    h_forc_dt.push_back(fm.dt);
                    h_forc_nT.push_back(steps2d);

                    // ─────────────────── DEBUG: force‐loading ───────────────────────────
                    // /// DEBUG: forcing‐loading summary
                    // std::cout << "Loaded nForc=" << nForc << " forcings:\n";
                    // for (int j = 0; j < nForc; ++j) {
                    //     std::cout
                    //     << "  forcing[" << j         << "]  "
                    //     << "dt_min="     << h_forc_dt[j]
                    //     << "  samples="  << h_forc_nT[j]
                    //     << "\n";
                    // }
                    // ───────────────────────────────────────────────────────────────

            


                    

                    // ────────────────────────────────────────────────────────────────

                    // get (approximate) calendar times for each step if your loader supports it:
                    // auto times = loader.getTimeValues(0, steps2d);  // e.g. returns vector<double> of “hours since …”

                    // std::cout << "Forcing #" << i
                    //         << " (“" << fm.var << "”), dt=" << fm.dt << " h, nT="<< steps2d << "\n";
                    // for (size_t t = 0; t < std::min<size_t>(5, steps2d); ++t) {
                    //     // just print the first 5 time‐levels for sanity
                    //     std::cout << "   t="<< t
                    //             << " (≈ "<< times[t] <<" h) → [first system] = "
                    //             << raw.get()[t * gridPts + streamPoint[0]]
                    //             << "\n";
                    // }
                    // ────────────────────────────────────────────────────────────────

                    // size_t gridPts = loader.getLatSize() * loader.getLonSize();
                    // float *basePtr = raw.get();

                    // for (size_t t = 0; t < steps2d; ++t) {
                    //     float *slice = basePtr + t * gridPts;
                    //     for (int s = 0; s < num_systems; ++s) {
                    //         // grab the value into 'val' BEFORE you print it:
                    //         // float val = slice[ streamPoint[s] ];
                    //         h_forc_data.push_back(slice[ streamPoint[s] ]);

                    //         // — DEBUG PRINT —
                    //         // prints: t (time‐step), s (stream‐index), 
                    //         //         streams[s].id (unique ID), 
                    //         //         streamPoint[s] (flat cell index), 
                    //         //         val (the value you're about to push)
                    //         // only print for the one stream and first 5 timesteps:
                    //         // if (streams[s].id == 420555774 && t < 5) {
                    //         //     std::cout
                    //         //     << "t="       << std::setw(2) << t
                    //         //     << "  s="     << std::setw(4) << s
                    //         //     << "  id="    << std::setw(10) << streams[s].id
                    //         //     << "  pt="    << std::setw(6) << streamPoint[s]
                    //         //     << "  val="   << std::fixed << std::setprecision(6) << val
                    //         //     << "\n";
                    //         // }

                    //         // h_forc_data.push_back(val);
                    //     }
                    // }



                    // NEW:
                    float *basePtr = raw.get();
                    // size_t gridPts = lat_sizes[j] * lon_sizes[j]; //already defined above
                    for (size_t t = 0; t < steps2d; ++t) {
                    float *slice = basePtr + t * gridPts;
                    for (int s = 0; s < num_systems; ++s) {
                        h_forc_data.push_back(
                        slice[ streamPoint_per_forcing[j][s] ]
                        );
                    }
                    }

                }
                
                // C) Upload to device
                float* d_forc_ptr = nullptr;
                // cudaMalloc(&d_forc_ptr, sizeof(float) * h_forc_data.size());
                // cudaMemcpy(d_forc_ptr,
                //         h_forc_data.data(),
                //         sizeof(float) * h_forc_data.size(),
                //         cudaMemcpyHostToDevice);

                // cudaError_t err;
                size_t bytes = sizeof(float) * h_forc_data.size();

                // 1) allocate
                err = cudaMalloc(&d_forc_ptr, bytes);
                if (err != cudaSuccess) {
                    std::cerr << "[CUDA ERROR] cudaMalloc(d_forc_ptr, " 
                            << (bytes/(1024.0*1024.0)) << " MiB) failed: "
                            << cudaGetErrorString(err) << "\n";
                    std::exit(1);
                }

                // 2) copy
                err = cudaMemcpy(d_forc_ptr,
                                h_forc_data.data(),
                                bytes,
                                cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "[CUDA ERROR] cudaMemcpy ("
                            << (bytes/(1024.0*1024.0)) << " MiB) failed: "
                            << cudaGetErrorString(err) << "\n";
                    std::exit(1);
                }


                // D) Push dt, nT, pointer and count into device symbols
                {

                    
                    // copy forcing time‐step sizes (hours)
                    cudaMemcpyToSymbol(c_forc_dt, h_forc_dt.data(),
                                    sizeof(double) * nForc);
                    // copy number of samples per forcing
                    cudaMemcpyToSymbol(c_forc_nT, h_forc_nT.data(),
                                    sizeof(size_t) * nForc);

                    // copy pointer to big forcing array
                    // cudaMemcpyToSymbol(d_forc_data, &d_forc_ptr,
                    //                 sizeof(d_forc_ptr));
                    // copy the count
                    // cudaMemcpyToSymbol(nForc,       &nForc,
                    //                 sizeof(nForc));
                }



                // ────────── End uploading forcings ──────────



                // ───────── Define time span (first‐2‐day test) ─────────
                // constexpr double daysToSim = daysToLoad;
                // const double t0 = 0.0;
                // const double tf = daysToSim * 24.0 * 60.0;

                // ───────── Define time span for this chunk ─────────
                // double daysToSim = double(daysThisChunk);
                // const double t0 = 0.0;
                // const double tf = daysToSim * 24.0 * 60.0;
                // compute absolute minutes of chunk start
                double baseMinutes = double(dayOffset) * 24.0 * 60.0;
                double t0 = baseMinutes;
                // run until end of this chunk in absolute minutes
                double tf = baseMinutes + daysThisChunk * 24.0 * 60.0;



                // ───────── Compute SciPy‐style initial step and upload devParams ─────────
                {
                    int N_EQ = Model204::N_EQ;
                    const SpatialParams* host_sp_ptr = spatialParams.data();
                    std::vector<double> y0_cpu(N_EQ, 0.0), f0_cpu(N_EQ), scale(N_EQ);
                    Model204::rhs(t0, y0_cpu.data(), f0_cpu.data(), N_EQ, 0, host_sp_ptr, h_forc_data.data(), nForc);

                    double rtol = 1e-6, atol = 1e-9;
                    // double rtol = 1e-2, atol = 1e-4; // to match old code tolerances
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
            

                // // ───────── Define query times (first 2 days, hourly) ─────────
                // std::vector<double> h_query_times;
                // for (double t = t0; t <= tf; t += 60.0) {
                //     h_query_times.push_back(t);
                // }

                // ───────── Define query times (chunk start + hourly) ─────────
                // std::vector<double> h_query_times;
                // // offset (in minutes) of this chunk’s 00:00
                // double baseMinutes = double(dayOffset) * 24 * 60;
                // // every 60 min over this chunk
                // for (double dt = 0.0; dt <= double(daysThisChunk) * 24 * 60; dt += 60.0) {
                //     h_query_times.push_back(baseMinutes + dt);
                // }
                std::vector<double> h_query_times;
                // then build query_times exactly as you have:
                for (double m = baseMinutes; m <= tf; m += 60.0) {
                    h_query_times.push_back(m);
                }
                
                int num_queries = int(h_query_times.size());


                // ——————————————— Print dense‐output size ———————————————
                std::uint64_t total_dense = std::uint64_t(num_queries)
                                        * std::uint64_t(num_systems)
                                        * std::uint64_t(Model204::N_EQ);
                std::uint64_t bytes_dense = total_dense * sizeof(double);
                double gib = double(bytes_dense) / (1024.0*1024.0*1024.0);

                // enforce a hard limit of 2e9 doubles (14.9 GiB)
                constexpr std::uint64_t MAX_ELEMENTS = 2000000000ULL;
                constexpr double MAX_BYTES    = double(MAX_ELEMENTS) * sizeof(double);
                constexpr double MAX_GIB      = MAX_BYTES / (1024.0*1024.0*1024.0);

                if (total_dense > MAX_ELEMENTS) {
                    std::cerr 
                        << "ERROR: requested dense‐output size (" << total_dense << " doubles, ≈ "
                        << gib << " GiB) exceeds the maximum of "
                        << MAX_ELEMENTS << " doubles (≈ " 
                        << MAX_GIB << " GiB)\n";
                    std::exit(1);
                }

                std::cout << "Program will allocate "
                        << total_dense << " doubles (≈ "
                        << gib << " GiB) for dense output\n";
                // ————————————————————————————————————————————————————————

                

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


                // Debug print of the very first system’s initial state
                // debugDeviceY0<<<1,1>>>(d_y0_all, ns);
                // cudaDeviceSynchronize();

                // checking radau gets launched 
                // after your rk45_then_radau<<<…>>> and cudaDeviceSynchronize():
                // int h_stiff_idx = 0;                          // pick system 0
                // int* d_stiff_idx;
                // cudaMalloc(&d_stiff_idx, sizeof(int));
                // cudaMemcpy(d_stiff_idx, &h_stiff_idx, sizeof(int), cudaMemcpyHostToDevice);

                // radau_kernel_multi<Model204><<<1,1>>>(
                //     d_y0_all, d_y_final_all,
                //     d_query_times, d_dense_all,
                //     num_systems, num_queries,
                //     t0, tf,
                //     d_sp,
                //     d_stiff_idx,           // 1 stiff system
                //     1,                     // n_stiff = 1
                //     d_forc_ptr,            // your forcing pointer
                //     nForc                  // your forcing count
                // );
                // cudaDeviceSynchronize();
                // cudaFree(d_stiff_idx);



                // ───────── Diagnostic launch + checks ─────────
                // ──────────Old way to set threads and blocks───────────
                // const int THREADS_PER_BLOCK = 128;
                // int numBlocks = (ns + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                // dim3 blocks(1,numBlocks,numBlocks), threads(1,4,4);

                // std::printf("Launching kernel with %d blocks, %d threads/block (ns=%d)\n",
                //             numBlocks, THREADS_PER_BLOCK, ns);

                // err = cudaGetLastError();
                // if (err != cudaSuccess) {
                //     std::fprintf(stderr, "Pre-launch cudaGetLastError: %s\n",
                //                 cudaGetErrorString(err));
                // }

                // ────────── end Old way to set threads and blocks───────────

                // ────────── New way to set threads and blocks ──────────

                int minGridSize = 0, bestBlockSize = 0;

                // Setting CUDA occupancy parameters
                cudaOccupancyMaxPotentialBlockSize(
                    &minGridSize,
                    &bestBlockSize,
                    rk45_then_radau_multi<Model204>,
                    0,  // shared memory per block (set if needed)
                    0   // max threads per block (0 = let CUDA decide)
                );

                int numBlocks = (ns + bestBlockSize - 1) / bestBlockSize;

                std::printf("Occupancy-optimized kernel launch:\n  blocks=%d  threads/block=%d\n", numBlocks, bestBlockSize);
                // ────────── end New way to set threads and blocks ──────────

                // —─────────────────────── launching error kernel
                cudaMemcpyToSymbol(d_forc_data, &d_forc_ptr, sizeof(d_forc_ptr));
                cudaMemcpyToSymbol(nForc, &nForc, sizeof(nForc));

                // push forcing‐pointer into device symbol
                // CUDA_CHECK(cudaMemcpyToSymbol(d_forc_data, &d_forc_ptr, sizeof(d_forc_ptr)));

                // push nForc into device symbol
                // CUDA_CHECK(cudaMemcpyToSymbol(nForc, &nForc, sizeof(nForc)));


                // debugMinuteForcings<<<1, 128>>>(d_forc_ptr, ns);
                // cudaDeviceSynchronize();

                // ns = number of systems, nForc = your host int
                // checkForcingsOnDevice<<<1,1>>>(d_forc_ptr, ns, nForc);
                // cudaDeviceSynchronize();


                // Launch the debug‐rhs kernel
                // debugRhsForcings<<<1,1>>>(d_sp, d_forc_ptr, ns, nForc);
                // cudaDeviceSynchronize();

                // ────────── Debug: print loaded forcings ──────────
                std::cout 
                << "[DEBUG] Loaded nForc=" << nForc 
                << " forcings:\n";
                for (int j = 0; j < nForc; ++j) {
                    double dt_hr   = h_forc_dt[j];
                    double dt_min  = dt_hr * 60.0;
                    size_t samples = h_forc_nT[j];
                    std::cout
                    << "    forcing["<< j <<"]  dt_hr="<< dt_hr
                    << "  dt_min="<< dt_min
                    << "  samples="<< samples
                    << "\n";
                }
                std::cout << std::flush;



                // ────────── Timing solver kernel ──────────
                cudaEvent_t start_CUDA, stop_CUDA;
                cudaEventCreate(&start_CUDA);
                cudaEventCreate(&stop_CUDA);

                cudaEventRecord(start_CUDA, 0);

                // ───────── Launch the main solver kernel ─────────
                // rk45_then_radau_multi<Model204><<<blocks,threads>>>(
                //     d_y0_all, d_y_final_all,
                //     d_query_times, d_dense_all,
                //     ns, nq,
                //     t0, tf,
                //     d_sp,      // spatial params
                //     d_stiff,   // stiffness flags
                //     d_forc_ptr,  // forcing data pointer
                //     nForc         // number of forcings
                // );

                rk45_then_radau_multi<Model204><<<numBlocks, bestBlockSize>>>(
                    d_y0_all, d_y_final_all,
                    d_query_times, d_dense_all,
                    ns, nq,
                    t0, tf,
                    d_sp,      // spatial params
                    d_stiff,   // stiffness flags
                    d_forc_ptr,  // forcing data pointer
                    nForc         // number of forcings
                );

                cudaEventRecord(stop_CUDA, 0);
                cudaEventSynchronize(stop_CUDA);

                float ms;
                cudaEventElapsedTime(&ms, start_CUDA, stop_CUDA);
                std::printf("Solver kernel took %.6f seconds\n", ms * 1e-3f);

                

                // err = cudaGetLastError();
                // if (err != cudaSuccess) {
                //     std::fprintf(stderr, "Kernel launch failed: %s\n",
                //                 cudaGetErrorString(err));
                //     return 1;
                // } else {
                //     std::puts("Kernel launch: OK");
                // }

                // err = cudaDeviceSynchronize();
                // if (err != cudaSuccess) {
                //     std::fprintf(stderr, "Kernel execution failed: %s\n",
                //                 cudaGetErrorString(err));
                //     return 1;
                // } else {
                //     std::puts("Kernel execution: OK");
                // }
                // ————————————————————————————————

                // ───────── Retrieve results & free buffers ─────────

                // if(rank)
                auto [h_y_final, h_dense] = retrieve_and_free<Model204>(
                    d_y0_all, d_y_final_all,
                    d_query_times, d_dense_all, d_stiff,
                    ns, nq,
                    t0, tf, d_sp
                );

                CUDA_CHECK(cudaFree(d_forc_ptr));

                // ───────── Carry the final state over into the next chunk ─────────
                for (int s = 0; s < num_systems; ++s) {
                for (int i = 0; i < Model204::N_EQ; ++i) {
                    streams[s].y0[i] = h_y_final[s*Model204::N_EQ + i];
                }
                }
                



                // ───────── Debug: print first few timesteps for system 0 ─────────
                // {
                //     int sys = 0;
                //     std::cout << "=== First few steps for sys="<< sys
                //             << " (stream ID="<< streams[sys].id << ")\n";
                //     int nprint = std::min(5, num_queries);
                //     for(int q=0; q < nprint; ++q) {
                //         double t = h_query_times[q];
                //         std::cout << "  t["<< q << "] = " << t << "  →  ";
                //         for(int eq = 0; eq < Model204::N_EQ; ++eq) {
                //             int idx = (sys * num_queries + q) * Model204::N_EQ + eq;
                //             std::cout << "y"<< eq <<"="<< h_dense[idx] << "  ";
                //         }
                //         std::cout << "\n";
                //     }
                //     std::cout << std::endl;
                // }

                // // print the *true* initial y0 for sys0
                // {
                // int sys = 0;
                // std::cout << "Initial y0 for sys="<< sys
                //             << " (stream ID="<< streams[sys].id << "): ";
                // for (int eq = 0; eq < Model204::N_EQ; ++eq) {
                //     std::cout << streams[sys].y0[eq] << "  ";
                // }
                // std::cout << "\n";
                // }


                // // ───────── Write final.csv ─────────
                // {
                //     std::ofstream final_file("final_204_a.csv");
                //     final_file << "h_snow";
                //     for (int i = 1; i < Model204::N_EQ; ++i) {
                //         final_file << ",var" << i;
                //     }
                //     final_file << "\n";
                //     for (int s = 0; s < num_systems; ++s) {
                //         for (int i = 0; i < Model204::N_EQ; ++i) {
                //             final_file << h_y_final[s * Model204::N_EQ + i];
                //             if (i + 1 < Model204::N_EQ) final_file << ",";
                //         }
                //         final_file << "\n";
                //     }
                // }

                // // ───────── Write dense.csv ─────────
                // {
                //     std::ofstream dense_file("dense_204_a.csv");
                //     dense_file << "time";
                //     for (int s = 0; s < num_systems; ++s) {
                //         for (int i = 0; i < Model204::N_EQ; ++i) {
                //             dense_file << ",var" << i << "_sys" << s;
                //         }
                //     }
                //     dense_file << "\n";
                //     for (int q = 0; q < num_queries; ++q) {
                //         dense_file << std::fixed << std::setprecision(8)
                //                 << h_query_times[q];
                //         for (int s = 0; s < num_systems; ++s) {
                //             for (int i = 0; i < Model204::N_EQ; ++i) {
                //                 int idx = (s * num_queries + q) * Model204::N_EQ + i;
                //                 dense_file << "," << std::setprecision(9)
                //                         << h_dense[idx];
                //             }
                //         }
                //         dense_file << "\n";
                //     }
                // }

                // // ───────── Print a quick summary ─────────
                // std::printf("Final states at t = %.1f:\n", tf);
                // for (int s = 0; s < num_systems; ++s) {
                //     std::printf(" System %d:", s);
                //     for (int i = 0; i < Model204::N_EQ; ++i) {
                //         std::printf(" y%d=%.6f", i, h_y_final[s * Model204::N_EQ + i]);
                //     }
                //     std::printf("\n");
                // }

                // ———————————————————————————————— 
                // // ───────── Write to netcdf  ─────────

                // !!!! NEED TO CHANGE TO ACCESS ACTUAL ID AND STATE INDEXES !!!.    
                int N_EQ = Model204::N_EQ;

                // std::vector<int> linkid_vals(num_systems);
                // for (int s = 0; s < num_systems; ++s) linkid_vals[s] = s;

                // Build 32-bit system IDs (will truncate to lower 32 bits)
                std::vector<uint32_t> linkid_vals(num_systems);
                for (int s = 0; s < num_systems; ++s) {
                    linkid_vals[s] = streams[s].id;  // full 64-bit stream ID
                }


                std::vector<int> state_vals(N_EQ);
                for (int v = 0; v < N_EQ; ++v) state_vals[v] = v;

                // ───────── Netcdf file attributes (will be defined in yaml) ───────── 
                std::string final_filename = "/scratch/gpfs/mb6477/Tiger_HLM_GPU/outputs/final_large_"  + sDate + "_" + eDate + ".nc";
                std::string dense_filename = "/scratch/gpfs/mb6477/Tiger_HLM_GPU/outputs/dense_large_"  + sDate + "_" + eDate + ".nc";
                std::string runoff_filename = "/scratch/gpfs/mb6477/Tiger_HLM_GPU/outputs/runoff_large_" + sDate + "_" + eDate + ".nc";
                std::string selected_filename    = "/scratch/gpfs/mb6477/Tiger_HLM_GPU/outputs/selected_large_" + sDate + "_" + eDate + ".nc";

                int compression_level = 0;

                // ───────── Write only the final time step (2D output) ─────────
                std::cout << "Writing FINAL → "  << final_filename  << "\n";
                write_final_netcdf(final_filename,
                                h_y_final.data(),
                                linkid_vals.data(),
                                state_vals.data(),
                                num_systems,
                                N_EQ,
                                compression_level);

                // ───────── Write the dense output (3D output) ─────────
                std::cout << "Writing DENSE → "  << dense_filename  << "\n";
                auto start = std::chrono::high_resolution_clock::now();
                write_dense_netcdf(dense_filename,
                                h_dense.data(),
                                h_query_times.data(),
                                linkid_vals.data(),
                                state_vals.data(),
                                num_queries,
                                num_systems,
                                N_EQ,
                                time_origin,
                                compression_level);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "Writing dense output took " << elapsed.count() << " seconds\n";

                // ───────── Write runoff data (2D output) ─────────
                std::cout << "Writing RUNOFF → " << runoff_filename << "\n";
                // Write only surface + total runoff
                auto start_ro = std::chrono::high_resolution_clock::now();
                write_runoff_dense_netcdf(runoff_filename,
                                        h_dense.data(),
                                        h_query_times.data(),
                                        linkid_vals.data(),
                                        num_queries,
                                        num_systems,
                                        time_origin,
                                        compression_level);
                auto end_ro = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_ro = end_ro - start_ro;
                std::cout << "Writing runoff output took " 
                        << elapsed_ro.count() 
                        << " seconds\n";


                // // ───────── Write selected states to NetCDF ─────────
                std::cout << "Writing SELECTED → "<< selected_filename    << "\n";
                int states[] = {
                    Model204::STATE_SURF_RUNOFF,
                    Model204::STATE_TOTAL_RUNOFF
                };
                const char* names[] = {
                    "surface_runoff",
                    "total_runoff"
                };

                auto t0_runoff = std::chrono::high_resolution_clock::now();
                write_selected_dense_netcdf(selected_filename,
                                            h_dense.data(),
                                            h_query_times.data(),
                                            linkid_vals.data(),
                                            states,
                                            names,
                                            /*num_selected=*/2,
                                            num_queries,
                                            num_systems,
                                            /*full_N_EQ=*/Model204::N_EQ,
                                            time_origin,
                                            compression_level);
                auto t1_runoff = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> dt = t1_runoff - t0_runoff;
                std::cout << "Writing selected states took "
                        << dt.count() << " seconds\n";


                    // ───────── NetCDF writer setup ───────────────────────────────────────────────────────────────
                    // // 1) build 32‐bit system IDs
                    // std::vector<uint32_t> linkIDs(num_systems);
                    // for (int s = 0; s < num_systems; ++s) {
                    //     linkIDs[s] = static_cast<uint32_t>(streams[s].id);
                    // }

                    // // 2) build state‐var IDs [0…N_EQ-1]
                    // std::vector<int> stateIDs(Model204::N_EQ);
                    // std::iota(stateIDs.begin(), stateIDs.end(), 0);

                    // // 3) pick “selected” outputs
                    // std::vector<SelectedVar> sel = {
                    //     { Model204::STATE_SURF_RUNOFF,  "surface_runoff" },
                    //     { Model204::STATE_TOTAL_RUNOFF, "total_runoff"   }
                    // };

                    // // ────── how much dense‐output we will write ──────
                    // size_t totalSteps_actual = num_queries;
                    // size_t nSystems   = size_t(num_systems);
                    // size_t nVars      = size_t(Model204::N_EQ);
                    // size_t nValues    = totalSteps_actual * nSystems * nVars;            // total doubles
                    // double sizeMiB    = nValues * sizeof(double) / (1024.0*1024.0);

                    // std::cout << "[MEMCHK] Dense‐output to write: "
                    //         << totalSteps_actual << " time‐steps × "
                    //         << nSystems   << " systems × "
                    //         << nVars      << " vars = "
                    //         << nValues    << " values ("
                    //         << std::fixed << std::setprecision(2)
                    //         << sizeMiB    << " MiB)"
                    //         << std::endl;
                    // // ─────────────────────────────────────────────────

                    // // 4) instantiate the writer (one file per MPI rank)
                    // std::string outPath =
                    //     "/scratch/gpfs/mb6477/Tiger_HLM_GPU/outputs/dense_small_100days_e6_rank"
                    //     + std::to_string(rank) + ".nc";

                        

                    // int mode = DENSE; //| RUNOFF | SELECTED;
                    // StreamingNetCDF writer(
                    //     outPath,
                    //     linkIDs,
                    //     stateIDs,
                    //     Model204::N_EQ,
                    //     mode,
                    //     sel,
                    //     /*deflate=*/0
                    // );

                    // // ────── begin dense‐output timing ──────
                    // std::cout << "Starting NetCDF dense output write..." << std::endl;
                    // auto write_start = std::chrono::high_resolution_clock::now();


                    // // 5) append in 1-day (24-step) slabs
                    // int  slabSize    = 24;               // hourly → 24 per day, till 100 days it works fine
                    // int  slabStart   = 0;                // our new loop‐counter (must be int!)
                    // int  totalSteps  = num_queries;      // also int

                    // while (slabStart < totalSteps) {
                    //     // both args to std::min are now ints
                    //     int slabLen = std::min(slabSize, totalSteps - slabStart);

                    //     // pointer‐arithmetic uses the int offset
                    //     const double* timePtr  = h_query_times.data() + slabStart;
                    //     const double* densePtr = h_dense.data()
                    //         + size_t(slabStart) * num_systems * Model204::N_EQ;

                    //     writer.appendSlab(densePtr, timePtr, slabStart, slabLen);

                    //     slabStart += slabLen;
                    // }

                    // auto write_end = std::chrono::high_resolution_clock::now();
                    // auto write_dur = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start);
                    // std::cout << "Finished NetCDF dense output write in "
                    //         << write_dur.count() << " ms." << std::endl;
                    
                    // ────────────────────────────────────────────────────────────────────────────────────────────────────
                    // move to next chunk’s start date
                    advanceDate(chunkStartYear, chunkStartMon, chunkStartDay, daysThisChunk);
            } // end of 100-day loop






        
            
    }

    MPI_Finalize();
    return 0;
}