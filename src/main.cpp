// Standard Library Headers
#include <sstream>
#include <string>
#include <cstdio>       
#include <cstdlib>      
#include <cmath>        
#include <algorithm>    
#include <numeric>      
#include <vector>       
#include <array>        
#include <fstream>      
#include <iomanip>      
#include <utility>      
#include <cstdint>     
#include <stdexcept>    
#include <iostream>     
#include <ctime>        
#include <filesystem>   
#include <unordered_map>
namespace fs = std::filesystem;  
#include <limits>


// CUDA & GPU Headers
#include <cuda_runtime.h>           

// Project Core Headers
#include "solver/rk45.h"                  // Runge-Kutta 4/5 core solver interface (host side)
#include "solver/rk45_api.hpp"            // Host-side RK45 API: setup_gpu_buffers, launch_rk45_kernel, etc.
#include "solver/rk45_step_dense.cuh"     // Device kernel for one RK45 step + dense output
#include "solver/radau_step_dense.cuh"    // Device kernel for Radau-IIA step + dense output
#include "solver/radau_kernel.cuh"        // Radau-only kernel for specific model (e.g. Runoff5)
#include "solver/small_lu.cuh"            // Small-matrix LU solver used by Radau (for implicit steps)
#include "solver/event_detector.cuh"      // Event detection (e.g., slope jumps, stiffness) on device
#include "simulation_driver.hpp"          // Simulation core loop (manages time stepping, control flow)

// Model & Parameters
#include "models/model_Runoff5.hpp"     // SpatialParams and model-specific logic
#include "models/model_registry.hpp"    // Registers models and parameters
#include "stream.hpp"                   // Stream wrapper for simulation (manages model ID, initial state, etc.)
#include "I_O/config_loader.hpp"        // Loads model configuration settings (e.g., from JSON)
#include "I_O/parameters_loader.hpp"    // Loads parameters (e.g., spatial or physical) from CSV

// Forcing & Input/Output
#include "I_O/forcing_loader.hpp"  // Loads external forcing data from NetCDF and maps it
#include "I_O/forcing_data.h"      // Provides forcing data structure
#include "I_O/output_series.hpp"   // Serial output to NetCDF format

// MPI & Timing Utilities
#include <mpi.h>                   
#include <chrono>                  



// ───────── Global Variables ─────────
double GLOBAL_QUERY_DT = 60.0;   // default output interval (minutes)

// Bring in RK45 API functions
using rk45_api::setup_gpu_buffers;   
using rk45_api::retrieve_and_free; 

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

// Helper macros for CUDA error checking
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr,                                                    \
        "[CUDA ERROR] %s:%d: “%s” failed %s\n",                               \
        __FILE__, __LINE__, #call, cudaGetErrorString(err));                  \
      std::exit(1);                                                           \
    }                                                                         \
  } while(0)

// Type alias for time points
double elapsedSeconds(TimePoint start, TimePoint end) {
    return std::chrono::duration<double>(end - start).count();
}

// Parses ISO date string like "2019-12-25" or "2019-12-25T00:00:00Z"
std::tm parseDate(const std::string& iso_date) {
    std::tm t{};
    sscanf(iso_date.c_str(), "%4d-%2d-%2d", &t.tm_year, &t.tm_mon, &t.tm_mday);
    t.tm_year -= 1900;
    t.tm_mon -= 1;
    return t;
}

// Compute days since Jan 1 of that year
int computeDayOfYear(const std::tm& t) {
    std::tm jan1 = t;
    jan1.tm_mon = 0;
    jan1.tm_mday = 1;
    time_t start = timegm(&jan1);
    time_t target = timegm(const_cast<std::tm*>(&t));
    return static_cast<int>((target - start) / 86400);
}

// Equal split helper: returns [start,end) for 'index' among 'parts'
inline std::pair<int,int> split_equal(int total, int parts, int index) {
    int base = total / parts;                 // floor
    int rem  = total % parts;                 // first 'rem' chunks get one extra
    int start = index * base + std::min(index, rem);
    int count = base + (index < rem ? 1 : 0);
    return {start, start + count};
}

// Sort SpatialParams by stream/system ID so slices are contiguous and reproducible
inline void sort_by_stream(std::vector<SpatialParams>& v) {
    std::sort(v.begin(), v.end(),
              [](const SpatialParams& a, const SpatialParams& b){
                  return a.stream < b.stream;
              });
}

// Print “[IC CHECK] …” only once across the whole run (and only on rank 0)
static bool g_ic_printed_once = false;

// Is this the very first chunk of the whole simulation window?
// static bool is_first_chunk_of_sim(const ModelConfig& cfg,
//                                   int simYear, int dayOffset) {
//     std::tm s = parseDate(cfg.time_start);
//     const int start_year = s.tm_year + 1900;
//     const int start_day  = computeDayOfYear(s);
//     return (simYear == start_year) && (dayOffset == start_day);
// }



// check if the device has enough memory for the forcing data
static void verifyDeviceForcingsHostSide(const NCForcing& chunk) {
    // 1) Pull back the scalars/arrays from __constant__ to host and compare
    int dev_nForc = -1;
    CUDA_CHECK(cudaMemcpyFromSymbol(&dev_nForc, nForc, sizeof(int)));
    if (dev_nForc != chunk.nForc) {
        std::cerr << "[FORC-CHECK] nForc mismatch: host=" << chunk.nForc
                  << " device=" << dev_nForc << "\n";
    }

    std::vector<double> dev_dt(chunk.nForc, -1.0);
    std::vector<size_t> dev_nT(chunk.nForc, 0);
    CUDA_CHECK(cudaMemcpyFromSymbol(dev_dt.data(), c_forc_dt, sizeof(double) * chunk.nForc));
    CUDA_CHECK(cudaMemcpyFromSymbol(dev_nT.data(), c_forc_nT, sizeof(size_t) * chunk.nForc));

    for (int f = 0; f < chunk.nForc; ++f) {
        if (dev_dt[f] != chunk.dt[f] || dev_nT[f] != chunk.nT[f]) {
            std::cerr << "[FORC-CHECK] meta mismatch f="<<f
                      << " dt host="<<chunk.dt[f]<<" dev="<<dev_dt[f]
                      << " nT host="<<chunk.nT[f]<<" dev="<<dev_nT[f] << "\n";
        }
    }

    // 2) Pull back a few specific values from the big d_forc_data and compare
    float* dptr = nullptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&dptr, d_forc_data, sizeof(dptr)));

    auto baseF = [&](int f) {
        size_t base = 0;
        for (int k = 0; k < f; ++k) base += chunk.nT[k] * (size_t)chunk.systems;
        return base;
    };

    auto check_point = [&](int f, size_t t, size_t s) {
        size_t idx = baseF(f) + t * (size_t)chunk.systems + s;
        float hval = -999, dval = -999;
        // host value
        hval = chunk.h_data[idx];
        // device value
        CUDA_CHECK(cudaMemcpy(&dval, dptr + idx, sizeof(float), cudaMemcpyDeviceToHost));
        if (!(std::isfinite(hval) && std::isfinite(dval)) || hval != dval) {
            std::cerr << "[FORC-CHECK] value mismatch f="<<f<<" t="<<t<<" s="<<s
                      << " host="<<hval<<" dev="<<dval<<" idx="<<idx<<"\n";
        } else {
            std::cout << "[FORC-CHECK] OK f="<<f<<" t="<<t<<" s="<<s<<" = "<<dval<<"\n";
        }
    };

    // check a tiny set (safe even for 10M systems)
    check_point(0, 0, 0);
    if (chunk.nT[0] > 1) check_point(0, 1, 0);
    if (chunk.nForc > 1) check_point(1, 0, 0);
}




__global__ void checkForcingsOnDeviceSafe(const float* d_forc,
                                          int nForc, int days, int systems,
                                          size_t totalCount)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    printf(">>> [GPU] SAFE FORCING CHECK <<<\n");
    size_t expected = 0;
    for (int k = 0; k < nForc; ++k) expected += c_forc_nT[k] * (size_t)systems;
    printf("    nForc=%d, days=%d, systems=%d, expected_total=%llu (allocated=%llu)\n",
        nForc, days, systems,
        (unsigned long long)expected, (unsigned long long)totalCount);


    int maxF = min(nForc, 2);
    int maxD = min(days, 3);
    int maxS = min(systems, 1);

    for (int f = 0; f < maxF; ++f) {
        printf("  Forcing[%d]:\n", f);
        for (int d = 0; d < maxD; ++d) {
            for (int s = 0; s < maxS; ++s) {
                // Compute linear index
                // Formula:
                // idx = ( Σ_{k=0}^{f-1} c_forc_nT[k] * systems )   // skip all previous forcings
                //     + (d * systems)                              // skip previous timesteps in this forcing
                //     + s;                                         // offset for the current system
                size_t base = 0;
                // Step 1: accumulate the size of all forcings before 'f'
                // Each previous forcing k has c_forc_nT[k] timesteps, and each timestep
                // stores 'systems' values (one per system).
                for (int k = 0; k < f; ++k) {
                    base += c_forc_nT[k] * (size_t)systems;
                }
                // Step 2: add the offset for the current timestep d within forcing f.
                // Multiply by 'systems' because each timestep holds one value per system.
                // Step 3: add the offset + (size_t)s for the current system s within this timestep of this forcing.
                size_t idx = base + (size_t)d * (size_t)systems + (size_t)s;
                // Safety check: ensure index is within allocated array bounds 
                if (idx >= totalCount) {
                    printf("    [OOB] f=%d d=%d s=%d idx=%llu\n",
                           f, d, s, (unsigned long long)idx);
                    continue;
                }
                float val = d_forc[idx]; 
                printf("    f=%d d=%d s=%d → %f\n", f, d, s, val);
            }
        }
    }
    printf("─────────────────────────\n");
}



// ───────── Validate command-line arguments ─────────
bool validateArgs(int argc) {
    if (argc < 2) {
        std::cerr << "Usage: ./program <config.yaml>\n";
        return false;
    }
    return true;
}

// ─────────  Logging Utilities ─────────
void printSeparator() {
    std::cout << "────────────────────────────────────────────────────────────\n";
}

void printSimHeader(int year, int startDay, int endDay, int numSystems) {
    printSeparator();
    std::cout << "[SIM]   Year: " << year 
              << "  | Days: " << startDay << " → " << endDay
              << "  | Systems: " << numSystems << "\n";
    printSeparator();
}

void logInfo(const std::string& msg) {
    std::cout << "[INFO] " << msg << "\n";
}

void logGpu(const std::string& msg) {
    std::istringstream iss(msg);   
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); 
        std::cout << "[GPU] " << line << "\n";
    }
}

void logTimer(const std::string& label, double seconds) {
    // Log the elapsed time with fixed precision
    std::cout << "[TIMER] " << label << ": " 
              << std::fixed << std::setprecision(2) 
              << seconds << " s\n";
}

void logWrite(const std::string& file) {
    std::cout << "[WRITE] " << file << "\n";
}

void logDone(int systems, int queries) {
    std::cout << "[DONE]  Outputs written (" 
              << systems << " × " << queries << ")\n";
    printSeparator();
}
 

// ───────── Load model configuration from YAML ─────────
bool loadConfiguration(const std::string& path, ModelConfig& config) {
    try {
        config = ConfigLoader::loadConfig(path);
        std::cout << "[INFO] Successfully loaded configuration from: " << path << "\n";
        std::cout << "[CONFIG] Simulation window: " << config.time_start
          << " to " << config.time_end << "\n";

        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Failed to load config: " << ex.what() << "\n";
        return false;
    }
}


// ───────── Get number of available GPU devices ─────────
int getGpuCount() {
    int gpuCount = 0;
    cudaError_t err = cudaGetDeviceCount(&gpuCount);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] cudaGetDeviceCount() failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(1);
    }
    return gpuCount;
}

// ───────── Assign GPU device to this MPI rank ─────────
void assignGpuDevice(int rank, int gpuCount) {
    int nDevices = 0;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess || nDevices == 0) {
        fprintf(stderr, "[CUDA ERROR] Failed to get device count: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Map rank to a valid device index
    int device_id = rank % nDevices;

    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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
}



// ───────── Rank 0 distributes SpatialParams evenly across ranks ─────────
void distributeSpatialParams(const ModelConfig& config, int size) {
    // Load spatial parameters from configured CSV path
    auto spatialParams = loadSpatialParams(config.parameters_path + config.spatially_varying_file);

    // Compute number of ranks to distribute to (excluding rank 0)
    int totalRanks = size - 1;
    int totalRows  = spatialParams.size();
    int baseChunk  = totalRows / totalRanks;
    int remainder  = totalRows % totalRanks;

    int start = 0;

    std::cout << "[MPI] Total spatial rows: " << totalRows << "\n";
    std::cout << "[MPI] Total worker ranks:  " << totalRanks << "\n";

    for (int r = 1; r <= totalRanks; ++r) {
        // Compute chunk size for this rank (distribute remainder fairly)
        int chunkSize = baseChunk + (r <= remainder ? 1 : 0);
        int end = std::min(start + chunkSize, totalRows);

        // Slice spatial subset for this rank
        std::vector<SpatialParams> subset(spatialParams.begin() + start, spatialParams.begin() + end);
        int count = subset.size();

        std::cout << "[MPI] Sending " << count << " SpatialParams to rank " << r << "\n";

        // Send count first
        MPI_Send(&count, 1, MPI_INT, r, 0, MPI_COMM_WORLD);

        // Send raw data as bytes
        MPI_Send(subset.data(), count * sizeof(SpatialParams), MPI_BYTE, r, 1, MPI_COMM_WORLD);

        // Move start to next chunk
        start = end;
    }
}

// ───────── Non-root ranks receive their SpatialParams subset ─────────
std::vector<SpatialParams> receiveSpatialParams() {
    int count;

    // Receive the number of SpatialParams this rank should expect
    MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Allocate buffer and receive the actual data
    std::vector<SpatialParams> receivedSubset(count);
    MPI_Recv(receivedSubset.data(), count * sizeof(SpatialParams), MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return receivedSubset;
}


// ───────── Build vector of Stream<Runoff5> from SpatialParams ─────────
std::vector<Stream<Runoff5>> buildStreams(const std::vector<SpatialParams>& params,
                                          const ModelConfig& config) {
    std::array<double, Runoff5::N_EQ> y0_common;

    if (config.initial_mode == "from_file") {
        // placeholder; will be overwritten by load_initial_from_final_nc_serial()
        y0_common = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    } else {
        // constant mode: prefer YAML values if present, else fallback to old default
        std::array<double, Runoff5::N_EQ> fallback = {0.01, 0.1, 0.0, 0.1, 0.01, 1, 1, 0, 0};
        
        // // ─────────DEBUG print initial values ────────────────────────────────────
        // std::cerr << "[CFG] initial_mode=" << config.initial_mode
        //         << " values.size=" << config.initial_values.size() << "\n";
        // for (size_t i = 0; i < config.initial_values.size(); ++i)
        //     std::cerr << "[CFG] values[" << i << "]=" << config.initial_values[i] << "\n";
        // // ────────────────────────────────────────────────────────────────────────

        if (!config.initial_values.empty()) {
            for (int i = 0; i < Runoff5::N_EQ; ++i)
                y0_common[i] = config.initial_values[i];
        } else {
            y0_common = fallback;
        }
    }

    std::vector<Stream<Runoff5>> streams;
    streams.reserve(params.size());
    for (const auto& sp : params) {
        streams.emplace_back(sp, y0_common);
    }
    return streams;
}


// ───────── Allocate & upload SpatialParams to device ─────────
void setupGpu(const std::vector<SpatialParams>& spatialParams,
              const std::vector<Stream<Runoff5>>& streams) {

    int num_systems = static_cast<int>(streams.size());

    // Extract SpatialParams from stream objects
    std::vector<SpatialParams> hostSP;
    hostSP.reserve(num_systems);
    for (const auto& st : streams) {
        hostSP.push_back(st.sp);
    }

    // Allocate GPU memory for SpatialParams
    size_t byteCount = num_systems * sizeof(SpatialParams);
    SpatialParams* d_sp = nullptr;
    cudaError_t err = cudaMalloc(&d_sp, byteCount);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(d_sp) failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }

    // Copy SpatialParams to device
    err = cudaMemcpy(d_sp, hostSP.data(), byteCount, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy(d_sp) failed: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }

    // Full compare host→device→host copy
    {
        std::vector<SpatialParams> hostCheck(num_systems);
        err = cudaMemcpy(hostCheck.data(), d_sp, byteCount, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
            for (int i = 0; i < num_systems; ++i) {
                const auto& in  = hostSP[i];
                const auto& out = hostCheck[i];
                if (in.stream != out.stream ||
                    std::fabs(in.Hu    - out.Hu)    > 1e-12 ||
                    std::fabs(in.infil - out.infil) > 1e-12 ||
                    std::fabs(in.perco - out.perco) > 1e-12)
                {
                    std::fprintf(stderr,
                        "[MISMATCH] idx %d: host(stream=%ld, Hu=%g) vs device(stream=%ld, Hu=%g)\n",
                        i, in.stream, in.Hu, out.stream, out.Hu);
                }
            }
        }
    }

    // Set device-side constant pointer to devSpatialParamsPtr
    err = cudaMemcpyToSymbol(devSpatialParamsPtr, &d_sp, sizeof(d_sp));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyToSymbol(devSpatialParamsPtr) failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(1);
    }
}

// Pretty print the query grid for one chunk
static void logQueryGrid(const SolverInputs& in, const char* where) {
    const double qdt = GLOBAL_QUERY_DT;          // minutes
    const double span = in.tf - in.t0;           // minutes
    const int    nq   = (int)in.h_query_times.size();
    const double rem  = span - std::floor(span / qdt) * qdt;

    std::cout << "[QGRID] (" << where << ") "
              << "t0=" << in.t0 << " min, tf=" << in.tf << " min, "
              << "Δ=" << qdt << " min, span=" << span << " min, "
              << "nq=" << nq << ", remainder=" << rem << " min\n";

    // Print first/last few query times
    auto peek = [&](int i){
        if (i >= 0 && i < nq) std::cout << "  t[" << i << "]=" << in.h_query_times[i] << " min\n";
    };
    if (nq > 0) {
        peek(0);
        if (nq > 1) peek(1);
        if (nq > 2) peek(2);
        if (nq > 5) {
            std::cout << "  ... (" << nq-6 << " more) ...\n";
        }
        if (nq > 5) {
            peek(nq-3); peek(nq-2); peek(nq-1);
        } else {
            for (int i = 3; i < nq; ++i) peek(i);
        }
    }
}

// ───────── Determines if a given year is a leap year ───────── 
// Leap year rule: divisible by 4, but not by 100 unless also divisible by 400
bool isLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

// --- SUPER-SIMPLE GPU PROBE (prints a few SpatialParams + first 2 forcing samples) ---!!! 
__global__ void gpu_quick_probe(const SpatialParams* sp,
                                const float* d_forc,
                                const size_t* c_forc_nT_dev,
                                const double* c_forc_dt_dev,
                                int nForc, int systems,
                                int first_sys, int count_sys)
{
    if (blockIdx.x || threadIdx.x) return;

    int s0 = max(0, first_sys);
    int s1 = min(systems, s0 + max(0, count_sys));

    printf(">>> [GPU] QUICK PROBE: nForc=%d systems=%d range=[%d..%d)\n",
           nForc, systems, s0, s1);

    for (int s = s0; s < s1; ++s) {
        const SpatialParams& P = sp[s];
        printf("  [SP] sys=%d stream=%lld Hu=%.6g infil=%.6g perco=%.6g "
               "lat=%.5f lon=%.5f slope=%.6g n=%.4g L=%.3f A_h=%.3f "
               "a3=%.1f a4=%.1f melt_f=%.6g thr=%.3g c1=%.9g\n",
               s, (long long)P.stream, P.Hu, P.infil, P.perco,
               P.lat, P.lon, P.slope, P.n_mann, P.L, P.A_h,
               P.alpha3, P.alpha4, P.melt_f, P.temp_thr, P.c1);

        // print first up to 2 timesteps for each forcing for this system
        size_t base = 0;
        for (int f = 0; f < nForc; ++f) {
            size_t nTf = c_forc_nT_dev[f];
            double dt_hr = c_forc_dt_dev ? c_forc_dt_dev[f] : -1.0;
            printf("    [F] f=%d nT=%llu dt_hr=%.3f\n",
                   f, (unsigned long long)nTf, dt_hr);

            size_t tmax = nTf < 2 ? nTf : 2;
            for (size_t t = 0; t < tmax; ++t) {
                size_t idx = base + t * (size_t)systems + (size_t)s;
                float v = d_forc[idx];
                printf("      t=%llu val=%g (idx=%llu)\n",
                       (unsigned long long)t, (double)v, (unsigned long long)idx);
            }
            base += nTf * (size_t)systems;
        }
    }
    printf(">>> [GPU] END PROBE\n");
}



// ───────── Loop over a single simulation year ─────────
void simulateYear(int year, const ModelConfig& config, std::vector<Stream<Runoff5>>& streams,
                int rank, bool usingMPI) {
    
    // Initialize chunk start date
    int chunkStartYear = year, chunkStartMon = 1, chunkStartDay = 1;
    int TOTAL_DAYS = isLeapYear(year) ? 366 : 365; // determine if leap year
    int num_systems = streams.size();

    // Parse full start and end dates
    std::tm start_tm = parseDate(config.time_start);
    std::tm end_tm   = parseDate(config.time_end);

    // Only simulate if this year overlaps with config window
    if (start_tm.tm_year + 1900 > year || end_tm.tm_year + 1900 < year)
        return;

    // Determine bounds within this year
    int start_day = (start_tm.tm_year + 1900 == year) ? computeDayOfYear(start_tm) : 0;
    int end_day   = (end_tm.tm_year + 1900 == year)   ? computeDayOfYear(end_tm)   : (TOTAL_DAYS - 1);

    // Determine chunk size (days per chunk)
    int DAYS_PER_CHUNK = -1;

    // Print simulation header
    printSimHeader(year, start_day, end_day, num_systems);

    // Check if any forcing variable has a valid, positive chunk size
    for (const auto& f : config.forcing_variables) {
        if (f.time_chunk_size > 0) {
            DAYS_PER_CHUNK = f.time_chunk_size;
            break;  // Use the first specified one
        }
    }

    // Fallback to auto-compute
    if (DAYS_PER_CHUNK <= 0) {
        DAYS_PER_CHUNK = computeDaysPerChunk(num_systems, config);
        std::cout << "[INFO] Using computed DAYS_PER_CHUNK = "
                << DAYS_PER_CHUNK << "\n";
    } else {
        std::cout << "[INFO] Using user-specified DAYS_PER_CHUNK = " << DAYS_PER_CHUNK << "\n";
        int num_chunks = (end_day - start_day + DAYS_PER_CHUNK) / DAYS_PER_CHUNK;
        std::cout << "[INFO] This year will be simulated in " << num_chunks
                << " chunks of up to " << DAYS_PER_CHUNK << " days each\n";

    }


    for (int dayOffset = start_day; dayOffset <= end_day; dayOffset += DAYS_PER_CHUNK) {
        // Ensure we do not simulate beyond the configured end_day
        int remainingDays = end_day - dayOffset + 1;
        int daysThisChunk = std::min(DAYS_PER_CHUNK, remainingDays);

        // simulateChunk(config, streams, year, dayOffset, daysThisChunk, rank, usingMPI);!!!
        // Determine if this is the last chunk of the year → include tf sample
        // bool include_tf = (dayOffset + daysThisChunk - 1) == end_day; //!!!
        
        // Always include the chunk-end sample (tf). Safe because queries start at t0+Δ.
        // bool include_tf = true;
        // const bool is_last_chunk = (dayOffset + daysThisChunk - 1) == end_day;
        // bool include_tf = is_last_chunk;
        bool include_tf = true;  // include the seam sample in every chunk

        // DEBUG: log chunk info
        std::cout << "[CHUNK DEBUG] Year " << year
                << " offset=" << dayOffset
                << " days=" << daysThisChunk
                << " → include_tf=" << std::boolalpha << include_tf << "\n";

        simulateChunk(config, streams, year, dayOffset, daysThisChunk, rank, usingMPI, include_tf);


        // Advance date pointer for next chunk
        advanceDate(chunkStartYear, chunkStartMon, chunkStartDay, daysThisChunk);
    }

}

// ───────── Estimate DAYS_PER_CHUNK based on GPU memory ─────────
int computeDaysPerChunk(int num_systems, const ModelConfig& config) {
    const int N_EQ = Runoff5::N_EQ;
    
    const double usable_bytes =
        config.max_gpu_mem_gb *
        (1.0 - config.gpu_mem_buffer_pct / 100.0) *
        1024.0 * 1024.0 * 1024.0;

    

    // Use the single source of truth
    double qdt = GLOBAL_QUERY_DT;
    if (!std::isfinite(qdt) || qdt <= 0.0) qdt = 60.0;
    if (qdt > 1440.0) qdt = 1440.0;

    const double samples_per_day = 1440.0 / qdt;

    // Dense output estimate (float)
    const double bytes_per_day_dense = double(num_systems) * double(N_EQ) * samples_per_day * sizeof(float);

    // Keep a little slack (-1 day)
    const double safe = std::max(1.0, bytes_per_day_dense);
    int days = std::max(1, static_cast<int>(std::floor(usable_bytes / safe - 1.0)));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "Total memory = " << config.max_gpu_mem_gb << " GiB\n"
        << "Buffer reserved = " << config.gpu_mem_buffer_pct << "%\n"
        << "Usable memory = " << (usable_bytes / (1024.0*1024.0*1024.0)) << " GiB\n"
        << "Query interval = " << qdt << " min (" << samples_per_day << " samples/day)\n"
        << "Estimated DAYS_PER_CHUNK = " << days;
    logGpu(oss.str());

    return days;
}


// ───────── Simulate a chunk of days within a year ─────────
void simulateChunk(const ModelConfig& config,
                   std::vector<Stream<Runoff5>>& streams,
                   int simYear,
                   int dayOffset,
                   int daysThisChunk,
                   int rank, bool usingMPI,
                   bool include_tf) {
    // Determine if this is the first chunk of the whole simulation window
    // const bool first_chunk = is_first_chunk_of_sim(config, simYear, dayOffset);

    int num_systems = streams.size();

    // Log date range per chunk
    int Y = simYear, M = 1, D = 1;
    advanceDate(Y, M, D, dayOffset);
    std::string chunkStart = formatDate(Y, M, D);
    advanceDate(Y, M, D, daysThisChunk - 1);
    std::string chunkEnd = formatDate(Y, M, D);

    std::cout << "[CHUNK] Simulating from " << chunkStart
            << " to " << chunkEnd
            << " (days=" << daysThisChunk << ")\n";


    // ───────── Load forcing data for this chunk ─────────
    TimePoint t_input_start = Clock::now();
    NCForcing forcingChunk = loadForcingData(config, simYear, dayOffset, daysThisChunk, num_systems, streams);

    // ───── DEBUG: Print SpatialParams for system s=0 ─────
    // if (!streams.empty()) {
    //     const SpatialParams& sp = streams[0].sp;
    //     std::cout << "[DEBUG SPATIAL] System s=0 SpatialParams:\n"
    //             << "  streamID=" << sp.stream
    //             << " A_h=" << sp.A_h
    //             << " slope=" << sp.slope
    //             << " n_mann=" << sp.n_mann
    //             // Add more fields from SpatialParams as needed
    //             << std::endl;
    // }
    
    // ───────── Copy forcings to device ─────────
    uploadForcingsToGpu(forcingChunk);
    verifyDeviceForcingsHostSide(forcingChunk);   // round-trip check!!!

    // --- DEBUG: quick GPU probe of first few SpatialParams + forcings ---
    // {
    //     // Get device pointers/symbols
    //     SpatialParams* d_sp = nullptr;
    //     CUDA_CHECK(cudaMemcpyFromSymbol(&d_sp, devSpatialParamsPtr, sizeof(d_sp)));

    //     float* d_forc_ptr = nullptr;
    //     CUDA_CHECK(cudaMemcpyFromSymbol(&d_forc_ptr, d_forc_data, sizeof(d_forc_ptr)));

    //     // Addresses of device-constant arrays as raw device pointers
    //     size_t* d_c_forc_nT = nullptr;
    //     CUDA_CHECK(cudaGetSymbolAddress((void**)&d_c_forc_nT, c_forc_nT));

    //     double* d_c_forc_dt = nullptr;
    //     CUDA_CHECK(cudaGetSymbolAddress((void**)&d_c_forc_dt, c_forc_dt));

    //     int dev_nForc = 0;
    //     CUDA_CHECK(cudaMemcpyFromSymbol(&dev_nForc, nForc, sizeof(int)));

    //     // Safety
    //     if (d_sp && d_forc_ptr && dev_nForc > 0) {
    //         // Probe first 3 systems (adjust as needed)
    //         int first_sys = 0;
    //         int count_sys = std::min(3, static_cast<int>(forcingChunk.systems));
    //         // (Optional) enlarge device printf buffer if you want to see more output
    //         // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8*1024*1024));

    //         gpu_quick_probe<<<1,1>>>(d_sp, d_forc_ptr,
    //                                 d_c_forc_nT, d_c_forc_dt,
    //                                 dev_nForc, forcingChunk.systems,
    //                                 first_sys, count_sys);
    //         CUDA_CHECK(cudaDeviceSynchronize());
    //     }
    // }
    // ──────────────────────────────────────────────────────────────── 
    
    TimePoint t_input_end = Clock::now();
    logTimer("Input", elapsedSeconds(t_input_start, t_input_end));

    // DEBUG: print first 3 precipitation & t2m values for system 0
    // std::cout << "[DEBUG-FORC] first 3 pr/t2m at sys=0:\n";
    for (int f = 0; f < forcingChunk.nForc; ++f) {
    // std::cout << " forcing " << f << " dt=" << forcingChunk.dt[f]
    //             << " steps=" << forcingChunk.nT[f] << ": ";
    for (int t = 0; t < std::min<size_t>(3, forcingChunk.nT[f]); ++t) {
        // h_data is interleaved [f][t][system]
        size_t base = 0;
        for (int k = 0; k < f; ++k) base += forcingChunk.nT[k] * forcingChunk.systems;
        size_t idx = base + t * forcingChunk.systems + 0;
        // Print first 3 values for system 0
        // std::cout << forcingChunk.h_data[idx]
        //         << (t+1 < std::min<size_t>(3,forcingChunk.nT[f]) ? ", " : "");
    }
    // std::cout << "\n";
    }


    // ───────── Run GPU DEBUG forcing check ─────────
    // std::cout << "[DEBUG] Running safe GPU forcing check...\n";
    // float* d_forc_ptr = nullptr;
    // CUDA_CHECK(cudaMemcpyFromSymbol(&d_forc_ptr, d_forc_data, sizeof(float*)));
    // checkForcingsOnDeviceSafe<<<1,1>>>(d_forc_ptr,
    //                                 forcingChunk.nForc,
    //                                 forcingChunk.days,
    //                                 forcingChunk.systems,
    //                                 forcingChunk.h_data.size());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ───────── Prepare solver input arrays ─────────
    // DEBUG: print cumulative runoff values at chunk start (for first system)
    if (!streams.empty()) {
        std::cout << "[DEBUG] chunk start surf_runoff="
                << streams[0].y0[Runoff5::STATE_SURF_RUNOFF]
                << " total="
                << streams[0].y0[Runoff5::STATE_TOTAL_RUNOFF]
                << std::endl;
    }
    SolverInputs solverInputs = prepareSolverInputs(simYear, dayOffset, daysThisChunk, streams, include_tf);

    
    // ───── DEBUG: print the initial conditions sent to the GPU (system 0), ONCE ─────
    if (!streams.empty() && rank == 0 && !g_ic_printed_once) {
        std::cout << "[IC CHECK] y0 for s=0: ";
        for (int i = 0; i < Runoff5::N_EQ; ++i) std::cout << solverInputs.h_y0[i] << " ";
        std::cout << "\n";
        g_ic_printed_once = true;
    }

    // ───────── Launch ODE solver on GPU ─────────
    TimePoint t_solver_start = Clock::now();
    std::cout << "[HOST DEBUG] Launching kernel with nForc=" << forcingChunk.nForc << std::endl;
    SolverOutputs solverOutputs = launchSolverKernel(solverInputs, forcingChunk.nForc);
    TimePoint t_solver_end = Clock::now();
    logTimer("Solver runtime", elapsedSeconds(t_solver_start, t_solver_end));

    // ───────── Retrieve results and handle outputs ─────────
    TimePoint t_output_start = Clock::now();
    handleSolverOutputs(config, simYear, dayOffset, daysThisChunk, solverInputs, solverOutputs, streams, rank, usingMPI);
    TimePoint t_output_end = Clock::now();
    std::cout << "[TIMER] Output writing took:"
            << elapsedSeconds(t_output_start, t_output_end) << " seconds\n";
}


// ───────── Load NetCDF forcing data for this chunk ─────────
NCForcing loadForcingData(const ModelConfig& config,
                          int simYear,
                          int dayOffset,
                          int daysThisChunk,
                          int num_systems,
                          const std::vector<Stream<Runoff5>>& streams) {
    // Compute absolute dates for this chunk
    int Ys = simYear, Ms = 1, Ds = 1;
    advanceDate(Ys, Ms, Ds, dayOffset);
    int Ye = Ys, Me = Ms, De = Ds;
    advanceDate(Ye, Me, De, daysThisChunk - 1);

    // Build {forcing-name -> csv path} once
    std::unordered_map<std::string, std::string> name2csv;
    for (const auto& m : config.forcing_mappings) {
        name2csv[m.name] = config.forcing_mappings_path + "/" + m.file;
    }

    std::vector<ForcingEntry> forcings;
    forcings.reserve(config.forcing_variables.size());

    for (const auto& f : config.forcing_variables) {
        const double dt_hr =
            (f.time_resolution == "1h")  ? 1.0 :
            (f.time_resolution == "24h") ? 24.0 :
            throw std::runtime_error("Unsupported resolution: " + f.time_resolution);

        std::vector<std::string> files;

        if (!f.ranged_by_name) {
            // YEARLY mode → single file with {year} replacement
            std::string file = f.file;
            if (auto pos = file.find("{year}"); pos != std::string::npos) {
                file.replace(pos, 6, std::to_string(simYear));
            }
            files = { config.forcings_path + file };
        } else {
            // RANGED mode
            if (!f.files.empty()) {
                files.reserve(f.files.size());
                for (const auto& p : f.files) {
                    // allow absolute or relative under forcings_path
                    if (!p.empty() && (p[0] == '/' || (p.size() > 1 && p[1] == ':'))) {
                        files.push_back(p);
                    } else {
                        files.push_back(config.forcings_path + p);
                    }
                }
            } else if (!f.file_pattern.empty()) {
                // Split the pattern into <prefix>{start}_{end}<suffix>
                const std::string& pat = f.file_pattern;
                auto startPos = pat.find("{start}");
                auto endPos   = pat.find("{end}");
                if (startPos == std::string::npos || endPos == std::string::npos || endPos < startPos) {
                    throw std::runtime_error("file_pattern must contain {start}_{end}: " + pat);
                }

                // Expect "...{start}_{end}..." (underscore in real filenames)
                std::string prefix = pat.substr(0, startPos);
                std::string suffix = pat.substr(endPos + std::string("{end}").size());

                // Scan directory that contains the files
                fs::path dir = fs::path(config.forcings_path) / fs::path(prefix).parent_path();
                std::string fnPrefix = fs::path(prefix).filename().string();
                std::string fnSuffix = fs::path(suffix).filename().string();

                // Pick overlapping files
                auto picked = select_ranged_files_overlapping(dir, fnPrefix, fnSuffix,
                                                              Ys, Ms, Ds, Ye, Me, De);
                if (picked.empty()) {
                    std::ostringstream oss;
                    oss << "No ranged files picked for variable '" << f.name
                        << "' in " << dir.string()
                        << " overlapping " << formatDate(Ys,Ms,Ds) << "-" << formatDate(Ye,Me,De);
                    throw std::runtime_error(oss.str());
                }
                files = std::move(picked);
            } else {
                throw std::runtime_error(
                    "Ranged forcing '" + f.name + "' needs either 'files' or 'file_pattern'.");
            }
        }

        std::cout << "[FORCING PICK] " << f.var_name << " files for "
                  << formatDate(Ys,Ms,Ds) << "-" << formatDate(Ye,Me,De) << ":\n";
        for (auto& p : files) std::cout << "   - " << p << "\n";

        // Build entry and attach per-variable mapper CSV (by YAML `name`, e.g. "pr","t2m")
        ForcingEntry entry{
            /*files=*/std::move(files),
            /*var_name=*/f.var_name,
            /*dt_hours=*/dt_hr,
            /*dims_time=*/f.dims_time,
            /*dims_lat=*/f.dims_lat,
            /*dims_lon=*/f.dims_lon,
            /*lookup_csv=*/{} // set below
        };

        auto it = name2csv.find(f.name);
        if (it != name2csv.end()) {
            entry.lookup_csv = it->second;
            std::cout << "[MAPPING] " << f.name << " -> " << entry.lookup_csv << "\n";
        } else {
            throw std::runtime_error(
                "No forcing_mappings CSV for variable '" + f.name + "'");
        }

        forcings.push_back(std::move(entry));
    }

    NCForcing chunk;
    chunk.entries = std::move(forcings);
    chunk.days    = daysThisChunk;
    chunk.offset  = dayOffset;
    chunk.systems = num_systems;

    // store absolute date range (used by NCForcing::loadData)
    chunk.Ys = Ys; chunk.Ms = Ms; chunk.Ds = Ds;
    chunk.Ye = Ye; chunk.Me = Me; chunk.De = De;

    // system IDs
    chunk.systemIds.resize(num_systems);
    for (size_t i = 0; i < static_cast<size_t>(num_systems); ++i) {
        chunk.systemIds[i] = streams[i].sp.stream;
    }

    return chunk;
}


// ───────── Upload forcing data to device and copy pointer + metadata to device symbols
void uploadForcingsToGpu(NCForcing& chunk) {
    // Load the actual values from NetCDF using the paths in NCForcing
    chunk.loadData();  // populates chunk.h_data, dt, nT, nForc, etc.
    if (chunk.h_data.empty()) {
        throw std::runtime_error("No forcing data loaded for this chunk");
    }

    // Convert cadence from HOURS → MINUTES (kernel expects minutes)
    for (int f = 0; f < chunk.nForc; ++f) {
        chunk.dt[f] *= 60.0;
    }


    // --- Make absolutely sure no kernels are still using the old buffer ---
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- CLEANUP old allocation if it exists ---
    float* old_ptr = nullptr;
    CUDA_CHECK(cudaMemcpyFromSymbol(&old_ptr, d_forc_data, sizeof(old_ptr)));
    if (old_ptr) {
        CUDA_CHECK(cudaFree(old_ptr));
        // Set symbol to nullptr to avoid dangling pointer
        old_ptr = nullptr;
        CUDA_CHECK(cudaMemcpyToSymbol(d_forc_data, &old_ptr, sizeof(old_ptr)));
    }

    // --- Allocate for the new chunk ---
    float* d_ptr = nullptr;
    size_t bytes = sizeof(float) * chunk.h_data.size();
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, chunk.h_data.data(), bytes, cudaMemcpyHostToDevice));

    // --- Update device symbols ---
    CUDA_CHECK(cudaMemcpyToSymbol(d_forc_data, &d_ptr, sizeof(d_ptr)));
    CUDA_CHECK(cudaMemcpyToSymbol(nForc, &chunk.nForc, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_forc_dt, chunk.dt.data(), sizeof(double) * chunk.nForc));
    CUDA_CHECK(cudaMemcpyToSymbol(c_forc_nT, chunk.nT.data(), sizeof(size_t) * chunk.nForc));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0));
    logInfo("Forcing uploaded to GPU (" + oss.str() + " MiB)");
}


SolverInputs prepareSolverInputs(int simYear,
                                  int dayOffset,
                                  int daysThisChunk,
                                  const std::vector<Stream<Runoff5>>& streams,
                                  bool include_tf) {
    SolverInputs input;
    int N_EQ = Runoff5::N_EQ;
    int num_systems = streams.size();

    // Absolute simulation times
    input.t0 = dayOffset * 24.0 * 60.0;
    input.tf = input.t0 + daysThisChunk * 24.0 * 60.0;

    // Flatten initial conditions (zero cumulative runoff states)
    input.h_y0.resize((size_t)num_systems * N_EQ);
    for (int s = 0; s < num_systems; ++s) {
        const auto& y = streams[s].y0;
        double* dst = &input.h_y0[(size_t)s * N_EQ];
        for (int i = 0; i < N_EQ; ++i) dst[i] = y[i];
        
        // Zero cumulative states for this chunk
        dst[Runoff5::STATE_SURF_RUNOFF]  = 0.0;
        dst[Runoff5::STATE_TOTAL_RUNOFF] = 0.0;
    }

    // ── Build query times: t0, t0+Δ, t0+2Δ, ..., tf (if on-grid) ──
    input.h_query_times.clear();
    input.h_query_times.push_back(input.t0);  // ← CRITICAL: anchor point
    
    const double qdt = GLOBAL_QUERY_DT;
    const double span = input.tf - input.t0;
    const int nsteps = (int)std::floor(span / qdt);
    
    for (int i = 1; i <= nsteps; ++i) {
        input.h_query_times.push_back(input.t0 + i * qdt);
    }
    
    // Include tf only if it's on the grid
    if (include_tf) {
        const double r = std::fmod(span, qdt);
        const bool tf_on_grid = (std::fabs(r) < 1e-9) || (std::fabs(r - qdt) < 1e-9);
        if (tf_on_grid &&
            (input.h_query_times.empty() || 
             std::fabs(input.h_query_times.back() - input.tf) > 1e-9)) {
            input.h_query_times.push_back(input.tf);
        }
    }

    logQueryGrid(input, "prepareSolverInputs");
    return input;
}


// ───────── Allocate GPU buffers and launch solver kernel
SolverOutputs launchSolverKernel(const SolverInputs& input, int nForc) {
    SolverOutputs out;
    int num_queries = input.h_query_times.size();
    int num_systems = input.h_y0.size() / Runoff5::N_EQ;

    // Setup buffers
    auto [d_y0_all, d_y_final_all, d_query_times,
        d_dense_ptr, d_stiff_ptr, sys_count, query_count] =
        setup_gpu_buffers<Runoff5>(input.h_y0, input.h_query_times);

    // Assign to output struct
    out.d_y0_all      = d_y0_all;
    out.d_y_final_all = d_y_final_all;
    out.d_query_times = d_query_times;
    out.d_dense_all   = d_dense_ptr;
    out.d_stiff       = d_stiff_ptr;
    out.num_systems   = sys_count;
    out.num_queries   = query_count;

    // Compute dense buffer size safely (use size_t to avoid 32-bit overflow)
    const size_t dense_elems =
        size_t(out.num_systems) * size_t(Runoff5::N_EQ) * size_t(out.num_queries);
    const size_t dense_bytes = dense_elems * sizeof(float); // Use float for dense output

    // Optional paranoid overflow guard:
    if (out.num_systems > 0 && out.num_queries > 0) {
        const size_t back = dense_elems / size_t(Runoff5::N_EQ) / size_t(out.num_queries);
        if (back != size_t(out.num_systems)) {
            throw std::runtime_error("Overflow computing dense buffer size");
        }
    }

    CUDA_CHECK(cudaMemset(out.d_dense_all, 0xAB, dense_bytes));



    // Determine launch configuration
    int blockSize = 0, minGridSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       rk45_then_radau_multi<Runoff5>, 0, 0);

    int numBlocks = (out.num_systems + blockSize - 1) / blockSize;

    logGpu("Launching kernel: blocks=" + std::to_string(numBlocks) +
       ", threads=" + std::to_string(blockSize));
    logGpu("Systems=" + std::to_string(out.num_systems) +
       ", Queries=" + std::to_string(out.num_queries));


    SpatialParams* d_sp = nullptr;
    cudaMemcpyFromSymbol(&d_sp, devSpatialParamsPtr, sizeof(d_sp));

    // Launch the solver
    rk45_then_radau_multi<Runoff5><<<numBlocks, blockSize>>>(
        out.d_y0_all, out.d_y_final_all,    // Initial and final states
        out.d_query_times, out.d_dense_all, // Query times and dense output
        out.num_systems, out.num_queries,   // Number of systems and queries
        input.t0, input.tf,                 // Simulation time bounds
        d_sp,                               // Device pointer to spatial parameters
        out.d_stiff                        // Flags buffer
    );

    CUDA_CHECK( cudaPeekAtLastError() );     // reports launch-time errors
    CUDA_CHECK( cudaDeviceSynchronize() );   // catches run-time errors

    return out;
}


// // Drop-in: always returns rates in mm/hr
void compute_runoff_rate_mm_per_hr(
    const std::vector<float>& dense,     // cumulative meters at query times
    const std::vector<double>& y0,       // state at chunk start (GLOBAL cumulative meters)
    int ns, int nq, int N_EQ,
    int idx_surf, int idx_total,
    double query_dt_min,                 // ← GLOBAL_QUERY_DT
    std::vector<float>& rate_surf_mmhr,
    std::vector<float>& rate_total_mmhr
) {
    rate_surf_mmhr.assign(size_t(ns)*size_t(nq), 0.0f);
    rate_total_mmhr.assign(size_t(ns)*size_t(nq), 0.0f);
    if (ns<=0 || nq<=0 || N_EQ<=0) return;
    if ((int)dense.size() < ns*nq*N_EQ) return;
    if ((int)y0.size()    < ns*N_EQ)    return;

    const double dt_hr = query_dt_min / 60.0;
    if (!(dt_hr > 0.0)) return;

    auto IDX = [&](int s,int t,int k)->size_t {
        return (size_t(s)*size_t(nq)+size_t(t))*size_t(N_EQ)+size_t(k);
    };

    constexpr double EPS_MM = 1e-6;

    for (int s = 0; s < ns; ++s) {
        
        const double base_surf_m  = y0[size_t(s)*size_t(N_EQ)+size_t(idx_surf)];
        const double base_total_m = y0[size_t(s)*size_t(N_EQ)+size_t(idx_total)];

        // First bin: compare to baseline (GLOBAL or LOCAL)
        {
            const double c_surf_m  = (double)dense[IDX(s,0,idx_surf)];
            const double c_total_m = (double)dense[IDX(s,0,idx_total)];

            // If kernel’s dense is LOCAL since t0, baseline is 0.0; otherwise baseline=y0
            const bool dense_is_local_surf  = c_surf_m  + 1e-12 < base_surf_m;
            const bool dense_is_local_total = c_total_m + 1e-12 < base_total_m;

            const double d_surf_mm  = (c_surf_m  - (dense_is_local_surf  ? 0.0 : base_surf_m)) * 1000.0;
            const double d_total_mm = (c_total_m - (dense_is_local_total ? 0.0 : base_total_m)) * 1000.0;

            rate_surf_mmhr [size_t(s)*size_t(nq)+0] = (float)((d_surf_mm  < EPS_MM ? 0.0 : d_surf_mm ) / dt_hr);
            rate_total_mmhr[size_t(s)*size_t(nq)+0] = (float)((d_total_mm < EPS_MM ? 0.0 : d_total_mm) / dt_hr);
        }

        // Subsequent bins: difference consecutive cumulatives
        for (int t = 1; t < nq; ++t) {
            const double p_surf_m  = (double)dense[IDX(s,t-1,idx_surf)];
            const double c_surf_m  = (double)dense[IDX(s,t,  idx_surf)];
            const double p_total_m = (double)dense[IDX(s,t-1,idx_total)];
            const double c_total_m = (double)dense[IDX(s,t,  idx_total)];

            double d_surf_mm  = (c_surf_m  - p_surf_m ) * 1000.0;
            double d_total_mm = (c_total_m - p_total_m) * 1000.0;
            if (d_surf_mm  < EPS_MM) d_surf_mm  = 0.0;
            if (d_total_mm < EPS_MM) d_total_mm = 0.0;

            rate_surf_mmhr [size_t(s)*size_t(nq)+t] = (float)(d_surf_mm  / dt_hr);
            rate_total_mmhr[size_t(s)*size_t(nq)+t] = (float)(d_total_mm / dt_hr);
        }
    }
}

// Converts LOCAL cumulative runoff (meters since chunk t0) at each query time
// into per-bin rates (mm/hr). Assumes a fixed query interval (query_dt_min)
// and that dense holds values at times t1, t2, ... (no sample at t0).
//
// Memory layout:
//   dense[((s * nq) + t) * N_EQ + k]  ← cumulative meters for state k
// Output arrays:
//   out.surf_mmhr [s * nq + t]  ← rate for bin (t-1 → t), mm/hr
//   out.total_mmhr[s * nq + t]  ← rate for bin (t-1 → t), mm/hr
//
// Preconditions (for best results):
//   - The two cumulative states were zeroed at the start of this chunk in y0,
//     so dense truly contains LOCAL cumulatives (monotone non-decreasing).
//   - query_dt_min is constant for this run (e.g., 60 for hourly).
//
// Notes:
//   - First bin uses baseline=0 at t0 (because cumulatives are LOCAL).
//   - Tiny negative deltas from numerical noise are clamped to 0.
//   - Units: input is meters; deltas→mm (×1000); divide by hours for mm/hr.
//
struct RunoffOut {
    std::vector<float> surf_mmhr;   // [ns * nq]
    std::vector<float> total_mmhr;  // [ns * nq]
};

// Keeps the running end-of-chunk cumulative (in meters) for each system
struct SeamState {
    bool initialized = false;
    std::vector<double> last_surf_m;   // size = num_systems
    std::vector<double> last_total_m;  // size = num_systems

    void ensure(int ns) {
        if ((int)last_surf_m.size() != ns) {
            last_surf_m.assign(ns, 0.0);
            last_total_m.assign(ns, 0.0);
            initialized = false; // first chunk for this size
        }
    }
};

// Global seam cache (one per process)
static SeamState g_seam;
// Compute mm/hr from LOCAL cumulatives by stitching a baseline from the previous chunk.
// Output has nq bins aligned to time[0..nq-1] (bin k is from baseline/prev to time[k]).
// Assumes dense_local contains LOCAL cumulatives (meters since the start of this chunk).
inline RunoffOut compute_runoff_rates_stitched(
    const std::vector<float>& dense_local,  // [ns * nq * N_EQ], LOCAL cumulatives (meters)
    int ns, int nq, int N_EQ,
    int idx_surf, int idx_total,
    double query_dt_min,                    // e.g., 60
    SeamState& seam,                        // carries baselines across chunks
    const char* tag = "RUNOFF_STITCH"
) {
    RunoffOut out;
    out.surf_mmhr .assign((size_t)ns * (size_t)nq, 0.0f);
    out.total_mmhr.assign((size_t)ns * (size_t)nq, 0.0f);

    if (ns <= 0 || nq <= 0 || N_EQ <= 0) return out;

    const size_t need = (size_t)ns * (size_t)nq * (size_t)N_EQ;
    if (dense_local.size() < need) return out;

    const double dt_hr = query_dt_min / 60.0;
    if (!(dt_hr > 0.0) || !std::isfinite(dt_hr)) return out;

    auto IDX = [&](int s, int t, int k) -> size_t {
        // Layout: ((s * nq + t) * N_EQ + k)
        return ((size_t)s * (size_t)nq + (size_t)t) * (size_t)N_EQ + (size_t)k;
    };

    // Ensure seam storage sized for current ns
    seam.ensure(ns);

    // Treat tiny millimeter deltas as numerical noise
    constexpr double ABS_EPS_MM = 1e-3; // 0.001 mm

    double gmin = +1e300, gmax = -1e300;

    for (int s = 0; s < ns; ++s) {
        // ---- Monotone repair of LOCAL cumulatives (per system) ----
        // We repair on-the-fly and compute per-bin deltas:
        //   t==0 delta = repaired[0] - 0
        //   t>0  delta = repaired[t] - repaired[t-1]
        double prev_loc_surf  = 0.0;
        double prev_loc_total = 0.0;

        for (int t = 0; t < nq; ++t) {
            double v_s = (double)dense_local[IDX(s, t, idx_surf )];
            double v_t = (double)dense_local[IDX(s, t, idx_total)];

            // NaN/Inf → hold previous repaired value
            if (!std::isfinite(v_s)) v_s = prev_loc_surf;
            if (!std::isfinite(v_t)) v_t = prev_loc_total;

            // Enforce non-decreasing LOCAL cumulatives
            if (v_s < prev_loc_surf ) v_s = prev_loc_surf;
            if (v_t < prev_loc_total) v_t = prev_loc_total;

            // Per-bin delta in mm
            double dmm_surf  = (v_s - (t ? prev_loc_surf  : 0.0)) * 1000.0;
            double dmm_total = (v_t - (t ? prev_loc_total : 0.0)) * 1000.0;

            // Clamp tiny jitter; physically negative runoff is not expected here
            if (std::abs(dmm_surf ) < ABS_EPS_MM) dmm_surf  = 0.0;
            if (std::abs(dmm_total) < ABS_EPS_MM) dmm_total = 0.0;
            if (dmm_surf  < 0.0) dmm_surf  = 0.0;
            if (dmm_total < 0.0) dmm_total = 0.0;

            const size_t oi = (size_t)s * (size_t)nq + (size_t)t;
            out.surf_mmhr [oi] = (float)(dmm_surf  / dt_hr);
            out.total_mmhr[oi] = (float)(dmm_total / dt_hr);

            gmin = std::min(gmin, (double)out.surf_mmhr[oi]);
            gmin = std::min(gmin, (double)out.total_mmhr[oi]);
            gmax = std::max(gmax, (double)out.surf_mmhr[oi]);
            gmax = std::max(gmax, (double)out.total_mmhr[oi]);

            prev_loc_surf  = v_s;
            prev_loc_total = v_t;
        }

        // Advance seam baselines using the **repaired** last LOCAL cumulatives
        seam.last_surf_m[s]  += prev_loc_surf;
        seam.last_total_m[s] += prev_loc_total;
    }

    std::printf("[%s] rate_min=%.6f mm/hr  rate_max=%.6f mm/hr\n", tag, gmin, gmax);
    seam.initialized = true;
    return out;
}


// -----------------------------------------------------------------------------
// Global tolerances for runoff rate computation
// -----------------------------------------------------------------------------
constexpr double RUNOFF_ABS_EPS_MM  = 0.01;   // mm, physical jitter tolerance
constexpr double RUNOFF_REPAIR_EPS_M = 1e-8;  // m, numeric tolerance for monotone repair

inline RunoffOut compute_runoff_rates_xdiff(
    const std::vector<float>& dense_local,  // [ns * nq * N_EQ], LOCAL cumulatives (meters)
    int ns, int nq, int N_EQ,
    int idx_surf, int idx_total,
    double query_dt_min,                    // e.g., 60
    const char* tag = "RUNOFF_XDIFF",
    bool allow_negative = false
) {
    RunoffOut out;
    const int nbins = std::max(0, nq - 1);
    out.surf_mmhr .assign((size_t)ns * (size_t)nbins, 0.0f);
    out.total_mmhr.assign((size_t)ns * (size_t)nbins, 0.0f);

    if (ns <= 0 || nq <= 1 || N_EQ <= 0) return out;

    const size_t need = (size_t)ns * (size_t)nq * (size_t)N_EQ;
    if (dense_local.size() < need) return out;

    const double dt_hr = query_dt_min / 60.0;
    if (!(dt_hr > 0.0) || !std::isfinite(dt_hr)) return out;

    auto IDX = [&](int s,int t,int k)->size_t {
        return ((size_t)s * (size_t)nq + (size_t)t) * (size_t)N_EQ + (size_t)k;
    };

    double gmin = +1e300, gmax = -1e300;

    for (int s = 0; s < ns; ++s) {
        for (int t = 1; t < nq; ++t) {
            // Consecutive LOCAL cumulatives (meters)
            double a_surf  = (double)dense_local[IDX(s, t-1, idx_surf )];
            double b_surf  = (double)dense_local[IDX(s, t  , idx_surf )];
            double a_total = (double)dense_local[IDX(s, t-1, idx_total)];
            double b_total = (double)dense_local[IDX(s, t  , idx_total)];

            // NaN/Inf guards (hold neighbor)
            if (!std::isfinite(a_surf )) a_surf  = b_surf;
            if (!std::isfinite(a_total)) a_total = b_total;
            if (!std::isfinite(b_surf )) b_surf  = a_surf;
            if (!std::isfinite(b_total)) b_total = a_total;

            // Monotone repair with numeric tolerance:
            // if b is below a by more than RUNOFF_REPAIR_EPS_M, snap b up to a
            if (b_surf + RUNOFF_REPAIR_EPS_M  < a_surf ) b_surf  = a_surf;
            if (b_total + RUNOFF_REPAIR_EPS_M < a_total) b_total = a_total;

            // Delta (mm) for this bin
            double dmm_surf  = (b_surf  - a_surf ) * 1000.0;
            double dmm_total = (b_total - a_total) * 1000.0;

            // Kill tiny jitter (physical tolerance)
            if (std::abs(dmm_surf ) < RUNOFF_ABS_EPS_MM)  dmm_surf  = 0.0;
            if (std::abs(dmm_total) < RUNOFF_ABS_EPS_MM)  dmm_total = 0.0;

            // Optional: no negative rates
            if (!allow_negative) {
                if (dmm_surf  < 0.0) dmm_surf  = 0.0;
                if (dmm_total < 0.0) dmm_total = 0.0;
            }

            const size_t oi = (size_t)s * (size_t)nbins + (size_t)(t - 1);
            const float r_s = (float)(dmm_surf  / dt_hr);
            const float r_t = (float)(dmm_total / dt_hr);

            out.surf_mmhr [oi] = r_s;
            out.total_mmhr[oi] = r_t;

            gmin = std::min(gmin, (double)r_s);
            gmax = std::max(gmax, (double)r_s);
            gmin = std::min(gmin, (double)r_t);
            gmax = std::max(gmax, (double)r_t);
        }
    }

    std::printf("[%s] nbins=%d  rate_min=%.6f mm/hr  rate_max=%.6f mm/hr  (neg=%s)\n",
                tag, nbins, gmin, gmax, allow_negative ? "allowed" : "clamped");
    return out;
}

// Monotone + de-jitter version (xarray.diff semantics: output has nq-1 bins)
inline RunoffOut compute_runoff_debug(
    const std::vector<float>& dense,   // [ns * nq * N_EQ], LOCAL cumulatives in meters
    int ns, int nq, int N_EQ,
    int idx_surf, int idx_total,
    double query_dt_min,               // minutes; e.g., 60
    std::vector<int> preview_systems = {},
    int preview_steps = 6,
    const char* tag = "RUNOFF_DIFF"
) {
    RunoffOut out;
    const int nbins = std::max(0, nq - 1);
    out.surf_mmhr .assign((size_t)ns * (size_t)nbins, 0.0f);
    out.total_mmhr.assign((size_t)ns * (size_t)nbins, 0.0f);

    if (ns <= 0 || nq <= 1 || N_EQ <= 0) {
        std::printf("[%s] Nothing to compute (ns=%d nq=%d N_EQ=%d)\n", tag, ns, nq, N_EQ);
        return out;
    }
    const size_t need = (size_t)ns * (size_t)nq * (size_t)N_EQ;
    if (dense.size() < need) {
        std::fprintf(stderr, "[%s][ERR] dense too small (%zu < %zu)\n", tag, dense.size(), need);
        return out;
    }

    const double dt_hr = query_dt_min / 60.0;
    if (!(dt_hr > 0.0) || !std::isfinite(dt_hr)) {
        std::fprintf(stderr, "[%s][ERR] invalid query_dt_min=%.6f (min)\n", tag, query_dt_min);
        return out;
    }

    // Tunables:
    // tiny absolute delta in mm to treat as 0 (post-diff, pre-division)
    constexpr double ABS_EPS_MM   = 1e-3;   // 0.001 mm
    // tiny relative delta (to magnitude) to treat as 0
    constexpr double REL_EPS      = 1e-7;
    // hard floor on *rates* (mm/hr) to squash residual wiggles
    constexpr double RATE_FLOOR   = 1e-3;   // 0.001 mm/hr

    auto IDX = [&](int s, int t, int k) -> size_t {
        // Layout: ((s * nq + t) * N_EQ + k)
        return ((size_t)s * (size_t)nq + (size_t)t) * (size_t)N_EQ + (size_t)k;
    };

    // Prepare preview list
    if (preview_systems.empty()) {
        if (ns >= 1) preview_systems.push_back(0);
        if (ns >= 2) preview_systems.push_back(ns / 2);
        if (ns >= 3) preview_systems.push_back(ns - 1);
    }
    std::sort(preview_systems.begin(), preview_systems.end());
    preview_systems.erase(std::unique(preview_systems.begin(), preview_systems.end()),
                          preview_systems.end());

    double gmin = +1e300, gmax = -1e300;

    for (int s = 0; s < ns; ++s) {
        for (int t = 0; t < nbins; ++t) {
            // consecutive cumulatives (meters)
            double a_surf  = (double)dense[IDX(s, t,     idx_surf )];
            double b_surf  = (double)dense[IDX(s, t + 1, idx_surf )];
            double a_total = (double)dense[IDX(s, t,     idx_total)];
            double b_total = (double)dense[IDX(s, t + 1, idx_total)];

            // NaN/Inf guards (hold neighbor)
            if (!std::isfinite(a_surf )) a_surf  = b_surf;
            if (!std::isfinite(a_total)) a_total = b_total;
            if (!std::isfinite(b_surf )) b_surf  = a_surf;
            if (!std::isfinite(b_total)) b_total = a_total;

            // Local monotone repair (cumulative cannot decrease)
            if (b_surf  < a_surf )  b_surf  = a_surf;
            if (b_total < a_total)  b_total = a_total;

            // meters → mm deltas for this bin
            double dmm_surf  = (b_surf  - a_surf ) * 1000.0;
            double dmm_total = (b_total - a_total) * 1000.0;

            // Relative + absolute epsilon on the *delta* magnitude
            const double mag_s_mm = std::max({std::abs(a_surf)*1000.0, std::abs(b_surf)*1000.0, 1.0});
            const double mag_t_mm = std::max({std::abs(a_total)*1000.0, std::abs(b_total)*1000.0, 1.0});
            const double eps_s = std::max(ABS_EPS_MM, REL_EPS * mag_s_mm);
            const double eps_t = std::max(ABS_EPS_MM, REL_EPS * mag_t_mm);

            if (std::abs(dmm_surf ) < eps_s) dmm_surf  = 0.0;
            if (std::abs(dmm_total) < eps_t) dmm_total = 0.0;

            // Convert to rates (mm/hr) and apply a hard floor
            const size_t oi = (size_t)s * (size_t)nbins + (size_t)t;
            float r_s = (float)(dmm_surf  / dt_hr);
            float r_t = (float)(dmm_total / dt_hr);

            if (std::abs(r_s) < RATE_FLOOR) r_s = 0.0f;
            if (std::abs(r_t) < RATE_FLOOR) r_t = 0.0f;

            out.surf_mmhr [oi] = r_s;
            out.total_mmhr[oi] = r_t;

            gmin = std::min(gmin, (double)r_s);
            gmin = std::min(gmin, (double)r_t);
            gmax = std::max(gmax, (double)r_s);
            gmax = std::max(gmax, (double)r_t);

            // Light preview
            if (std::binary_search(preview_systems.begin(), preview_systems.end(), s) &&
                t < preview_steps)
            {
                std::printf("[%s] s=%d t=%d  aS=%.9g bS=%.9g ΔS=%.6g mm → %.6g mm/hr | "
                            "aT=%.9g bT=%.9g ΔT=%.6g mm → %.6g mm/hr\n",
                            tag, s, t,
                            a_surf, b_surf, dmm_surf, r_s,
                            a_total, b_total, dmm_total, r_t);
            }
        }
    }

    std::printf("[%s] Summary: rate_min=%.6f mm/hr  rate_max=%.6f mm/hr  (floor=%.3g)\n",
                tag, gmin, gmax, RATE_FLOOR);
    return out;
}


// ───────── Handle outputs from the solver: retrieve results, write NetCDF files
void handleSolverOutputs(const ModelConfig& config, 
                         int simYear,
                         int dayOffset,
                         int daysThisChunk,
                         const SolverInputs& input,
                         const SolverOutputs& output,
                         std::vector<Stream<Runoff5>>& streams,
                         int rank, bool usingMPI) {
    const int N_EQ = Runoff5::N_EQ;
    const int ns   = output.num_systems;
    const int nq   = output.num_queries;

    // Retrieve results from device and free buffers
    auto [h_y_final, h_dense] = retrieve_and_free<Runoff5>(
        output.d_y0_all, output.d_y_final_all,
        output.d_query_times, output.d_dense_all,
        output.d_stiff,
        ns, nq,
        input.t0, input.tf,
        devSpatialParamsPtr
    );

    // ── Seam check: look at the very last interval in this chunk ──
    if (rank==0 && ns>0 && nq>=2) {
        auto IDX = [&](int s,int t,int k)->size_t{
            return (size_t(s)*size_t(nq)+size_t(t))*size_t(N_EQ)+size_t(k);
        };
        const int s  = 0;                // inspect first system
        const int t0 = nq - 2;           // penultimate sample
        const int t1 = nq - 1;           // last sample
        const double dt_last_hr = (input.h_query_times[t1] - input.h_query_times[t0]) / 60.0;
        const double c0 = (double)h_dense[IDX(s,t0,Runoff5::STATE_SURF_RUNOFF)];   // meters (cumulative)
        const double c1 = (double)h_dense[IDX(s,t1,Runoff5::STATE_SURF_RUNOFF)];   // meters (cumulative)

        std::cout << std::setprecision(12)
                << "[SEAM] s=0 last interval: "
                << "t0=" << input.h_query_times[t0]
                << "  t1=" << input.h_query_times[t1]
                << "  dt_hr=" << dt_last_hr
                << "  c0=" << c0 << "  c1=" << c1
                << "  rate(mm/hr)=" << (dt_last_hr>0 ? ((c1-c0)*1000.0/dt_last_hr) : -1.0)
                << "\n";
    }

    // Compute runoff rates (now with t0 included in queries)
    RunoffOut r = compute_runoff_rates_xdiff(
        h_dense, ns, nq, N_EQ,
        Runoff5::STATE_SURF_RUNOFF,
        Runoff5::STATE_TOTAL_RUNOFF,
        GLOBAL_QUERY_DT,
        "RUNOFF_XDIFF",
        false
    );

    // Output has (nq-1) bins aligned to time[1:nq]
    const int nt_hourly = std::max(0, nq - 1);
    const double* hourly_times = (nq > 1) ? (input.h_query_times.data() + 1) : nullptr;


    // Compact sanity stats (few prints only)!!!
    auto vec_minmax = [&](const std::vector<float>& v){
        double mn = +1e300, mx = -1e300, sum = 0.0;
        for (size_t i=0; i<v.size(); ++i) { mn = std::min(mn,(double)v[i]); mx = std::max(mx,(double)v[i]); sum += v[i]; }
        return std::tuple<double,double,double>(mn,mx,sum);
    };


    
    // Helper for indexing dense array: ((s * nq + t) * N_EQ + k)
    auto IDX = [&](int s,int t,int k)->size_t {
        return (size_t(s)*size_t(nq)+size_t(t))*size_t(N_EQ)+size_t(k);
    };

    // Define state indices before using them
    constexpr int IDX_SURF  = Runoff5::STATE_SURF_RUNOFF;
    constexpr int IDX_TOTAL = Runoff5::STATE_TOTAL_RUNOFF;


    // Update stream objects with final y0 state 
    for (int s = 0; s < ns; ++s) {
        for (int i = 0; i < N_EQ; ++i) {
            streams[s].y0[i] = h_y_final[s * N_EQ + i];
        }
    }


    // Format start and end date strings for filenames
    int Y = simYear, M = 1, D = 1;
    advanceDate(Y, M, D, dayOffset);
    std::string sDate = formatDate(Y, M, D);
    advanceDate(Y, M, D, daysThisChunk - 1);
    std::string eDate = formatDate(Y, M, D);

    // Determine rank tag for filenames
    // Append rank only when truly running multi-rank under MPI
    int world_size = 1;
    if (usingMPI) {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }
    std::string rank_tag = (usingMPI && world_size > 1)
                            ? ("_rank" + std::to_string(rank))
                            : ""; // Only append rank if using MPI and more than 1 rank
    // Construct output file paths
    std::string final_file    = config.output_path + "/final_"   + sDate + "_" + eDate + rank_tag + ".nc";
    std::string dense_file    = config.output_path + "/dense_"   + sDate + "_" + eDate + rank_tag + ".nc";
    std::string runoff_file   = config.output_path + "/runoff_"  + sDate + "_" + eDate + rank_tag + ".nc";


    // Prepare metadata arrays
    std::vector<uint32_t> link_ids(ns);
    for (int i = 0; i < ns; ++i) {
        link_ids[i] = streams[i].id;
    }

    std::vector<int> state_ids(N_EQ);
    std::iota(state_ids.begin(), state_ids.end(), 0);

    std::string time_origin = std::to_string(simYear) + "-01-01T00:00:00Z";

    // Write final state to NetCDF (2D)
    if (!config.final_output_file.empty()) {
        logWrite(std::filesystem::path(final_file).filename().string());
        write_final_netcdf(final_file, h_y_final.data(),
                        link_ids.data(), state_ids.data(),
                        ns, N_EQ);
    } else {
        std::cout << "[SKIP] final output disabled via config\n";
    }


    // Write dense time output to NetCDF (3D) 
    if (!config.output_file.empty()) {
        logWrite(std::filesystem::path(dense_file).filename().string());
        write_dense_netcdf(dense_file, h_dense.data(),
                        input.h_query_times.data(),
                        link_ids.data(), state_ids.data(),
                        nq, ns, N_EQ, time_origin);
    } else {
        std::cout << "[SKIP] dense output disabled via config\n";
    }

    // Make hourly arrays now
    std::vector<float> hourly_surf_mmhr  = std::move(r.surf_mmhr);
    std::vector<float> hourly_total_mmhr = std::move(r.total_mmhr);
    if (!config.runoff_output_file.empty()) {
        logWrite(std::filesystem::path(runoff_file).filename().string());
        write_runoff_rates_netcdf(
            runoff_file,
            /*surf_mmhr*/  hourly_surf_mmhr.data(),   // ns * nq
            /*total_mmhr*/ hourly_total_mmhr.data(),  // ns * nq
            /*time_vals*/  hourly_times,              // length nq, endpoints for each bin
            /*link_ids*/   link_ids.data(),
            /*num_systems*/ns,
            /*num_queries*/nt_hourly,                 // nq
            /*origin*/     time_origin
        );
    } 

    //std::cout << "[DONE] Outputs written for " << ns << " systems × " << nq << " time steps\n";
    std::cout << "[DONE] Rank " << rank << ": Outputs written for "
          << ns << " systems × " << nq << " time steps\n";



}


// ───────── Formats a date (YYYY, MM, DD) into a string: "YYYYMMDD" ─────────
std::string formatDate(int Y, int M, int D) {
    char buf[9]; // YYYYMMDD + null terminator
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", Y, M, D);
    return std::string(buf); // Return as std::string
}

// ───────── Advances a calendar date by a number of days ─────────
// Inputs: year (Y), month (M), day (D), number of days to advance
// Updates Y, M, D in-place using UTC-safe conversion
void advanceDate(int &Y, int &M, int &D, int days) {
    std::tm tm = {};
    tm.tm_year = Y - 1900; // std::tm expects years since 1900
    tm.tm_mon  = M - 1;    // months are 0-indexed
    tm.tm_mday = D;

    time_t t = timegm(&tm);   // Convert to UTC time_t
    t += days * 86400;        // Advance by N days (in seconds)

    gmtime_r(&t, &tm);        // Convert back to UTC struct tm

    // Update original date variables
    Y = tm.tm_year + 1900;
    M = tm.tm_mon + 1;
    D = tm.tm_mday;
}


// ───────── Extracts start year (YYYY) from config.time_start ─────────
// Expected format: "YYYY-MM-DDTHH:MM:SSZ" (ISO 8601)
// Only the first 4 digits are used
int startYear(const ModelConfig& config) {
    if (config.time_start.size() < 4)
        throw std::runtime_error("Invalid time_start format in config");
    return std::stoi(config.time_start.substr(0, 4));
}

// ───────── Extracts end year (YYYY) from config.time_end ─────────
// Expected format: "YYYY-MM-DDTHH:MM:SSZ" (ISO 8601)
// Only the first 4 digits are used
int endYear(const ModelConfig& config) {
    if (config.time_end.size() < 4)
        throw std::runtime_error("Invalid time_end format in config");
    return std::stoi(config.time_end.substr(0, 4));
}

inline void printSpatialParams(const SpatialParams& sp, std::ostream& os = std::cout) {
    os << "=== DEBUG: SpatialParams ===\n";
    os << "stream ID     : " << sp.stream << "\n";
    os << "next_stream   : " << sp.next_stream << "\n";
    os << "c1 (m/min per mm/hr): " << sp.c1 << "\n";

    os << "Hu (m)        : " << sp.Hu << "\n";
    os << "infil (m/min) : " << sp.infil << "\n";
    os << "perco (m/min) : " << sp.perco << "\n";
    os << "alpha3 (min)  : " << sp.alpha3 << "\n";
    os << "alpha4 (min)  : " << sp.alpha4 << "\n";

    os << "lat (deg)     : " << sp.lat << "\n";
    os << "lon (deg)     : " << sp.lon << "\n";
    os << "slope (m/m)   : " << sp.slope << "\n";
    os << "n_mann        : " << sp.n_mann << "\n";
    os << "L (m)         : " << sp.L << "\n";
    os << "A_h (m^2)     : " << sp.A_h << "\n";

    os << "sw (unitless) : " << sp.sw << "\n";
    os << "ss (unitless) : " << sp.ss << "\n";

    os << "melt_f (m/min/°C): " << sp.melt_f << "\n";
    os << "temp_thr (°C) : " << sp.temp_thr << "\n";
    os << "============================\n";
}



// ───────── Main function to run the RK45 solver on GPU ─────────
int main(int argc, char** argv) {
    // Validate command-line arguments (expects path to config file)
    if (!validateArgs(argc)) {
        return 1;
    }

    // Load model configuration from YAML file
    ModelConfig config;
    if (!loadConfiguration(argv[1], config)) {
        return 1;
    }

    // ---- Validate GPU memory settings ----
    if (config.max_gpu_mem_gb <= 0.0) {
        throw std::runtime_error("Invalid config: max_gpu_mem_gb must be > 0");
    }
    if (config.gpu_mem_buffer_pct < 0.0 || config.gpu_mem_buffer_pct >= 100.0) {
        throw std::runtime_error("Invalid config: gpu_mem_buffer_pct must be in [0, 100)");
    }

    // Optional warning for suspiciously low memory values
    if (config.max_gpu_mem_gb < 2.0) {
        std::cerr << "[WARN] max_gpu_mem_gb is very low (" 
                << config.max_gpu_mem_gb << " GiB). Check your config.\n";
    }


    if (!config.use_mpi) {
        const char* ompi = std::getenv("OMPI_COMM_WORLD_SIZE");
        const char* pmi  = std::getenv("PMI_SIZE");
        int sz = ompi ? std::atoi(ompi) : (pmi ? std::atoi(pmi) : 1);

        if (sz > 1) {
            std::cerr
                << "\nNote: This job was launched with MPI (" << sz << " processes)\n"
                << "but the configuration has 'flags.use_mpi: false'.\n"
                << "\n"
                << "In this mode, each process will run the full simulation\n"
                << "independently and write to the same output files.\n"
                << "This usually causes duplicated work and may overwrite outputs.\n"
                << "\n"
                << "If you want to run in parallel, set 'flags.use_mpi: true'\n"
                << "in your config. If you want a single serial run, launch with:\n"
                << "    ./bin/runoff config.yaml\n"
                << "or:\n"
                << "    mpirun -np 1 ./bin/runoff config.yaml\n"
                << "\nStopping now to avoid duplicate runs.\n"
                << std::endl;
            return EXIT_FAILURE;
        }
    }



    // Set GLOBAL_QUERY_DT from config, default-safe and clamped
    {
        double qdt = config.query_dt_minutes; // already defaults to 60.0 if key missing
        if (!std::isfinite(qdt) || qdt <= 0.0) {
            std::cout << "[WARN] output.query_dt is missing/invalid (" << qdt
                    << "). Using default 60.0 minutes.\n";
            qdt = 60.0;
        }
        if (qdt > 1440.0) {
            std::cout << "[WARN] output.query_dt (" << qdt
                    << " min) exceeds 1 day. Clamping to 1440.0 minutes.\n";
            qdt = 1440.0;
        }
        GLOBAL_QUERY_DT = qdt;
        std::cout << "[CONFIG] Query output interval = " << GLOBAL_QUERY_DT << " minutes\n";
    }



    // Check if user has enabled MPI via config
    bool usingMPI = config.use_mpi;

    int rank = 0, size = 1;


// Initialize MPI if enabled
if (usingMPI) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

// Assign GPU for this rank (Slurm sets CUDA_VISIBLE_DEVICES; usually 1 visible per rank)
int gpuCount = getGpuCount();
assignGpuDevice(rank, gpuCount);

// Log GPU info and warn if configured memory exceeds physical memory
cudaDeviceProp prop;
int dev;
cudaGetDevice(&dev);
cudaGetDeviceProperties(&prop, dev);

double total_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0*1024.0*1024.0);
if (config.max_gpu_mem_gb > total_gb) {
    std::cerr << "[WARN] Configured max_gpu_mem_gb (" << config.max_gpu_mem_gb
              << " GiB) exceeds physical GPU memory (" << total_gb << " GiB).\n";
}


std::vector<SpatialParams> spatialParams;

if (!usingMPI) {
    // ---- SERIAL: load all, run on one GPU ----
    spatialParams = loadSpatialParams(config.parameters_path + config.spatially_varying_file);
} else {
    // ---- MPI: ALL RANKS COMPUTE (equal split by stream ID) ----
    // 1) Every rank loads the same CSV from shared FS (simple & robust)
    std::vector<SpatialParams> all =
        loadSpatialParams(config.parameters_path + config.spatially_varying_file);

    // 2) Sort once by stream/system ID for clean contiguous partitions
    sort_by_stream(all);

    // 3) Equal whole-number split across 'size' ranks
    auto [s, e] = split_equal(static_cast<int>(all.size()), size, rank);
    spatialParams.assign(all.begin() + s, all.begin() + e);

    std::cout << "[MPI] rank " << rank << "/" << size
              << " handling " << spatialParams.size()
              << " systems (rows " << s << " .. " << (e ? e-1 : 0) << ")\n";
}

    {// ─ Set up GPU solver parameters from config ──
        Runoff5::Parameters hostP;
        // Start with model defaults (match struct defaults)
        hostP.rtol        = 1e-6;
        hostP.atol        = 1e-9;
        hostP.safety      = 0.9;
        hostP.minScale    = 0.2;
        hostP.maxScale    = 10.0;
        hostP.initialStep = 0.01;

        // If user asked to override tolerances, apply them
        if (config.override_tolerances) {
            if (std::isfinite(config.rtol) && config.rtol > 0) hostP.rtol = config.rtol;
            if (std::isfinite(config.atol) && config.atol > 0) hostP.atol = config.atol;
            if (std::isfinite(config.safety) && config.safety > 0) hostP.safety = config.safety;
            if (std::isfinite(config.min_scale) && config.min_scale > 0) hostP.minScale = config.min_scale;
            if (std::isfinite(config.max_scale) && config.max_scale >= hostP.minScale) hostP.maxScale = config.max_scale;
        }

        // If user asked to override initial step, apply it
        if (config.override_initial_step) {
            if (std::isfinite(config.initial_step) && config.initial_step > 0) {
                hostP.initialStep = config.initial_step;
            }
        }

        // Optional: log what we actually use
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(2)
            << "rtol=" << hostP.rtol << " atol=" << hostP.atol
            << std::fixed
            << " safety=" << hostP.safety
            << " minScale=" << hostP.minScale
            << " maxScale=" << hostP.maxScale
            << " initialStep=" << hostP.initialStep
            << (config.override_tolerances ? " [tols:override]" : " [tols:default]")
            << (config.override_initial_step ? " [h0:override]" : " [h0:default]");
        logGpu(std::string("Solver params: ") + oss.str());

        // Push to device
        CUDA_CHECK(cudaMemcpyToSymbol(devParams, &hostP, sizeof(hostP)));
    }

// Build, upload, run simulation
auto streams = buildStreams(spatialParams, config);

// --- Hot start from final.nc if configured ---
if (config.initial_mode == "from_file") {
    std::cout << "[INFO] Loading initial state from file: "
              << config.initial_file << "\n";
    load_initial_from_final_nc_serial(config.initial_file, streams);
}

// Now upload and run
setupGpu(spatialParams, streams);
for (int simYear = startYear(config); simYear <= endYear(config); ++simYear) {
    simulateYear(simYear, config, streams, rank, usingMPI);
}


if (usingMPI) {
    MPI_Finalize();
}
return 0;
}