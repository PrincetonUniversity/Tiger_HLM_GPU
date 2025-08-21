// main.cpp
// Standard Library Headers
#include <cstdio>       // for printf, fprintf
#include <cstdlib>      // for std::exit
#include <cmath>        // for std::sqrt, std::pow, std::fabs
#include <algorithm>    // for std::max, std::fmin, std::fmax
#include <numeric>      // for std::iota
#include <vector>       // for std::vector
#include <array>        // for std::array
#include <fstream>      // for std::ifstream, std::ofstream
#include <iomanip>      // for std::setw, std::setprecision
#include <utility>      // for std::tie
#include <cstdint>      // for std::uint64_t, std::int64_t
#include <stdexcept>    // for std::runtime_error
#include <iostream>     // for std::cout, std::cerr
#include <ctime>        // for std::tm, timegm
#include <filesystem>   // for std::filesystem::path (used to get filename)

// CUDA & GPU Headers
#include <cuda_runtime.h>           // CUDA runtime API

// Project Core Headers
#include "solver/rk45.h"                  // Runge-Kutta 4/5 core solver interface (host side)
#include "solver/rk45_api.hpp"     // Host-side RK45 API: setup_gpu_buffers, launch_rk45_kernel, etc.
#include "solver/rk45_step_dense.cuh"     // Device kernel for one RK45 step + dense output
#include "solver/radau_step_dense.cuh"    // Device kernel for Radau-IIA step + dense output
#include "solver/radau_kernel.cuh"        // Radau-only kernel for specific model (e.g. Runoff5)
#include "solver/small_lu.cuh"            // Small-matrix LU solver used by Radau (for implicit steps)
#include "solver/event_detector.cuh"      // Event detection (e.g., slope jumps, stiffness) on device
#include "simulation_driver.hpp"   // Simulation core loop (manages time stepping, control flow)

// Model & Parameters
#include "models/model_Runoff5.hpp"           // Brings in SpatialParams and model-specific logic
#include "models/model_registry.hpp"      // Registers models and parameters
#include "stream.hpp"                     // Stream wrapper for simulation (manages model ID, initial state, etc.)
#include "I_O/config_loader.hpp"              // Loads model configuration settings (e.g., from JSON)
#include "I_O/parameters_loader.hpp"          // Loads parameters (e.g., spatial or physical) from CSV

// Forcing & Input/Output
#include "I_O/forcing_loader.hpp"  // Loads external forcing data from NetCDF and maps it
#include "I_O/forcing_data.h"      // Provides forcing data structure
#include "I_O/output_series.hpp"       // Serial output to NetCDF format

// MPI & Timing Utilities
#include <mpi.h>                   // MPI for parallel communication
#include "chrono"                  // Timing utility (possibly custom or wrapper around std::chrono)

// tell the compiler/linker about our new 2-arg solver-launch function
SolverOutputs launchSolverKernel(const SolverInputs& input, int nForc);

// ───────── Global Variables ─────────
double GLOBAL_QUERY_DT = 60.0;   // default output interval (minutes)

// Bring in RK45 API functions
using rk45_api::setup_gpu_buffers;   
using rk45_api::retrieve_and_free; 

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

// Parses ISO date string like "2019-12-25" or "2019-12-25T00:00:00Z"
std::tm parseDate(const std::string& iso_date) {
    std::tm t{};
    sscanf(iso_date.c_str(), "%4d-%2d-%2d", &t.tm_year, &t.tm_mon, &t.tm_mday);
    t.tm_year -= 1900;
    t.tm_mon -= 1;
    return t;
}

// Days in month for (Y, M)
// int daysInMonth(int Y, int M) {
//     static const int dm[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
//     if (M == 2) return isLeapYear(Y) ? 29 : 28;
//     return dm[M-1];
// }

// // Convert (year + 0-based day-of-year offset) -> (Y,M,D)
// struct YMD { int Y, M, D; };
// YMD ymdFromYearAndDayOffset(int year, int dayOffset) {
//     int Y = year, M = 1, D = 1;
//     advanceDate(Y, M, D, dayOffset);
//     return {Y, M, D};
// }

// Compute days since Jan 1 of that year
int computeDayOfYear(const std::tm& t) {
    std::tm jan1 = t;
    jan1.tm_mon = 0;
    jan1.tm_mday = 1;
    time_t start = timegm(&jan1);
    time_t target = timegm(const_cast<std::tm*>(&t));
    return static_cast<int>((target - start) / 86400);
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
    printf("---------------------------------\n");
}

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

// Type alias for time points
double elapsedSeconds(TimePoint start, TimePoint end) {
    return std::chrono::duration<double>(end - start).count();
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
    // Log general information messages
    std::cout << "[INFO]  " << msg << "\n";
}

void logGpu(const std::string& msg) {
    //  // Log GPU-specific messages
    std::cout << "[GPU]   " << msg << "\n";
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
    // Log that outputs have been written
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
std::vector<Stream<Runoff5>> buildStreams(const std::vector<SpatialParams>& params) {
    // Define a common initial state y0 for all streams (9-variable model)
    std::array<double, Runoff5::N_EQ> y0_common = {0.01, 0.1, 0.0, 0.0, 0.01, 1, 1, 0, 0};

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

// ───────── Determines if a given year is a leap year ─────────
// Leap year rule: divisible by 4, but not by 100 unless also divisible by 400
bool isLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}


// ───────── Loop over a single simulation year ─────────
void simulateYear(int year, const ModelConfig& config, std::vector<Stream<Runoff5>>& streams) {
    
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

    // int DAYS_PER_CHUNK = computeDaysPerChunk(num_systems);
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
        DAYS_PER_CHUNK = computeDaysPerChunk(num_systems);
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

        simulateChunk(config, streams, year, dayOffset, daysThisChunk);

        // Advance date pointer for next chunk
        advanceDate(chunkStartYear, chunkStartMon, chunkStartDay, daysThisChunk);
    }

}


// void simulateYear(int year, const ModelConfig& config, std::vector<Stream<Runoff5>>& streams) {
//     const int TOTAL_DAYS = isLeapYear(year) ? 366 : 365;
//     const int num_systems = static_cast<int>(streams.size());

//     // Parse full start and end dates
//     std::tm start_tm = parseDate(config.time_start);
//     std::tm end_tm   = parseDate(config.time_end);

//     // Only simulate if this year overlaps with config window
//     if (start_tm.tm_year + 1900 > year || end_tm.tm_year + 1900 < year)
//         return;

//     // Determine bounds within this year
//     int start_day = (start_tm.tm_year + 1900 == year) ? computeDayOfYear(start_tm) : 0;
//     int end_day   = (end_tm.tm_year + 1900 == year)   ? computeDayOfYear(end_tm)   : (TOTAL_DAYS - 1);

//     printSimHeader(year, start_day, end_day, num_systems);

//     // Memory cap in days based on 15 GiB limit
//     const int memCapDays = std::max(1, computeDaysPerChunk(num_systems));

//     // Optional user cap in days from YAML
//     int userCapDays = -1;
//     for (const auto& f : config.forcing_variables) {
//         if (f.time_chunk_size > 0) {
//             userCapDays = f.time_chunk_size;
//             break;
//         }
//     }

//     // Detect if we are in monthly file mode
//     bool monthlyFiles = false;
//     for (const auto& f : config.forcing_variables) {
//         if (f.file.find("{month}") != std::string::npos) {
//             monthlyFiles = true;
//             break;
//         }
//     }

//     // Loop over chunks
//     for (int dayOffset = start_day; dayOffset <= end_day; ) {
//         int remainingDays = end_day - dayOffset + 1;
//         int daysThisChunk = remainingDays;

//         if (monthlyFiles) {
//             auto [Y, M, D] = ymdFromYearAndDayOffset(year, dayOffset);
//             const int dim           = daysInMonth(Y, M);
//             const int dayInMonth0   = D - 1;             // 0-based index
//             const int daysLeftMonth = dim - dayInMonth0; // incl. today
//             daysThisChunk = std::min(daysThisChunk, daysLeftMonth);
//         }

//         // Apply memory and user caps
//         daysThisChunk = std::min(daysThisChunk, memCapDays);
//         if (userCapDays > 0) {
//             daysThisChunk = std::min(daysThisChunk, userCapDays);
//         }

//         simulateChunk(config, streams, year, dayOffset, daysThisChunk);
//         dayOffset += daysThisChunk;
//     }
// }

// ───────── Dynamically compute how many days per chunk fit in memory ─────────
int computeDaysPerChunk(int num_systems) {
    constexpr std::uint64_t MAX_BYTES = 15ULL * 1024 * 1024 * 1024; // 15 GiB limit
    int N_EQ = Runoff5::N_EQ;
    double max_chunk_days = (double(MAX_BYTES) / (8.0 * N_EQ * num_systems) - 1.0) / 24.0; // double
    int DAYS_PER_CHUNK = std::max(1, int(std::floor(max_chunk_days)));

    return DAYS_PER_CHUNK;
}

// ───────── Simulate a chunk of days within a year ─────────
void simulateChunk(const ModelConfig& config,
                   std::vector<Stream<Runoff5>>& streams,
                   int simYear,
                   int dayOffset,
                   int daysThisChunk) {
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
    SolverInputs solverInputs = prepareSolverInputs(simYear, dayOffset, daysThisChunk, streams);

    // ───────── Launch ODE solver on GPU ─────────
    TimePoint t_solver_start = Clock::now();
    std::cout << "[HOST DEBUG] Launching kernel with nForc=" << forcingChunk.nForc << std::endl;
    SolverOutputs solverOutputs = launchSolverKernel(solverInputs, forcingChunk.nForc);
    TimePoint t_solver_end = Clock::now();
    logTimer("Solver runtime", elapsedSeconds(t_solver_start, t_solver_end));


    // ───────── Retrieve results and handle outputs ─────────
    TimePoint t_output_start = Clock::now();
    handleSolverOutputs(config, simYear, dayOffset, daysThisChunk, solverInputs, solverOutputs, streams);
    TimePoint t_output_end = Clock::now();
    std::cout << "[TIMER] Output writing took "
            << elapsedSeconds(t_output_start, t_output_end) << " seconds\n";
}

// ───────── Load forcing data from NetCDF and map to streams ─────────
NCForcing loadForcingData(const ModelConfig& config,
                          int simYear,
                          int dayOffset,
                          int daysThisChunk,
                          int num_systems,
                          const std::vector<Stream<Runoff5>>& streams) {
    std::vector<ForcingEntry> forcings;
    for (const auto& f : config.forcing_variables) {
        double dt_hr = (f.time_resolution == "1h") ? 1.0 :
                       (f.time_resolution == "24h") ? 24.0 :
                       throw std::runtime_error("Unsupported resolution: " + f.time_resolution);

        std::string file = f.file;
        size_t pos = file.find("{year}");
        if (pos != std::string::npos)
            file.replace(pos, 6, std::to_string(simYear));

        forcings.push_back({ config.forcings_path + file, f.var_name, dt_hr });
    }

    NCForcing chunk;
    chunk.entries = std::move(forcings);
    chunk.days    = daysThisChunk;
    chunk.offset  = dayOffset;
    chunk.systems = num_systems;

    // Initialize mapper
    std::string mapperFile = config.forcing_mappings_path + "/" + config.forcing_mappings[0].file;
    chunk.mapper = LookupMapper(mapperFile);

    if (!chunk.mapper.load()) {
        throw std::runtime_error("Failed to load stream → lat/lon mapping");
    }

    // Fill system IDs
    chunk.systemIds.resize(num_systems);
    for (size_t i = 0; i < num_systems; ++i) {
        chunk.systemIds[i] = streams[i].sp.stream; // assuming SpatialParams.stream is stream ID
    }

    return chunk;
}


// NCForcing loadForcingData(const ModelConfig& config,
//                           int simYear,
//                           int dayOffset,
//                           int daysThisChunk,
//                           int num_systems,
//                           const std::vector<Stream<Runoff5>>& streams) {
//     // Decide if monthly mode
//     bool monthly = false;
//     if (!config.forcing_variables.empty()) {
//         monthly = (config.forcing_variables[0].file.find("{month}") != std::string::npos);
//     }

//     // Calendar date of this chunk's first day
//     int Y = simYear, M = 1, D = 1;
//     if (monthly) {
//         auto ymd = ymdFromYearAndDayOffset(simYear, dayOffset);
//         Y = ymd.Y;
//         M = ymd.M;
//         D = ymd.D;
//     }

//     std::vector<ForcingEntry> forcings;
//     forcings.reserve(config.forcing_variables.size());
//     for (const auto& f : config.forcing_variables) {
//         double dt_hr = (f.time_resolution == "1h") ? 1.0 :
//                        (f.time_resolution == "24h") ? 24.0 :
//                        throw std::runtime_error("Unsupported resolution: " + f.time_resolution);

//         std::string file = f.file;

//         // Replace placeholders
//         if (auto pos = file.find("{year}"); pos != std::string::npos) {
//             file.replace(pos, 6, std::to_string(Y));
//         }
//         if (monthly) {
//             if (auto pos = file.find("{month}"); pos != std::string::npos) {
//                 char mm[3];
//                 std::snprintf(mm, sizeof(mm), "%02d", M);
//                 file.replace(pos, 7, mm); // zero-padded month
//             }
//         }

//         forcings.push_back({ config.forcings_path + file, f.var_name, dt_hr });
//     }

//     NCForcing chunk;
//     chunk.entries = std::move(forcings);
//     chunk.days    = daysThisChunk;
//     chunk.offset  = monthly ? (D - 1) : dayOffset; // day-of-month vs day-of-year
//     chunk.systems = num_systems;

//     // Initialize mapper
//     std::string mapperFile = config.forcing_mappings_path + "/" + config.forcing_mappings[0].file;
//     chunk.mapper = LookupMapper(mapperFile);
//     if (!chunk.mapper.load()) {
//         throw std::runtime_error("Failed to load stream → lat/lon mapping");
//     }

//     // Fill system IDs
//     chunk.systemIds.resize(num_systems);
//     for (size_t i = 0; i < num_systems; ++i) {
//         chunk.systemIds[i] = streams[i].sp.stream;
//     }

//     return chunk;
// }



// ───────── Upload forcing data to device and copy pointer + metadata to device symbols
void uploadForcingsToGpu(NCForcing& chunk) {
    // Load the actual values from NetCDF using the paths in NCForcing
    chunk.loadData();  // Assume this allocates & populates: chunk.h_data
    if (chunk.h_data.empty()) {
        throw std::runtime_error("No forcing data loaded for this chunk");
    }


    float* d_ptr = nullptr;
    size_t bytes = sizeof(float) * chunk.h_data.size();

    // Allocate GPU buffer and copy
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, chunk.h_data.data(), bytes, cudaMemcpyHostToDevice));

    // Copy the new chunk’s pointer + metadata into device globals:
    CUDA_CHECK(cudaMemcpyToSymbol(d_forc_data, &d_ptr, sizeof(float*)));
    CUDA_CHECK(cudaMemcpyToSymbol(nForc, &chunk.nForc, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_forc_dt, chunk.dt.data(), sizeof(double) * chunk.nForc));
    CUDA_CHECK(cudaMemcpyToSymbol(c_forc_nT, chunk.nT.data(), sizeof(size_t) * chunk.nForc));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0));
    logInfo("Forcing uploaded to GPU (" + oss.str() + " MiB)");

    // // ───────── DEBUG: Print forcing data on host ─────────
    // std::cout << "[HOST DEBUG] Forcing count chunk.nForc=" << chunk.nForc << std::endl;
    // for (int f = 0; f < chunk.nForc; ++f) {
    //     size_t base0 = 0;
    //     for (int k = 0; k < f; ++k) base0 += chunk.nT[k] * chunk.systems;
    //     std::cout << "   forcing[" << f << "] dt=" << chunk.dt[f]
    //             << " nT=" << chunk.nT[f]
    //             << " first=" << chunk.h_data[base0 + 0 * chunk.systems + 0] << "\n";
    // }


    


}

// ───────── Setup solver time bounds, initial conditions, and query times
SolverInputs prepareSolverInputs(int simYear,
                                  int dayOffset,
                                  int daysThisChunk,
                                  const std::vector<Stream<Runoff5>>& streams) {
    SolverInputs input;
    int N_EQ = Runoff5::N_EQ;
    int num_systems = streams.size();

    // Absolute simulation times
    input.t0 = dayOffset * 24.0 * 60.0;
    input.tf = input.t0 + daysThisChunk * 24.0 * 60.0;

    // Flatten initial conditions
    input.h_y0.resize(num_systems * N_EQ);
    for (int s = 0; s < num_systems; ++s) {
        for (int i = 0; i < N_EQ; ++i) {
            input.h_y0[s * N_EQ + i] = streams[s].y0[i];
        }
    }


    // Define query times using GLOBAL_QUERY_DT
    for (double m = input.t0; m <= input.tf; m += GLOBAL_QUERY_DT)
        input.h_query_times.push_back(m);


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


    // CUDA_CHECK(cudaMemset(out.d_dense_all, 0xAB,
    //                   num_systems * Runoff5::N_EQ * num_queries * sizeof(double)));

    // Compute dense buffer size safely (use size_t to avoid 32-bit overflow)
    const size_t dense_elems =
        size_t(out.num_systems) * size_t(Runoff5::N_EQ) * size_t(out.num_queries);
    const size_t dense_bytes = dense_elems * sizeof(double);

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
    //     d_forc_data,                        // Forcing data
    // nForc                                   // Number of forcings
    );

    CUDA_CHECK( cudaPeekAtLastError() );     // reports launch-time errors
    CUDA_CHECK( cudaDeviceSynchronize() );   // catches run-time errors

    return out;
}

// ───────── Handle outputs from the solver: retrieve results, write NetCDF files
void handleSolverOutputs(const ModelConfig& config, 
                         int simYear,
                         int dayOffset,
                         int daysThisChunk,
                         const SolverInputs& input,
                         const SolverOutputs& output,
                         std::vector<Stream<Runoff5>>& streams) {
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

    // ───── DEBUG: Print first 5 time steps for system s=0 ─────
    // std::cout << "[DEBUG STATE] First 5 time steps for s=0:\n";
    // for (int t = 0; t < std::min(5, nq); ++t) {
    //     std::cout << "  t=" << t << " → ";
    //     for (int i = 0; i < N_EQ; ++i) {
    //         double val = h_dense[(0 * nq + t) * N_EQ + i]; // correct index
    //         std::cout << val << (i < N_EQ - 1 ? ", " : "");
    //     }
    //     std::cout << "\n";
    // }

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
    // std::string prefix = std::to_string(simYear) + "_";

    std::string final_file    = config.output_path + "/final_"   + sDate + "_" + eDate + ".nc";
    std::string dense_file    = config.output_path + "/dense_"   + sDate + "_" + eDate + ".nc";
    std::string runoff_file   = config.output_path + "/runoff_"  + sDate + "_" + eDate + ".nc";

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


    // Write selected runoff states
    if (!config.runoff_output_file.empty()) {
        logWrite(std::filesystem::path(runoff_file).filename().string());
        write_runoff_dense_netcdf(runoff_file,
                                h_dense.data(),
                                input.h_query_times.data(),
                                link_ids.data(),
                                nq, ns,
                                time_origin);
    } else {
        std::cout << "[SKIP] runoff output disabled via config\n";
    }

    



    std::cout << "[DONE] Outputs written for " << ns << " systems × " << nq << " time steps\n";

    // ───── DEBUG: Final state for s=0 after the chunk ─────
    // std::cout << "[DEBUG OUTPUT] Final state for s=0 after last time step:\n";
    // for (int i = 0; i < N_EQ; ++i) {
    //     std::cout << "  y_final[" << i << "] = " << h_y_final[i] << std::endl;
    // }

    // ───── DEBUG Output: Inspecting Model State Variables for First System Across Initial Query Times ─────
    // { 
    //     const int SYS  = 0;                 // first system
    //     const int N_EQ = Runoff5::N_EQ;
    //     const int nq   = output.num_queries;
    //     const int max_q = std::min(6, nq);  // q = 0..5 inclusive

    //     static const char* STATE_NAMES[Runoff5::N_EQ] = {
    //         "SNOW_STORAGE",
    //         "STATIC_STORAGE",
    //         "SURFACE_STORAGE",
    //         "GRAVITATIONAL_STORAGE",
    //         "AQUIFER_STORAGE",
    //         "TEMP_AIR",
    //         "TEMP_SOIL",
    //         "SURFACE_RUNOFF",
    //         "TOTAL_RUNOFF"
    //     };

    //     std::cout << "[CHECK] States for system " << SYS
    //             << " at q=0.." << (max_q - 1) << "\n";

    //     for (int q = 0; q < max_q; ++q) {
    //         double tq = input.h_query_times[q]; // minutes since chunk start
    //         std::cout << "  q=" << q << "  t=" << tq << " min\n";
    //         for (int c = 0; c < N_EQ; ++c) {
    //             double val = h_dense[(SYS * nq + q) * N_EQ + c]; // [sys][q][comp]
    //             std::cout << "    [" << c << "] " << STATE_NAMES[c]
    //                     << " = " << val << "\n";
    //         }
    //     }
    // }



}


// ───────── Formats a date (YYYY, MM, DD) into a string: "YYYYMMDD" ─────────
// Used for naming output files with date ranges
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


    {// ─ Set up GPU solver parameters from config ──
        Runoff5::Parameters hostP;
        hostP.rtol        = config.rtol;
        hostP.atol        = config.atol;
        hostP.safety      = config.safety;
        hostP.minScale    = config.min_scale;
        hostP.maxScale    = config.max_scale;
        hostP.initialStep = config.initial_step;

        // push into GPU constant memory
        CUDA_CHECK(cudaMemcpyToSymbol(devParams, &hostP, sizeof(hostP)));
    }

    GLOBAL_QUERY_DT = config.query_dt_minutes;   // set from YAML
    if (GLOBAL_QUERY_DT > 60.0) {
        std::cout << "[WARN] query_dt in config (" << GLOBAL_QUERY_DT 
                << " min) exceeds max allowed (60 min). Using 60.\n";
        GLOBAL_QUERY_DT = 60.0;
    }


    // Check if user has enabled MPI via config
    bool usingMPI = config.use_mpi;

    int rank = 0, size = 1;

    // Initialize MPI if enabled
    if (usingMPI) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get this rank's ID
        MPI_Comm_size(MPI_COMM_WORLD, &size);  // Total number of ranks
    }

    // Assign GPU device based on rank (round-robin if multiple ranks)
    int gpuCount = getGpuCount();
    assignGpuDevice(rank, gpuCount);

    std::vector<SpatialParams> spatialParams;

    // Handle size==1 with use_mpi: true (treat like serial)
    if (usingMPI && size == 1) {
        std::cout << "[INFO] MPI requested but only 1 process detected; "
                    "running in single-rank mode.\n";
        usingMPI = false; // fall through to serial path below
    }

    
    if (usingMPI && size > 1 && rank == 0) { 
        // Root rank loads and distributes SpatialParams to workers
        std::cout << "[RANK 0] Acting as coordinator only\n";
        distributeSpatialParams(config, size);
        // Early exit: rank 0 is just a coordinator/distributor
        MPI_Finalize();
        return 0;
    }

    if (usingMPI && rank >= 1 && rank < size) {
        // Non-root ranks receive their assigned SpatialParams
        spatialParams = receiveSpatialParams();
    } else if (!usingMPI) {
        // Serial mode: load all parameters directly
        spatialParams = loadSpatialParams(config.parameters_path + config.spatially_varying_file);
    }

    if (!config.constant_parameters_index.empty()) {
        for (auto &sp : spatialParams) {
            for (size_t i = 0; i < config.constant_parameters_index.size(); ++i) {
                int idx = config.constant_parameters_index[i];
                double v = config.constant_parameters_values[i];

                switch (idx) {
                    case 0:  sp.c1     = v;  break;  // example: override c1
                    case 1:  sp.Hu     = v;  break;  // override Hu
                    case 2:  sp.infil  = v;  break;  // override infiltration rate
                    case 3:  sp.perco  = v;  break;  // override percolation rate
                    // … add further cases to map index → the correct SpatialParams field …
                    default:
                        std::cerr << "Warning: constant_param idx="<<idx
                                <<" not recognized\n";
                }
            }
        }
    }

    // Build stream objects for each spatial unit (each ODE system)
    auto streams = buildStreams(spatialParams);

    // Upload SpatialParams to GPU and set global device pointer
    setupGpu(spatialParams, streams);

    // Run simulations year by year
    for (int simYear = startYear(config); simYear <= endYear(config); ++simYear) {
        simulateYear(simYear, config, streams);
    }

    // Finalize MPI (if used)
    if (usingMPI) {
        MPI_Finalize();
    }

    return 0;
}

