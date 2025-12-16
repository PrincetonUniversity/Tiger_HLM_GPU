#pragma once

#include <cuda_runtime.h>           
#include <string>                   
#include <vector>                  
#include <memory>                 
#include <cstdint>                  
#include <utility>                 
#include <cstring>                 
#include "I_O/forcing_loader.hpp"

#include "I_O/config_loader.hpp"        // for ModelConfig
#include "stream.hpp"                   // for Stream<Runoff5>
#include "models/model_Runoff5.hpp"     // for Runoff5::N_EQ, etc.

// ──────────────────────────────────────────────────────────────────────────────
// Forcing Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * @brief Loads and prepares forcing data for a simulation chunk.
 */
NCForcing loadForcingData(const ModelConfig& config,    // Simulation model configuration
                          int simYear,                  // Simulation year
                          int dayOffset,                // Offset in days from the start of the simulation
                          int daysThisChunk,            // Number of days in this chunk
                          int num_systems,              // Number of systems (e.g., streams) to simulate
                          const std::vector<Stream<Runoff5>>& streams);  // Number of systems (e.g., streams) this forcing applies to

/**
 * @brief Uploads host-side forcing data to GPU memory.
 */
void uploadForcingsToGpu(NCForcing& forcing); 

// ──────────────────────────────────────────────────────────────────────────────
// Solver Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * @brief Inputs required by the solver kernel to run on the GPU.
 */
struct SolverInputs {
    double t0, tf;                      // Simulation time bounds (start and end)
    std::vector<double> h_y0;           // Initial conditions for all systems (flattened)
    std::vector<double> h_query_times;  // Query times for output (flattened)
};

/**
 * @brief Outputs returned from the GPU after simulation is complete.
 */
struct SolverOutputs {
    double* d_y0_all;       // Device pointer to initial conditions (flattened)
    double* d_y_final_all;  // Device pointer to final states (flattened)
    double* d_query_times;  // Device pointer to query times (flattened)
    float* d_dense_all;    // Device pointer to dense output (flattened)
    int* d_stiff;           // Device pointer to stiffness flags (1 if stiff, 0 if not)
    int num_systems;        // Number of systems (e.g., streams) processed
    int num_queries;        // Number of query times (e.g., hourly outputs)
};

/**
 * @brief Prepares solver input structures for a given simulation chunk.
 */
SolverInputs prepareSolverInputs(int simYear,                                   // Simulation year
                                 int dayOffset,                                 // Offset in days from the start of the simulation
                                 int daysThisChunk,                             // Number of days in this chunk
                                 const std::vector<Stream<Runoff5>>& streams,   // Streams to simulate
                                 bool include_tf);                              // Whether to include final time in queries
/**
 * @brief Launches the solver on the GPU.
 */
SolverOutputs launchSolverKernel(const SolverInputs& input, int nForc, const std::vector<Stream<Runoff5>>& streams); // Solver inputs

/**
 * @brief Transfers and post-processes the solver outputs.
 *
 * Updates system streams with final states and logs dense outputs.
 */
void handleSolverOutputs(const ModelConfig& config,                 // Simulation model configuration
                         int simYear,                               // Simulation year
                         int dayOffset,                             // Offset in days from the start of the simulation
                         int daysThisChunk,                         // Number of days in this chunk
                         const SolverInputs& input,                 // Solver inputs                                                       
                         const SolverOutputs& output,               // Solver outputs
                         std::vector<Stream<Runoff5>>& streams,      // Streams to update with final states
                         int rank = 0, bool usingMPI = false,      // Rank of the MPI process (default 0, for single-process runs)
                         bool is_last_chunk_of_year = false);       // Only write final.nc at year end if enabled


/**
 * @brief Copies final and dense simulation results from device to host and frees GPU memory.
 *
 * @tparam T               Model type (must have static N_EQ field).
 * @param d_y_final        Device pointer to final state data.
 * @param d_dense          Device pointer to dense output data.
 * @param num_systems      Number of systems.
 * @param num_queries      Number of dense output queries.
 * @return std::pair       Host-side vectors of final state and dense output.
 */

template <typename T>
std::pair<std::vector<double>, std::vector<double>> retrieve_and_free(
    double* d_y_final, double* d_dense, int num_systems, int num_queries) {

    size_t size_y = num_systems * T::N_EQ;
    size_t size_d = num_systems * num_queries * T::N_EQ;

    std::vector<double> h_y_final(size_y);
    std::vector<double> h_dense(size_d);

    cudaMemcpy(h_y_final.data(), d_y_final, size_y * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dense.data(), d_dense, size_d * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_y_final);
    cudaFree(d_dense);

    return {h_y_final, h_dense};
}

// ──────────────────────────────────────────────────────────────────────────────
// Simulation Utilities
// ──────────────────────────────────────────────────────────────────────────────
/**
 * @brief Computes the number of days per chunk based on the number of systems and memory limits.
 *
 * This function determines how many days can be simulated in a single chunk
 * based on the maximum allowed memory usage and the number of systems.
 *
 * @param num_systems Number of systems to simulate.
 * @return           Number of days per chunk.
 * @param config       Simulation configuration (contains memory limits).
 */
int computeDaysPerChunk(int num_systems, const ModelConfig& config);

/**
 * @brief Advances the given date by a specified number of days.
 *
 * @param year     Reference to current year.
 * @param month    Reference to current month.
 * @param day      Reference to current day.
 * @param num_days Number of days to advance.
 */
void advanceDate(int& year, int& month, int& day, int num_days);

/**
 * @brief Formats a date into a string in the format "YYYYMMDD".
 *
 * @param year  Year as an integer.
 * @param month Month as an integer (1-12).
 * @param day   Day as an integer (1-31).
 * @return      Formatted date string.
 */
std::string formatDate(int year, int month, int day);

/**
 * @brief Simulate a contiguous window ("chunk") of days for all streams on the GPU.
 *
 * Orchestrates one end-to-end chunk execution:
 *  1) Loads and maps external forcings for the window [simYear, dayOffset, daysThisChunk]
 *     using the configured forcing files and the stream→lat/lon mapper.
 *  2) Uploads forcing arrays and metadata to device symbols
 *     (d_forc_data, nForc, c_forc_dt, c_forc_nT).
 *  3) Prepares solver inputs (absolute minutes t0→tf, flattened y0, query times).
 *  4) Launches the RK45 kernel (with in-kernel access to forcings), times execution,
 *     and retrieves dense/final outputs.
 *  5) Writes NetCDF files (final_, dense_, runoff_) for the chunk and updates each
 *     stream’s y0 with the final state to seed the next chunk.
 *
 * Logging/Debug:
 *  - Prints the chunk date range and basic spatial params for system 0.
 *  - Optionally checks device forcing layout via checkForcingsOnDeviceSafe<<<1,1>>>.
 *  - Emits timing for solver and output writing.
 *
 * Assumptions:
 *  - Time units inside the solver are minutes since Jan 1 of simYear
 *    (t0 = dayOffset * 24 * 60; tf = t0 + daysThisChunk * 24 * 60).
 *  - Forcings are laid out on device as a single 1D array with blocks per forcing:
 *        [ forcing f ][ timestep t ][ system s ]
 *    and indexed via a prefix-sum over c_forc_nT to handle per-forcing lengths.
 *  - streams.size() == num_systems and each Stream<Runoff5> holds valid SpatialParams.
 *
 * Notes:
 *  - Updates device globals that the kernel reads (d_forc_data, nForc, c_forc_dt, c_forc_nT).
 *  - Writes NetCDF outputs under config.output_path.
 *  - Mutates `streams[*].y0` to the final state for continuity across chunks.
 *
 * @param config        Simulation configuration (paths, tolerances, output settings).
 * @param streams       Vector of stream systems to simulate (provides y0 & SpatialParams).
 * @param simYear       Four-digit year for this chunk (used for absolute time origin & filenames).
 * @param dayOffset     Zero-based day index within simYear where the chunk starts.
 * @param daysThisChunk Number of days to simulate in this chunk (>= 1).
 *
 * @throws std::runtime_error if forcing mapping/files cannot be loaded or if
 *         no forcing data is available for the requested window.
 *
 * @note This function does not return a value; results are written to disk and the
 *       in-memory streams are advanced to the chunk’s final state.
 */
void simulateChunk(const ModelConfig& config,
                   std::vector<Stream<Runoff5>>& streams,
                   int simYear, int dayOffset, int daysThisChunk,
                   int rank = 0, bool usingMPI = false,
                   bool include_tf = false,
                   bool is_last_chunk_of_year = false);

