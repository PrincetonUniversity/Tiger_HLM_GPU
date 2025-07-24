#pragma once

#include "config_loader.hpp"        // for ModelConfig
#include "stream.hpp"               // for Stream<Model204>
#include "models/model_204.hpp"     // for Model204::N_EQ, etc.
#include <cuda_runtime.h>           // for CUDA runtime API
#include <string>                   // for std::string
#include <vector>                   // for std::vector
#include <memory>                   // for std::shared_ptr
#include <cstdint>                  // for std::uint64_t, std::int64_t
#include <utility>                  // for std::pair
#include <cstring>                  // for memset

// ──────────────────────────────────────────────────────────────────────────────
// Forcing Entry + NCForcing Struct 
//// ────────────────────────────────────────────────────────────────────────────

/**
 * @brief Represents a single forcing input (e.g., temperature, precipitation).
 */
struct ForcingEntry {
    std::string path;       // Full path to the forcing data file
    std::string var_name;   // Variable name in the NetCDF file
    double dt_hr;           // Time step in hours for this forcing variable
};

/**
 * @brief Holds all forcing data for a simulation chunk.
 * 
 * This struct contains multiple ForcingEntry objects, along with
 * pre-allocated vectors for data and metadata needed for the simulation.
 */
struct NCForcing {
    std::vector<ForcingEntry> entries;  // Forcing entries for this chunk
    std::vector<float> h_data;          // Host-side data buffer for all forcings
    std::vector<double> dt;             // Time step for each forcing entry in hours
    std::vector<size_t> nT;             // Number of time steps for each forcing entry
    int nForc = 0;                      // Number of forcing entries loaded
    int days = 0;                       // Number of days in this chunk
    int offset = 0;                     // Day offset from the start of the simulation period
    int systems = 0;                    // Number of systems (e.g., streams) this forcing applies to

    void loadData() {
        nForc = entries.size();
        h_data.clear();
        dt.clear();
        nT.clear();

        for (const auto& entry : entries) {
            double dt_hr = entry.dt_hr;
            dt.push_back(dt_hr);

            size_t steps_per_day = static_cast<size_t>(24.0 / dt_hr);
            size_t total_steps = steps_per_day * days;
            nT.push_back(total_steps);

            // For each timestep, push data for all systems
            for (size_t t = 0; t < total_steps; ++t) {
                for (int s = 0; s < systems; ++s) {
                    h_data.push_back(1.0f);  // Placeholder
                }
            }
        }
    }

};

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
                          int num_systems);             // Number of systems (e.g., streams) this forcing applies to

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
    double* d_dense_all;    // Device pointer to dense output (flattened)
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
                                 const std::vector<Stream<Model204>>& streams); // Streams to simulate

/**
 * @brief Launches the solver on the GPU.
 */
SolverOutputs launchSolverKernel(const SolverInputs& input);


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
                         std::vector<Stream<Model204>>& streams);   // Streams to update with final states and dense outputs


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
 */
int computeDaysPerChunk(int num_systems);

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
 * @brief Parses a date string in the format "YYYY-MM-DD" into a std::tm structure.
 *
 * @param date_str Date string to parse.
 * @return Parsed std::tm structure.
 */
void simulateChunk(const ModelConfig& config,
                   std::vector<Stream<Model204>>& streams,
                   int simYear, int dayOffset, int daysThisChunk);

