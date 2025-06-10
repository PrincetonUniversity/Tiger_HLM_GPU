#pragma once
#include <string>

/**
* @file output_netcdf_parallel.cpp
* @brief Writes dense output data to a NetCDF file in parallel using MPI.
*
* This function creates a NetCDF4 file in parallel mode and writes simulation output data
* distributed across multiple MPI ranks. The output data is organized along three dimensions:
* system, time, and variable. Each MPI rank writes its chunk of the output data, while rank 0
* writes the coordinate variables.
*
* @param filename         Path to the NetCDF file to be created.
* @param comm             MPI communicator for parallel I/O.
* @param rank             Rank of the calling MPI process.
* @param num_systems      Number of systems (first dimension). (NEEDS TO BE CALLED LINKS IN FUTURE)
* @param num_queries      Total number of time points (second dimension).
* @param N_EQ             Number of variables (third dimension). (EQUATIONS OR NON-ODE variables)
* @param q_start          Starting index of the time points for this rank.
* @param q_count          Number of time points to be written by this rank.
* @param h_query_times    Vector of time values (length: num_queries).
* @param h_dense_chunk    Chunk of output data to be written by this rank.
* @param compression_level Compression level (0 = no compression, 1-9 = increasing compression).
* @note
* - The function assumes that the data in @p h_dense_chunk is organized in the order:
*   [system][time][variable], contiguous in memory.
* - All ranks must call this function collectively.
* - The NetCDF file is overwritten if it already exists.
* - Requires NetCDF library with parallel I/O support (netcdf_par.h).
* - Requires MPI to be initialized before calling this function.
*/
void write_dense_outputs_parallel(
    const std::string& filename,              // Output NetCDF file name
    MPI_Comm comm,                            // MPI communicator (usually MPI_COMM_WORLD)
    int rank,                                 // Rank of the calling process
    int num_systems,                          // Number of systems
    int num_queries,                          // Total number of time steps
    int N_EQ,                                 // Number of state variables (per system per time)
    int q_start,                              // Starting index in time dimension for this rank
    int q_count,                              // Number of time steps this rank is writing
    const std::vector<double>& h_query_times, // Full time coordinate array (written by rank 0)
    const std::vector<double>& h_dense_chunk, // Local chunk of output data for this rank
    int compression_level                 //Compression level (0 = no compression, 1-9 = increasing compression).
);