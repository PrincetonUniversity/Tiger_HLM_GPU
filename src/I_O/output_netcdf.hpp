#pragma once
#include <string>

/**
 * @brief Write a dense 3D array to a NetCDF file with optional compression.
 *
 * @param filename          Output NetCDF file name.
 * @param h_dense           Pointer to flattened 3D double array [system, time, variable].
 * @param time_vals         Pointer to array of time values (length: num_queries).
 * @param linkid_vals       Pointer to array of system/link IDs (length: num_systems).
 * @param state_vals        Pointer to array of state variable IDs (length: N_EQ).
 * @param num_queries       Number of time steps (size of time_vals).
 * @param num_systems       Number of systems/links (size of linkid_vals).
 * @param N_EQ              Number of state variables (size of state_vals).
 * @param compression_level Compression level (0 = no compression, 1-9 = increasing compression).
 */
void write_dense_netcdf(const std::string& filename,
                        const double* h_dense,
                        const double* time_vals,
                        const int* linkid_vals,
                        const int* state_vals,
                        int num_queries,
                        int num_systems,
                        int N_EQ,
                        int compression_level = 4);

/**
 * @brief Write a 2D array [system, variable] to a NetCDF file (no time dimension).
 *
 * @param filename          Output NetCDF file name.
 * @param h_y_final         Pointer to flattened 2D double array [system, variable] (size: num_systems * N_EQ).
 * @param linkid_vals       Pointer to array of system/link IDs (length: num_systems).
 * @param state_vals        Pointer to array of state variable IDs (length: N_EQ).
 * @param num_systems       Number of systems/links (size of linkid_vals).
 * @param N_EQ              Number of state variables (size of state_vals).
 * @param compression_level Compression level (0 = no compression, 1-9 = increasing compression).
 */
void write_final_netcdf(const std::string& filename,
                        const double* h_y_final,
                        const int* linkid_vals,
                        const int* state_vals,
                        int num_systems,
                        int N_EQ,
                        int compression_level = 4);