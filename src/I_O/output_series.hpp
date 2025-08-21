#pragma once
#include <string>
#include <cstdint>

/**
 * @brief Write a dense 3D array to a NetCDF file.
 *
 * @param filename          Output NetCDF file name.
 * @param h_dense           Pointer to flattened 3D double array [system, time, variable].
 * @param time_vals         Pointer to array of time values (length: num_queries).
 * @param linkid_vals       Pointer to array of system/link IDs (length: num_systems).
 * @param state_vals        Pointer to array of state variable IDs (length: N_EQ).
 * @param num_queries       Number of time steps (size of time_vals).
 * @param num_systems       Number of systems/links (size of linkid_vals).
 * @param N_EQ              Number of state variables (size of state_vals).
 */
void write_dense_netcdf(const std::string& filename,
                        const float* h_dense,
                        const double* time_vals,
                        const uint32_t* linkid_vals, 
                        const int* state_vals,
                        int num_queries,
                        int num_systems,
                        int N_EQ,
                        const std::string& time_origin); // e.g., "2023-01-01T00:00:00Z" passed from the user


/**
 * @brief Write a 2D array [system, variable] to a NetCDF file (no time dimension).
 *
 * @param filename          Output NetCDF file name.
 * @param h_y_final         Pointer to flattened 2D double array [system, variable] (size: num_systems * N_EQ).
 * @param linkid_vals       Pointer to array of system/link IDs (length: num_systems).
 * @param state_vals        Pointer to array of state variable IDs (length: N_EQ).
 * @param num_systems       Number of systems/links (size of linkid_vals).
 * @param N_EQ              Number of state variables (size of state_vals).
 */
void write_final_netcdf(const std::string& filename,
                        const double* h_y_final,
                        const uint32_t* linkid_vals, 
                        const int* state_vals,
                        int num_systems,
                        int N_EQ);


/**
 * @brief Write only the surface‐ and total‐runoff components from a dense
 *        3D array to a NetCDF file.
 *
 * @param filename          Output NetCDF file name.
 * @param h_dense           Pointer to flattened 3D double array [system, time, variable].
 *                          Only the entries at Runoff5::STATE_SURF_RUNOFF and
 *                          Runoff5::STATE_TOTAL_RUNOFF will be written.
 * @param time_vals         Pointer to array of time values (length: num_queries).
 * @param linkid_vals       Pointer to array of system/link IDs (length: num_systems).
 * @param num_queries       Number of time steps (size of time_vals).
 * @param num_systems       Number of systems/links (size of linkid_vals).
 */
void write_runoff_dense_netcdf(const std::string& filename,
                              const float*      h_dense,
                              const double*      time_vals,
                              const uint32_t*    linkid_vals,
                              int                num_queries,
                              int                num_systems,
                              const std::string& time_origin); // e.g., "2023-01-01T00:00:00Z" passed from the user
                            



/**
 * @brief Write only the selected state‐variables from a dense
 *        (system × time × variable) array into a NetCDF file.
 *
 * @param filename            Output NetCDF file name.
 * @param h_dense             Pointer to flattened 3D double array [system, time, variable].
 * @param time_vals           Pointer to array of time values (length: num_queries).
 * @param linkid_vals         Pointer to array of system/link IDs (length: num_systems).
 * @param selected_states     Pointer to array of state‐indices you wish to write.
 * @param state_names         Pointer to array of C‐strings, the NetCDF var names.
 * @param num_selected        Number of entries in selected_states (and state_names).
 * @param num_queries         Number of time steps (size of time_vals).
 * @param num_systems         Number of systems/links (size of linkid_vals).
 * @param full_N_EQ           Total number of variables in h_dense (Runoff5::N_EQ).
 */
void write_selected_dense_netcdf(const std::string& filename,
                                 const float*      h_dense,
                                 const double*      time_vals,
                                 const uint32_t*    linkid_vals,
                                 const int*         selected_states,
                                 const char* const* state_names,
                                 int                num_selected,
                                 int                num_queries,
                                 int                num_systems,
                                 int                full_N_EQ,
                                 const std::string& time_origin); // e.g., "2023-01-01T00:00:00Z" passed from the user
                            
