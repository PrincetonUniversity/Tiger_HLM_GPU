#include <netcdf.h>
#include <netcdf_par.h>
#include <vector>
#include <iostream>
#include <mpi.h>


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
* @param compression_level  Compression level (0 = no compression, 1-9 = increasing compression).
* @note
* - The function assumes that the data in @p h_dense_chunk is organized in the order:
*   [system][time][variable], contiguous in memory.
* - All ranks must call this function collectively.
* - The NetCDF file is overwritten if it already exists.
* - Requires NetCDF library with parallel I/O support (netcdf_par.h).
* - Requires MPI to be initialized before calling this function.
*/

void write_dense_outputs_parallel(
    const std::string& filename,
    MPI_Comm comm,
    int rank,
    int num_systems,
    int num_queries,
    int N_EQ,
    int q_start,
    int q_count,
    const std::vector<double>& h_query_times,
    const std::vector<double>& h_dense_chunk,
    int compression_level
) 
{
    int ncid, dim_system, dim_time, dim_variable;
    int var_system, var_time, var_variable, var_outputs;
    int retval;

    // Create file in parallel
    retval = nc_create_par(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, comm, MPI_INFO_NULL, &ncid);
    if (retval != NC_NOERR) {
        std::cerr << "nc_create_par error: " << nc_strerror(retval) << std::endl;
        MPI_Abort(comm, retval);
    }

    // Define dimensions
    nc_def_dim(ncid, "system", num_systems, &dim_system);
    nc_def_dim(ncid, "time", num_queries, &dim_time);
    nc_def_dim(ncid, "variable", N_EQ, &dim_variable);

    // Define coordinate variables
    nc_def_var(ncid, "system", NC_INT, 1, &dim_system, &var_system);
    nc_def_var(ncid, "time", NC_DOUBLE, 1, &dim_time, &var_time);
    nc_def_var(ncid, "variable", NC_INT, 1, &dim_variable, &var_variable);

    // Define output variable
    int dims_out[3] = {dim_system, dim_time, dim_variable};
    nc_def_var(ncid, "outputs", NC_FLOAT, 3, dims_out, &var_outputs); //!!!!! Discuss what data types

    // Set compression
    if (compression_level > 0) {
        retval = nc_def_var_deflate(ncid, var_outputs, 1, 1, compression_level);
        if (retval != NC_NOERR) {
            std::cerr << "nc_def_var_deflate error: " << nc_strerror(retval) << std::endl;
            MPI_Abort(comm, retval);
        }
    }

    // End define mode
    nc_enddef(ncid);

    // Set collective access for outputs
    nc_var_par_access(ncid, var_outputs, NC_COLLECTIVE);

    // Rank 0 writes coordinate variables
    if (rank == 0) {
        std::vector<int> system_vals(num_systems), variable_vals(N_EQ);
        for (int s = 0; s < num_systems; ++s) system_vals[s] = s;
        for (int v = 0; v < N_EQ; ++v) variable_vals[v] = v;

        nc_put_var_int(ncid, var_system, system_vals.data());
        nc_put_var_double(ncid, var_time, h_query_times.data());
        nc_put_var_int(ncid, var_variable, variable_vals.data());
    }

    // All ranks write their chunk of output
    size_t start[3] = {0, static_cast<size_t>(q_start), 0};
    size_t count[3] = {static_cast<size_t>(num_systems), static_cast<size_t>(q_count), static_cast<size_t>(N_EQ)};

    retval = nc_put_vara_double(ncid, var_outputs, start, count, h_dense_chunk.data());
    if (retval != NC_NOERR) {
        std::cerr << "nc_put_vara_double error: " << nc_strerror(retval) << std::endl;
        MPI_Abort(comm, retval);
    }

    nc_close(ncid);
}
