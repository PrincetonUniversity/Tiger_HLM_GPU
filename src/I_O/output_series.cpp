#include <netcdf.h>
#include <string>
#include <iostream>
#include <vector>

#define NC_CHECK(call) \
    do { \
        int status = (call); \
        if (status != NC_NOERR) { \
            std::cerr << "NetCDF error: " << nc_strerror(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return; \
        } \
    } while (0)

/**
 * @brief Write a dense 3D array to a NetCDF file with optional compression.
 */
void write_dense_netcdf(const std::string& filename,
                        const double* h_dense,
                        const double* time_vals,
                        const int* linkid_vals,
                        const int* state_vals,
                        int num_queries,
                        int num_systems,
                        int N_EQ,
                        int compression_level = 4) {
    int ncid, sys_dimid, time_dimid, var_dimid;
    int sys_varid, time_varid, var_varid, dense_varid;

    // Create file
    NC_CHECK(nc_create(filename.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid));

    // Define dimensions
    NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));
    NC_CHECK(nc_def_dim(ncid, "time", num_queries, &time_dimid));
    NC_CHECK(nc_def_dim(ncid, "variable", N_EQ, &var_dimid));

    // Define coordinate variables
    NC_CHECK(nc_def_var(ncid, "system", NC_INT, 1, &sys_dimid, &sys_varid));
    NC_CHECK(nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid));
    NC_CHECK(nc_def_var(ncid, "variable", NC_INT, 1, &var_dimid, &var_varid));

    // Add attributes
    NC_CHECK(nc_put_att_text(ncid, sys_varid, "long_name", 6, "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4, "Time"));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "units", 37, "minutes since start of simulation"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "long_name", 14, "state variable"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "units", 13, "various units"));

    // Define main data variable
    int dims[3] = {sys_dimid, time_dimid, var_dimid};
    NC_CHECK(nc_def_var(ncid, "outputs", NC_DOUBLE, 3, dims, &dense_varid));

    // Set compression if requested
    if (compression_level > 0) {
        NC_CHECK(nc_def_var_deflate(ncid, dense_varid, 1, 1, compression_level));
    }

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    NC_CHECK(nc_put_var_int(ncid, sys_varid, linkid_vals));
    NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));
    NC_CHECK(nc_put_var_int(ncid, var_varid, state_vals));

    // Write main data
    NC_CHECK(nc_put_var_double(ncid, dense_varid, h_dense));

    // Close file
    NC_CHECK(nc_close(ncid));
}

/**
 * @brief Write only the final time step of a dense 3D array to a NetCDF file (no time dimension).
 */
void write_final_netcdf(const std::string& filename,
                        const double* h_y_final,
                        const int* linkid_vals,
                        const int* state_vals,
                        int num_systems,
                        int N_EQ,
                        int compression_level) {
    int ncid, sys_dimid, var_dimid;
    int sys_varid, var_varid, final_varid;

    // Create file
    NC_CHECK(nc_create(filename.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid));

    // Define dimensions
    NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));
    NC_CHECK(nc_def_dim(ncid, "variable", N_EQ, &var_dimid));

    // Define coordinate variables
    NC_CHECK(nc_def_var(ncid, "system", NC_INT, 1, &sys_dimid, &sys_varid));
    NC_CHECK(nc_def_var(ncid, "variable", NC_INT, 1, &var_dimid, &var_varid));

    // Add attributes
    NC_CHECK(nc_put_att_text(ncid, sys_varid, "long_name", 6, "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "long_name", 14, "state variable"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "units", 13, "various units"));

    // Define main data variable
    int dims[2] = {sys_dimid, var_dimid};
    NC_CHECK(nc_def_var(ncid, "outputs", NC_DOUBLE, 2, dims, &final_varid));

    // Set compression if requested
    if (compression_level > 0) {
        NC_CHECK(nc_def_var_deflate(ncid, final_varid, 1, 1, compression_level));
    }

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    NC_CHECK(nc_put_var_int(ncid, sys_varid, linkid_vals));
    NC_CHECK(nc_put_var_int(ncid, var_varid, state_vals));

    // Write main data
    NC_CHECK(nc_put_var_double(ncid, final_varid, h_y_final));

    // Close file
    NC_CHECK(nc_close(ncid));
}