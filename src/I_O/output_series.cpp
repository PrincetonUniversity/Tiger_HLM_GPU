#include <netcdf.h>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstddef>            // for ptrdiff_t
#include "models/model_Runoff5.hpp"


#define NC_CHECK(call) \
    do { \
        int status = (call); \
        if (status != NC_NOERR) { \
            std::cerr << "NetCDF error: " << nc_strerror(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return; \
        } \
    } while (0)



/**
 * @brief Write a dense 3D array to a NetCDF file.
 */
void write_dense_netcdf(const std::string& filename,
                        const double* h_dense,
                        const double* time_vals,
                        // const int* linkid_vals, // change to stream ID
                        const uint32_t* linkid_vals, // new: 32-bit unsigned integer
                        const int* state_vals, 
                        int num_queries,
                        int num_systems,
                        int N_EQ,
                        const std::string& time_origin) { // e.g., "2023-01-01T00:00:00Z" passed from the user
                        
    int ncid, sys_dimid, time_dimid, var_dimid;
    int sys_varid, time_varid, var_varid, dense_varid;

    // Create file
    NC_CHECK(nc_create(filename.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid));

    // Define dimensions
    NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));
    NC_CHECK(nc_def_dim(ncid, "time", num_queries, &time_dimid));
    //NC_CHECK(nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dimid));
    NC_CHECK(nc_def_dim(ncid, "variable", N_EQ, &var_dimid));

    // Define coordinate variables
    // NC_CHECK(nc_def_var(ncid, "system", NC_INT, 1, &sys_dimid, &sys_varid));
    // new: 32-bit unsinged integer
    NC_CHECK(nc_def_var(ncid, "system", NC_UINT, 1, &sys_dimid, &sys_varid));
    NC_CHECK(nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid));
    NC_CHECK(nc_def_var(ncid, "variable", NC_INT, 1, &var_dimid, &var_varid));

    // Add attributes
    NC_CHECK(nc_put_att_text(ncid, sys_varid, "long_name", 6, "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4, "Time"));
    // NC_CHECK(nc_put_att_text(ncid, time_varid, "units", 37, "minutes since start of simulation"));
    std::string tu = "minutes since " + time_origin;
    NC_CHECK(nc_put_att_text(ncid, time_varid, "units",
                             tu.size(), tu.c_str()));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar",
                             9, "gregorian"));

    NC_CHECK(nc_put_att_text(ncid, var_varid, "long_name", 14, "state variable"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "units", 13, "various units"));

    // Define main data variable
    int dims[3] = {sys_dimid, time_dimid, var_dimid};
    NC_CHECK(nc_def_var(ncid, "outputs", NC_DOUBLE, 3, dims, &dense_varid));

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    // NC_CHECK(nc_put_var_int(ncid, sys_varid, linkid_vals));
    NC_CHECK(nc_put_var_uint( ncid, sys_varid, linkid_vals));// new: 32-bit unsigned integer
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
                        // const int* linkid_vals, //change to stream ID
                        const uint32_t* linkid_vals, // new: 32-bit unsigned integer
                        const int* state_vals,
                        int num_systems,
                        int N_EQ) {
    int ncid, sys_dimid, var_dimid;
    int sys_varid, var_varid, final_varid;

    // Create file
    NC_CHECK(nc_create(filename.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid));

    // Define dimensions
    NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));
    NC_CHECK(nc_def_dim(ncid, "variable", N_EQ, &var_dimid));

    // Define coordinate variables
    // NC_CHECK(nc_def_var(ncid, "system", NC_INT, 1, &sys_dimid, &sys_varid));
    // new: 32-bit unsinged integer
    NC_CHECK(nc_def_var(ncid, "system", NC_UINT, 1, &sys_dimid, &sys_varid));
    NC_CHECK(nc_def_var(ncid, "variable", NC_INT, 1, &var_dimid, &var_varid));

    // Add attributes
    NC_CHECK(nc_put_att_text(ncid, sys_varid, "long_name", 6, "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "long_name", 14, "state variable"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "units", 13, "various units"));

    // Define main data variable
    int dims[2] = {sys_dimid, var_dimid};
    NC_CHECK(nc_def_var(ncid, "outputs", NC_DOUBLE, 2, dims, &final_varid));

   

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    // NC_CHECK(nc_put_var_int(ncid, sys_varid, linkid_vals));
    NC_CHECK(nc_put_var_uint(ncid, sys_varid, linkid_vals)); // new: 32-bit unsigned integer
    NC_CHECK(nc_put_var_int(ncid, var_varid, state_vals));

    // Write main data
    NC_CHECK(nc_put_var_double(ncid, final_varid, h_y_final));

    // Close file
    NC_CHECK(nc_close(ncid));
}

/**
 * @brief Write only the surface‐ and total‐runoff components from a dense
 *        (system × time × variable) array into a NetCDF file.
 */
void write_runoff_dense_netcdf(const std::string& filename,
                              const double*      h_dense,
                              const double*      time_vals,
                              const uint32_t*    linkid_vals,
                              int                num_queries,
                              int                num_systems,
                              const std::string& time_origin) // e.g., "2023-01-01T00:00:00Z" passed from the user
{
    // indices into the last axis
    constexpr int SURF_IDX  = Runoff5::STATE_SURF_RUNOFF;
    constexpr int TOTAL_IDX = Runoff5::STATE_TOTAL_RUNOFF;

    int ncid, sys_dimid, time_dimid;
    int sys_varid, time_varid, surf_varid, total_varid;

    // Create file
    NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));

    // Define dimensions
    NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));
    NC_CHECK(nc_def_dim(ncid, "time",   num_queries, &time_dimid));

    // Define coordinate variables
    NC_CHECK(nc_def_var(ncid, "system", NC_UINT,   1, &sys_dimid,  &sys_varid));
    NC_CHECK(nc_def_var(ncid, "time",   NC_DOUBLE, 1, &time_dimid, &time_varid));

    // Add attributes
    NC_CHECK(nc_put_att_text(ncid, sys_varid,  "long_name", 6,  "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4,  "Time"));
    // NC_CHECK(nc_put_att_text(ncid, time_varid, "units",     37, "minutes since start of simulation"));
    std::string tu = "minutes since " + time_origin;
    NC_CHECK(nc_put_att_text(ncid, time_varid, "units",
                             tu.size(), tu.c_str()));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar",
                             9, "gregorian"));

    // Define runoff variables
    int dims2[2] = { sys_dimid, time_dimid };
    NC_CHECK(nc_def_var(ncid, "surface_runoff", NC_DOUBLE, 2, dims2, &surf_varid));
    NC_CHECK(nc_def_var(ncid, "total_runoff",   NC_DOUBLE, 2, dims2, &total_varid));

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    NC_CHECK(nc_put_var_uint(  ncid, sys_varid,  linkid_vals));
    NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));

    // Extract surface and total runoff into temporary buffers
    std::vector<double> surf_data(num_systems * num_queries);
    std::vector<double> total_data(num_systems * num_queries);
    for (int s = 0; s < num_systems; ++s) {
        for (int t = 0; t < num_queries; ++t) {
            int base = (s * num_queries + t) * Runoff5::N_EQ;
            surf_data[s * num_queries + t]  = h_dense[base + SURF_IDX];
            total_data[s * num_queries + t] = h_dense[base + TOTAL_IDX];
        }
    }

    // Write runoff data
    NC_CHECK(nc_put_var_double(ncid, surf_varid,  surf_data.data()));
    NC_CHECK(nc_put_var_double(ncid, total_varid, total_data.data()));

    // Close file
    NC_CHECK(nc_close(ncid));
}

/**
 * @brief Write selected states from a dense 3D array to a NetCDF file.
 * 
 */
void write_selected_dense_netcdf(const std::string& filename,
                                 const double*      h_dense,
                                 const double*      time_vals,
                                 const uint32_t*    linkid_vals,
                                 const int*         selected_states,
                                 const char* const* state_names,
                                 int                num_selected,
                                 int                num_queries,
                                 int                num_systems,
                                 int                full_N_EQ,
                                 const std::string& time_origin) // e.g., "2023-01-01T00:00:00Z" passed from the user
{
    int ncid, sys_dimid, time_dimid;
    int sys_varid, time_varid;
    std::vector<int> varids(num_selected);

    // Create file
    NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));

    // Define dimensions
    NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));
    NC_CHECK(nc_def_dim(ncid, "time",   num_queries, &time_dimid));

    // Define coordinate variables
    NC_CHECK(nc_def_var(ncid, "system", NC_UINT,   1, &sys_dimid,  &sys_varid));
    NC_CHECK(nc_def_var(ncid, "time",   NC_DOUBLE, 1, &time_dimid, &time_varid));

    // Add attributes
    NC_CHECK(nc_put_att_text(ncid, sys_varid,  "long_name", 6,  "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4,  "Time"));
    // NC_CHECK(nc_put_att_text(ncid, time_varid, "units",     37, "minutes since start of simulation"));
    std::string tu = "minutes since " + time_origin;
    NC_CHECK(nc_put_att_text(ncid, time_varid, "units",
                             tu.size(), tu.c_str()));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar",
                             9, "gregorian"));

    // Define one variable per selected state
    int dims2[2] = { sys_dimid, time_dimid };
    for (int i = 0; i < num_selected; ++i) {
        NC_CHECK(nc_def_var(ncid,
                            state_names[i],
                            NC_DOUBLE,
                            2,
                            dims2,
                            &varids[i]));
    }

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coords
    NC_CHECK(nc_put_var_uint(  ncid, sys_varid,  linkid_vals));
    NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));

    // For each selected variable, extract into a temp buffer and write
    std::vector<double> buffer(num_systems * num_queries);
    for (int i = 0; i < num_selected; ++i) {
        int idx = selected_states[i];
        for (int s = 0; s < num_systems; ++s) {
            for (int t = 0; t < num_queries; ++t) {
                // layout: (s * num_queries + t) * full_N_EQ + idx
                buffer[s * num_queries + t] =
                    h_dense[(s * num_queries + t) * full_N_EQ + idx];
            }
        }
        NC_CHECK(nc_put_var_double(ncid, varids[i], buffer.data()));
    }

    // Close file
    NC_CHECK(nc_close(ncid));
}




