#include <netcdf.h>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstddef>       
#include "models/model_Runoff5.hpp"     // for Runoff5::STATE_SURF_RUNOFF, etc.
#include "stream.hpp"                   // for Runoff5
#include <unordered_map>                // for LookupMapper


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
                        const float* h_dense,
                        const double* time_vals,
                        const uint32_t* linkid_vals, 
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
    NC_CHECK(nc_def_dim(ncid, "variable", N_EQ, &var_dimid));

    // Define coordinate variables
    NC_CHECK(nc_def_var(ncid, "system", NC_UINT, 1, &sys_dimid, &sys_varid));
    NC_CHECK(nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid));
    NC_CHECK(nc_def_var(ncid, "variable", NC_INT, 1, &var_dimid, &var_varid));

    // Add attributes
    NC_CHECK(nc_put_att_text(ncid, sys_varid, "long_name", 6, "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4, "Time"));
    std::string tu = "minutes since " + time_origin;
    NC_CHECK(nc_put_att_text(ncid, time_varid, "units",
                             tu.size(), tu.c_str()));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar",
                             9, "gregorian"));

    NC_CHECK(nc_put_att_text(ncid, var_varid, "long_name", 14, "state variable"));
    NC_CHECK(nc_put_att_text(ncid, var_varid, "units", 13, "various units"));

    // Define main data variable
    int dims[3] = {sys_dimid, time_dimid, var_dimid};
    NC_CHECK(nc_def_var(ncid, "outputs", NC_FLOAT, 3, dims, &dense_varid));

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    NC_CHECK(nc_put_var_uint( ncid, sys_varid, linkid_vals));
    NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));
    NC_CHECK(nc_put_var_int(ncid, var_varid, state_vals));

    // Write main data
    NC_CHECK(nc_put_var_float(ncid, dense_varid, h_dense));

    // Close file
    NC_CHECK(nc_close(ncid));
}

/**
 * @brief Write only the final time step of a dense 3D array to a NetCDF file (no time dimension).
 */
void write_final_netcdf(const std::string& filename,
                        const double* h_y_final,
                        const uint32_t* linkid_vals, 
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
    NC_CHECK(nc_put_var_uint(ncid, sys_varid, linkid_vals)); 
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
                              const float*      h_dense,
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
    std::string tu = "minutes since " + time_origin;
    NC_CHECK(nc_put_att_text(ncid, time_varid, "units",
                             tu.size(), tu.c_str()));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar",
                             9, "gregorian"));

    // Define runoff variables
    int dims2[2] = { sys_dimid, time_dimid };
    NC_CHECK(nc_def_var(ncid, "surface_runoff", NC_FLOAT, 2, dims2, &surf_varid));
    NC_CHECK(nc_def_var(ncid, "total_runoff",   NC_FLOAT, 2, dims2, &total_varid));

     // Add descriptive attributes (units = meters, cumulative)
    NC_CHECK(nc_put_att_text(ncid, surf_varid,  "long_name", 24, "Surface runoff (cumulative)"));
    NC_CHECK(nc_put_att_text(ncid, surf_varid,  "units",      1, "m"));
    NC_CHECK(nc_put_att_text(ncid, total_varid, "long_name", 22, "Total runoff (cumulative)"));
    NC_CHECK(nc_put_att_text(ncid, total_varid, "units",      1, "m"));

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    NC_CHECK(nc_put_var_uint(  ncid, sys_varid,  linkid_vals));
    NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));

    // Extract surface and total runoff into temporary buffers
    std::vector<float> surf_data(size_t(num_systems) * size_t(num_queries));
    std::vector<float> total_data(size_t(num_systems) * size_t(num_queries));

    for (int s = 0; s < num_systems; ++s) {
        for (int t = 0; t < num_queries; ++t) {
            const size_t base = (size_t(s) * size_t(num_queries) + size_t(t)) * size_t(Runoff5::N_EQ);
            surf_data[size_t(s) * size_t(num_queries) + size_t(t)]  = h_dense[base + size_t(SURF_IDX)];
            total_data[size_t(s) * size_t(num_queries) + size_t(t)] = h_dense[base + size_t(TOTAL_IDX)];
        }
    }

    // Write runoff data
    NC_CHECK(nc_put_var_float(ncid, surf_varid,  surf_data.data()));
    NC_CHECK(nc_put_var_float(ncid, total_varid, total_data.data()));

    // Close file
    NC_CHECK(nc_close(ncid));
}

/**
 * @brief Write selected states from a dense 3D array to a NetCDF file.
 * 
 */
void write_selected_dense_netcdf(const std::string& filename,
                                //  const double*      h_dense,
                                 const float*      h_dense,
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
                            NC_FLOAT,
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
    std::vector<float> buffer(size_t(num_systems) * size_t(num_queries));
    for (int i = 0; i < num_selected; ++i) {
        int idx = selected_states[i];
        for (int s = 0; s < num_systems; ++s) {
            for (int t = 0; t < num_queries; ++t) {
                // layout: (s * num_queries + t) * full_N_EQ + idx
                const size_t linear = size_t(s) * size_t(num_queries) + size_t(t);
                const size_t src    = linear * size_t(full_N_EQ) + idx;
                buffer[linear] = h_dense[src];
            }
        }
        NC_CHECK(nc_put_var_float(ncid, varids[i], buffer.data()));
    }

    // Close file
    NC_CHECK(nc_close(ncid));
}

// Making sure NC_CHECK is defined as the fatal variant for loader
#undef NC_CHECK
#define NC_CHECK(call) \
    do { \
        int status = (call); \
        if (status != NC_NOERR) { \
            std::cerr << "NetCDF error: " << nc_strerror(status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// Reads 2D (system, variable) 'outputs' from final.nc and fills streams[].y0
void load_initial_from_final_nc_serial(const std::string& path,
                                       std::vector<Stream<Runoff5>>& streams) {
    int ncid;
    NC_CHECK(nc_open(path.c_str(), NC_NOWRITE, &ncid));

    // var ids
    int sys_varid, var_varid, out_varid;
    NC_CHECK(nc_inq_varid(ncid, "system",  &sys_varid));
    NC_CHECK(nc_inq_varid(ncid, "variable",&var_varid)); // used for size check
    NC_CHECK(nc_inq_varid(ncid, "outputs", &out_varid));

    // dims
    int ndims = 0, dimids[NC_MAX_VAR_DIMS];
    NC_CHECK(nc_inq_varndims(ncid, out_varid, &ndims));
    if (ndims != 2) {
        std::cerr << "final.nc: expected outputs(system, variable) 2D, got ndims="
                  << ndims << "\n";
        std::exit(1);
    }
    NC_CHECK(nc_inq_vardimid(ncid, out_varid, dimids));
    size_t nsys=0, nvar=0;
    NC_CHECK(nc_inq_dim(ncid, dimids[0], nullptr, &nsys));
    NC_CHECK(nc_inq_dim(ncid, dimids[1], nullptr, &nvar));
    if (nvar != Runoff5::N_EQ) {
        std::cerr << "final.nc: variable dimension="<<nvar
                  << " but Runoff5::N_EQ="<<Runoff5::N_EQ << "\n";
        std::exit(1);
    }

    // read system coord (handle int/uint/int64)
    nc_type sys_t;
    NC_CHECK(nc_inq_vartype(ncid, sys_varid, &sys_t));
    std::vector<long long> sys_ids(nsys);
    if (sys_t == NC_UINT) {
        std::vector<unsigned int> tmp(nsys);
        NC_CHECK(nc_get_var_uint(ncid, sys_varid, tmp.data()));
        for (size_t i=0;i<nsys;++i) sys_ids[i] = tmp[i];
    } else if (sys_t == NC_INT) {
        std::vector<int> tmp(nsys);
        NC_CHECK(nc_get_var_int(ncid, sys_varid, tmp.data()));
        for (size_t i=0;i<nsys;++i) sys_ids[i] = tmp[i];
    } else if (sys_t == NC_INT64) {
        NC_CHECK(nc_get_var_longlong(ncid, sys_varid, sys_ids.data()));
    } else {
        std::cerr << "final.nc: unsupported 'system' type\n";
        std::exit(1);
    }

    // row map
    std::unordered_map<long long,size_t> row_of;
    row_of.reserve(nsys*1.3);
    for (size_t r=0;r<nsys;++r) row_of[sys_ids[r]] = r;

    // read outputs (float or double) and assign to y0
    nc_type out_t;
    NC_CHECK(nc_inq_vartype(ncid, out_varid, &out_t));
    if (out_t == NC_DOUBLE) {
        std::vector<double> all(nsys*nvar);
        NC_CHECK(nc_get_var_double(ncid, out_varid, all.data()));
        for (auto &st : streams) {
            auto it = row_of.find(static_cast<long long>(st.sp.stream));
            if (it == row_of.end()) {
                std::cerr << "final.nc missing stream ID " << st.sp.stream << "\n";
                std::exit(1);
            }
            size_t r = it->second;
            for (size_t c=0;c<nvar;++c) st.y0[c] = all[r*nvar + c];
        }
    } else if (out_t == NC_FLOAT) {
        std::vector<float> all(nsys*nvar);
        NC_CHECK(nc_get_var_float(ncid, out_varid, all.data()));
        for (auto &st : streams) {
            auto it = row_of.find(static_cast<long long>(st.sp.stream));
            if (it == row_of.end()) {
                std::cerr << "final.nc missing stream ID " << st.sp.stream << "\n";
                std::exit(1);
            }
            size_t r = it->second;
            for (size_t c=0;c<nvar;++c) st.y0[c] = static_cast<double>(all[r*nvar + c]);
        }
    } else {
        std::cerr << "final.nc: 'outputs' must be float or double\n";
        std::exit(1);
    }

    NC_CHECK(nc_close(ncid));
}

/**
 * @brief Write per-chunk mean runoff rates (mm/hr) as a 2D (system, time) file.
 *        Expects one time sample per chunk (time dimension = 1).
 */
void write_runoff_rates_netcdf(const std::string& filename,
                               const float*       surf_mmhr,   // size: num_systems * num_queries (usually 1)
                               const float*       total_mmhr,  // size: num_systems * num_queries (usually 1)
                               const double*      time_vals,   // minutes since time_origin
                               const uint32_t*    linkid_vals,
                               int                num_systems,
                               int                num_queries, // usually 1 for chunk mean
                               const std::string& time_origin)
{
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
    NC_CHECK(nc_put_att_text(ncid, sys_varid,  "long_name", 6, "LinkID"));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4, "Time"));

    const std::string tu = "minutes since " + time_origin;
    NC_CHECK(nc_put_att_text(ncid, time_varid, "units",    tu.size(), tu.c_str()));
    NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar", 9, "gregorian"));

    // Define data variables: surface and total mean rates (mm/hr)
    int dims2[2] = { sys_dimid, time_dimid };
    NC_CHECK(nc_def_var(ncid, "surface_runoff_mmhr", NC_FLOAT, 2, dims2, &surf_varid));
    NC_CHECK(nc_def_var(ncid, "total_runoff_mmhr",   NC_FLOAT, 2, dims2, &total_varid));

    // force contiguous layout (no chunking)
    NC_CHECK(nc_def_var_chunking(ncid, surf_varid,  NC_CONTIGUOUS, nullptr));
    NC_CHECK(nc_def_var_chunking(ncid, total_varid, NC_CONTIGUOUS, nullptr));

    // Variable metadata
    // NC_CHECK(nc_put_att_text(ncid, surf_varid,  "long_name", 27, "Surface runoff mean rate"));
    // NC_CHECK(nc_put_att_text(ncid, total_varid, "long_name", 25, "Total runoff mean rate"));
    NC_CHECK(nc_put_att_text(ncid, surf_varid,  "long_name", 22, "Surface runoff rate"));
    NC_CHECK(nc_put_att_text(ncid, total_varid, "long_name", 20, "Total runoff rate"));
    NC_CHECK(nc_put_att_text(ncid, surf_varid,  "units",      7, "mm hr-1"));
    NC_CHECK(nc_put_att_text(ncid, total_varid, "units",      7, "mm hr-1"));

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinates
    NC_CHECK(nc_put_var_uint(  ncid, sys_varid,  linkid_vals));
    NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));

    // Write data
    NC_CHECK(nc_put_var_float(ncid, surf_varid,  surf_mmhr));
    NC_CHECK(nc_put_var_float(ncid, total_varid, total_mmhr));

    // Close
    NC_CHECK(nc_close(ncid));
}



