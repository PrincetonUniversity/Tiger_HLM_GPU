#include <netcdf.h>
#include <string>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstddef>       
#include "models/model_Runoff5.hpp"     // for Runoff5::STATE_SURF_RUNOFF, etc.
#include "stream.hpp"                   // for Runoff5
#include <unordered_map>                // for LookupMapper
#include <algorithm>

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
    // NC_CHECK(nc_def_dim(ncid, "time", num_queries, &time_dimid));
    // Make time unlimited, we'll slice-write only num_queries samples
    NC_CHECK(nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dimid));
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
    // NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));
    NC_CHECK(nc_put_var_int(ncid, var_varid, state_vals));
    if (num_queries > 0) {
        size_t startT[1] = {0};
        size_t countT[1] = {static_cast<size_t>(num_queries)};
        NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals));
    }

    // Write main data
    // NC_CHECK(nc_put_var_float(ncid, dense_varid, h_dense));
    if (num_queries > 0) {
        size_t start3[3] = {0, 0, 0};
        size_t count3[3] = {
            static_cast<size_t>(num_systems),
            static_cast<size_t>(num_queries),
            static_cast<size_t>(N_EQ)
        };
        NC_CHECK(nc_put_vara_float(ncid, dense_varid, start3, count3, h_dense));
    }


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
    // NC_CHECK(nc_def_dim(ncid, "time",   num_queries, &time_dimid));
    // Unlimited time; slice write num_queries samples
    NC_CHECK(nc_def_dim(ncid, "time",   NC_UNLIMITED, &time_dimid));

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

    // Optional: Fill values for safer downstream handling
    {
        float fv = -9999.0f;
        NC_CHECK(nc_put_att_float(ncid, surf_varid,  "_FillValue",  NC_FLOAT, 1, &fv));
        NC_CHECK(nc_put_att_float(ncid, total_varid, "_FillValue",  NC_FLOAT, 1, &fv));
        NC_CHECK(nc_put_att_float(ncid, surf_varid,  "missing_value",  NC_FLOAT, 1, &fv));
        NC_CHECK(nc_put_att_float(ncid, total_varid, "missing_value",  NC_FLOAT, 1, &fv));
    }

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write coordinate variables
    NC_CHECK(nc_put_var_uint(  ncid, sys_varid,  linkid_vals));
    // NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));
    if (num_queries > 0) {
        size_t startT[1] = {0};
        size_t countT[1] = {static_cast<size_t>(num_queries)};
        NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals));
    }

    // Extract surface and total runoff into temporary buffers
    // std::vector<float> surf_data(size_t(num_systems) * size_t(num_queries));
    // std::vector<float> total_data(size_t(num_systems) * size_t(num_queries));

    std::vector<float> surf_data(
        static_cast<size_t>(num_systems) * static_cast<size_t>(std::max(0, num_queries)));
    std::vector<float> total_data(
        static_cast<size_t>(num_systems) * static_cast<size_t>(std::max(0, num_queries)));
 

    for (int s = 0; s < num_systems; ++s) {
        for (int t = 0; t < num_queries; ++t) {
            const size_t base = (size_t(s) * size_t(num_queries) + size_t(t)) * size_t(Runoff5::N_EQ);
            surf_data[size_t(s) * size_t(num_queries) + size_t(t)]  = h_dense[base + size_t(SURF_IDX)];
            total_data[size_t(s) * size_t(num_queries) + size_t(t)] = h_dense[base + size_t(TOTAL_IDX)];
        }
    }

    // Write runoff data
    // NC_CHECK(nc_put_var_float(ncid, surf_varid,  surf_data.data()));
    // NC_CHECK(nc_put_var_float(ncid, total_varid, total_data.data()));
    if (num_queries > 0) {
        size_t start2[2] = {0, 0};
        size_t count2[2] = {static_cast<size_t>(num_systems),
                            static_cast<size_t>(num_queries)};
        NC_CHECK(nc_put_vara_float(ncid, surf_varid,  start2, count2, surf_data.data()));
        NC_CHECK(nc_put_vara_float(ncid, total_varid, start2, count2, total_data.data()));
    }

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
    // NC_CHECK(nc_def_dim(ncid, "time",   num_queries, &time_dimid));
    // Unlimited time; we'll write only num_queries via slices
    NC_CHECK(nc_def_dim(ncid, "time",   NC_UNLIMITED, &time_dimid));

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
    // NC_CHECK(nc_put_var_uint(  ncid, sys_varid,  linkid_vals));
    // NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));
    NC_CHECK(nc_put_var_uint(ncid, sys_varid, linkid_vals));
    if (num_queries > 0) {
        size_t startT[1] = {0};
        size_t countT[1] = {static_cast<size_t>(num_queries)};
        NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals));
    }

    // For each selected variable, extract into a temp buffer and write
    // std::vector<float> buffer(size_t(num_systems) * size_t(num_queries));
    std::vector<float> buffer(
        static_cast<size_t>(num_systems) * static_cast<size_t>(std::max(0, num_queries)));
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
        // NC_CHECK(nc_put_var_float(ncid, varids[i], buffer.data()));
        if (num_queries > 0) {
            size_t start2[2] = {0, 0};
            size_t count2[2] = {static_cast<size_t>(num_systems),
                                static_cast<size_t>(num_queries)};
            NC_CHECK(nc_put_vara_float(ncid, varids[i], start2, count2, buffer.data()));
        }
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
// void write_runoff_rates_netcdf(const std::string& filename,
//                                const float*       surf_mmhr,   // size: num_systems * num_queries (usually 1)
//                                const float*       total_mmhr,  // size: num_systems * num_queries (usually 1)
//                                const double*      time_vals,   // minutes since time_origin
//                                const uint32_t*    linkid_vals,
//                                int                num_systems,
//                                int                num_queries, // usually 1 for chunk mean
//                                const std::string& time_origin)
// {
//     int ncid, sys_dimid, time_dimid;
//     int sys_varid, time_varid, surf_varid, total_varid;

//     // // Create file
//     // NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));

//     // ---------------------------------------------------------------------
//     // Sanity checks (prevents invalid NetCDF args when num_queries == 0)
//     // ---------------------------------------------------------------------
//     if (num_systems <= 0 || num_queries <= 0 || time_vals == nullptr ||
//         surf_mmhr == nullptr || total_mmhr == nullptr || linkid_vals == nullptr) {
//         std::cerr << "[RUNOFF_NC] Skipping write for " << filename
//                   << " (num_systems=" << num_systems
//                   << ", num_queries=" << num_queries << ")\n";
//         return;
//     }

//     std::cerr << "[RUNOFF_NC] Writing " << filename
//               << " with " << num_systems << " systems × "
//               << num_queries << " time steps.\n";

//     // Create file
//     // NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));
//     int status = nc_open(filename.c_str(), NC_WRITE, &ncid);
//     bool exists = (status == NC_NOERR);

//     if (!exists) {
//         NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));
//         // define dims/vars here
//         ...
//         NC_CHECK(nc_enddef(ncid));
//     } else {
//         // open existing
//         NC_CHECK(nc_inq_dimid(ncid, "system", &sys_dimid));
//         NC_CHECK(nc_inq_dimid(ncid, "time",   &time_dimid));
//         NC_CHECK(nc_inq_varid(ncid, "system", &sys_varid));
//         NC_CHECK(nc_inq_varid(ncid, "time",   &time_varid));
//         NC_CHECK(nc_inq_varid(ncid, "surface_runoff_mmhr", &surf_varid));
//         NC_CHECK(nc_inq_varid(ncid, "total_runoff_mmhr",   &total_varid));
//     }

//     // // Define dimensions
//     // NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));
//     // // NC_CHECK(nc_def_dim(ncid, "time",   num_queries, &time_dimid));
//     // // Unlimited time; slice-write the single (or few) samples
//     // NC_CHECK(nc_def_dim(ncid, "time",   NC_UNLIMITED, &time_dimid));

//     // NC_CHECK(nc_def_dim(ncid, "system", num_systems, &sys_dimid));

//     // // Define the time dimension explicitly for this file — runoff rates
//     // // have (nq - 1) samples compared to full-state arrays.
//     // NC_CHECK(nc_def_dim(ncid, "time", num_queries, &time_dimid));

//     // Define dimensions
//     NC_CHECK(nc_def_dim(ncid, "system",  num_systems,  &sys_dimid));
//     // Make time UNLIMITED so we can append chunks safely.
//     NC_CHECK(nc_def_dim(ncid, "time",    NC_UNLIMITED, &time_dimid));


//     // Define coordinate variables
//     NC_CHECK(nc_def_var(ncid, "system", NC_UINT,   1, &sys_dimid,  &sys_varid));
//     NC_CHECK(nc_def_var(ncid, "time",   NC_DOUBLE, 1, &time_dimid, &time_varid));
//     NC_CHECK(nc_put_att_text(ncid, sys_varid,  "long_name", 6, "LinkID"));
//     NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4, "Time"));

//     const std::string tu = "minutes since " + time_origin;
//     NC_CHECK(nc_put_att_text(ncid, time_varid, "units",    tu.size(), tu.c_str()));
//     NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar", 9, "gregorian"));

//     // Define data variables: surface and total mean rates (mm/hr)
//     int dims2[2] = { sys_dimid, time_dimid };
//     NC_CHECK(nc_def_var(ncid, "surface_runoff_mmhr", NC_FLOAT, 2, dims2, &surf_varid));
//     NC_CHECK(nc_def_var(ncid, "total_runoff_mmhr",   NC_FLOAT, 2, dims2, &total_varid));

//     // force contiguous layout (no chunking)
//     NC_CHECK(nc_def_var_chunking(ncid, surf_varid,  NC_CONTIGUOUS, nullptr));
//     NC_CHECK(nc_def_var_chunking(ncid, total_varid, NC_CONTIGUOUS, nullptr));

//     // Variable metadata
//     // NC_CHECK(nc_put_att_text(ncid, surf_varid,  "long_name", 27, "Surface runoff mean rate"));
//     // NC_CHECK(nc_put_att_text(ncid, total_varid, "long_name", 25, "Total runoff mean rate"));
//     NC_CHECK(nc_put_att_text(ncid, surf_varid,  "long_name", 22, "Surface runoff rate"));
//     NC_CHECK(nc_put_att_text(ncid, total_varid, "long_name", 20, "Total runoff rate"));
//     NC_CHECK(nc_put_att_text(ncid, surf_varid,  "units",      7, "mm hr-1"));
//     NC_CHECK(nc_put_att_text(ncid, total_varid, "units",      7, "mm hr-1"));

//     {
//         float fv = -9999.0f;
//         NC_CHECK(nc_put_att_float(ncid, surf_varid,  "_FillValue",    NC_FLOAT, 1, &fv));
//         NC_CHECK(nc_put_att_float(ncid, total_varid, "_FillValue",    NC_FLOAT, 1, &fv));
//         NC_CHECK(nc_put_att_float(ncid, surf_varid,  "missing_value", NC_FLOAT, 1, &fv));
//         NC_CHECK(nc_put_att_float(ncid, total_varid, "missing_value", NC_FLOAT, 1, &fv));
//     }

//     // End define mode
//     NC_CHECK(nc_enddef(ncid));

//     // // Write coordinates
//     // NC_CHECK(nc_put_var_uint(  ncid, sys_varid,  linkid_vals));
//     // // NC_CHECK(nc_put_var_double(ncid, time_varid, time_vals));
//     // if (num_queries > 0) {
//     //     size_t startT[1] = {0};
//     //     size_t countT[1] = {static_cast<size_t>(num_queries)};
//     //     NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals));
//     // }

//     // NC_CHECK(nc_put_var_uint(ncid, sys_varid, linkid_vals));

//     // // Write time coordinates — exactly num_queries entries
//     // {
//     //     size_t startT[1] = {0};
//     //     size_t countT[1] = {static_cast<size_t>(num_queries)};
//     //     NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals));
//     // }

//     // Coordinates: systems written all-at-once
//     NC_CHECK(nc_put_var_uint(ncid, sys_varid, linkid_vals));

//     // Compute current time length (append offset)
//     size_t tlen = 0;
//     NC_CHECK(nc_inq_dimlen(ncid, time_dimid, &tlen));

//     // Write time coordinates starting at tlen
//     {
//         size_t startT[1] = { tlen };
//         size_t countT[1] = { static_cast<size_t>(num_queries) };
//         NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals));
//     }


//     // // Write data
//     // // NC_CHECK(nc_put_var_float(ncid, surf_varid,  surf_mmhr));
//     // // NC_CHECK(nc_put_var_float(ncid, total_varid, total_mmhr));

//     // if (num_queries > 0) {
//     //     size_t start2[2] = {0, 0};
//     //     size_t count2[2] = {static_cast<size_t>(num_systems),
//     //                         static_cast<size_t>(num_queries)};
//     //     NC_CHECK(nc_put_vara_float(ncid, surf_varid,  start2, count2, surf_mmhr));
//     //     NC_CHECK(nc_put_vara_float(ncid, total_varid, start2, count2, total_mmhr));
//     // }

//     if (num_queries > 0) {
//         size_t start2[2] = { 0, tlen };   // append along time
//         size_t count2[2] = {
//             static_cast<size_t>(num_systems),
//             static_cast<size_t>(num_queries)
//         };
//         NC_CHECK(nc_put_vara_float(ncid, surf_varid,  start2, count2, surf_mmhr));
//         NC_CHECK(nc_put_vara_float(ncid, total_varid, start2, count2, total_mmhr));
//     }

//     // Close
//     NC_CHECK(nc_close(ncid));
// }


void write_runoff_rates_netcdf(const std::string& filename,
                               const float*       surf_mmhr,
                               const float*       total_mmhr,
                               const double*      time_vals,
                               const uint32_t*    linkid_vals,
                               int                num_systems,
                               int                num_queries,
                               const std::string& time_origin)
{
    int ncid, sys_dimid, time_dimid;
    int sys_varid, time_varid, surf_varid, total_varid;

    // Basic guards
    if (num_systems <= 0 || num_queries <= 0 ||
        surf_mmhr == nullptr || total_mmhr == nullptr ||
        time_vals == nullptr || linkid_vals == nullptr) {
        std::cerr << "[RUNOFF_NC] Skipping write for " << filename
                  << " (num_systems=" << num_systems
                  << ", num_queries=" << num_queries << ")\n";
        return;
    }

    // Try to open existing file for append
    int status = nc_open(filename.c_str(), NC_WRITE, &ncid);
    bool exists = (status == NC_NOERR);
    bool schema_ok = false;

    if (exists) {
        // Validate expected schema; if anything is missing, recreate the file
        int chk_sys_dim=-1, chk_time_dim=-1;
        int chk_sys_var=-1, chk_time_var=-1;
        int chk_surf=-1, chk_total=-1;

        if (nc_inq_dimid(ncid, "system", &chk_sys_dim)  == NC_NOERR &&
            nc_inq_dimid(ncid, "time",   &chk_time_dim) == NC_NOERR &&
            nc_inq_varid(ncid, "system", &chk_sys_var)  == NC_NOERR &&
            nc_inq_varid(ncid, "time",   &chk_time_var) == NC_NOERR &&
            nc_inq_varid(ncid, "surface_runoff_mmhr", &chk_surf)  == NC_NOERR &&
            nc_inq_varid(ncid, "total_runoff_mmhr",   &chk_total) == NC_NOERR) {
            // Also check that 'system' dimension matches
            size_t sys_len = 0;
            if (nc_inq_dimlen(ncid, chk_sys_dim, &sys_len) == NC_NOERR &&
                (int)sys_len == num_systems) {
                schema_ok = true;
            }
        }

        if (!schema_ok) {
            // Close the bad or incompatible file — we'll recreate
            nc_close(ncid);
            exists = false;
        }
    }

    if (!exists || !schema_ok) {
        // Create a fresh file with the expected schema
        NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));

        // Dimensions
        NC_CHECK(nc_def_dim(ncid, "system",  num_systems,  &sys_dimid));
        NC_CHECK(nc_def_dim(ncid, "time",    NC_UNLIMITED, &time_dimid));

        // Coordinate variables
        NC_CHECK(nc_def_var(ncid, "system", NC_UINT,   1, &sys_dimid,  &sys_varid));
        NC_CHECK(nc_def_var(ncid, "time",   NC_DOUBLE, 1, &time_dimid, &time_varid));
        NC_CHECK(nc_put_att_text(ncid, sys_varid,  "long_name", 6, "LinkID"));
        NC_CHECK(nc_put_att_text(ncid, time_varid, "long_name", 4, "Time"));
        const std::string tu = "minutes since " + time_origin;
        NC_CHECK(nc_put_att_text(ncid, time_varid, "units",    tu.size(), tu.c_str()));
        NC_CHECK(nc_put_att_text(ncid, time_varid, "calendar", 9, "gregorian"));

        // Data variables (system x time)
        int dims2[2] = { sys_dimid, time_dimid };
        NC_CHECK(nc_def_var(ncid, "surface_runoff_mmhr", NC_FLOAT, 2, dims2, &surf_varid));
        NC_CHECK(nc_def_var(ncid, "total_runoff_mmhr",   NC_FLOAT, 2, dims2, &total_varid));

        // Unlimited dims MUST be chunked; choose (all systems, 1 time) for efficient appends
        {
            size_t chunks[2] = { static_cast<size_t>(num_systems), 1 };
            NC_CHECK(nc_def_var_chunking(ncid, surf_varid,  NC_CHUNKED, chunks));
            NC_CHECK(nc_def_var_chunking(ncid, total_varid, NC_CHUNKED, chunks));
        }

        // (Optional) compression — comment out if you want raw speed over space
        // int shuffle=1, deflate=1, level=1;
        // NC_CHECK(nc_def_var_deflate(ncid, surf_varid, shuffle, deflate, level));
        // NC_CHECK(nc_def_var_deflate(ncid, total_varid, shuffle, deflate, level));

        // Metadata
        NC_CHECK(nc_put_att_text(ncid, surf_varid,  "long_name", 22, "Surface runoff rate"));
        NC_CHECK(nc_put_att_text(ncid, total_varid, "long_name", 20, "Total runoff rate"));
        NC_CHECK(nc_put_att_text(ncid, surf_varid,  "units",      7, "mm hr-1"));
        NC_CHECK(nc_put_att_text(ncid, total_varid, "units",      7, "mm hr-1"));
        {
            float fv = -9999.0f;
            NC_CHECK(nc_put_att_float(ncid, surf_varid,  "_FillValue",    NC_FLOAT, 1, &fv));
            NC_CHECK(nc_put_att_float(ncid, total_varid, "_FillValue",    NC_FLOAT, 1, &fv));
            NC_CHECK(nc_put_att_float(ncid, surf_varid,  "missing_value", NC_FLOAT, 1, &fv));
            NC_CHECK(nc_put_att_float(ncid, total_varid, "missing_value", NC_FLOAT, 1, &fv));
        }

        // End define mode and write static coordinate(s)
        NC_CHECK(nc_enddef(ncid));
        NC_CHECK(nc_put_var_uint(ncid, sys_varid, linkid_vals));
    } else {
        // Reuse IDs from existing, valid file
        NC_CHECK(nc_inq_dimid(ncid, "system", &sys_dimid));
        NC_CHECK(nc_inq_dimid(ncid, "time",   &time_dimid));
        NC_CHECK(nc_inq_varid(ncid, "system", &sys_varid));
        NC_CHECK(nc_inq_varid(ncid, "time",   &time_varid));
        NC_CHECK(nc_inq_varid(ncid, "surface_runoff_mmhr", &surf_varid));
        NC_CHECK(nc_inq_varid(ncid, "total_runoff_mmhr",   &total_varid));
    }

//     // Append at current time length
//     size_t tlen = 0;
//     NC_CHECK(nc_inq_dimlen(ncid, time_dimid, &tlen));

//     // Write the new time values at [tlen ... tlen+num_queries-1]
//     {
//         size_t startT[1] = { tlen };
//         size_t countT[1] = { static_cast<size_t>(num_queries) };
//         NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals));
//     }

//     // Write data slabs (system x time), appending along time
//     {
//         size_t start2[2] = { 0, tlen };
//         size_t count2[2] = { static_cast<size_t>(num_systems),
//                              static_cast<size_t>(num_queries) };
//         NC_CHECK(nc_put_vara_float(ncid, surf_varid,  start2, count2, surf_mmhr));
//         NC_CHECK(nc_put_vara_float(ncid, total_varid, start2, count2, total_mmhr));
//     }

//     // Flush and close
//     NC_CHECK(nc_sync(ncid));
//     NC_CHECK(nc_close(ncid));
// }
    // After locating var IDs and before nc_put_vara_double:
    double last_time = -std::numeric_limits<double>::infinity();
    size_t tlen = 0;
    NC_CHECK(nc_inq_dimlen(ncid, time_dimid, &tlen));
    if (tlen > 0) {
        size_t start1[1] = { tlen - 1 }, count1[1] = { 1 };
        NC_CHECK(nc_get_vara_double(ncid, time_varid, start1, count1, &last_time));
    }

    // Find first strictly greater sample
    int first = 0;
    while (first < num_queries && !(time_vals[first] > last_time)) ++first;

    if (first >= num_queries) {
        // Nothing new; close and return
        nc_close(ncid);
        return;
    }

    // Write times
    size_t startT[1] = { tlen };
    size_t countT[1] = { (size_t)(num_queries - first) };
    NC_CHECK(nc_put_vara_double(ncid, time_varid, startT, countT, time_vals + first));

    // Write data aligned to the kept times
    size_t start2[2] = { 0, tlen };
    size_t count2[2] = { (size_t)num_systems, (size_t)(num_queries - first) };
    NC_CHECK(nc_put_vara_float(ncid, surf_varid,  start2, count2, surf_mmhr  + (size_t)first * (size_t)num_systems));
    NC_CHECK(nc_put_vara_float(ncid, total_varid, start2, count2, total_mmhr + (size_t)first * (size_t)num_systems));

//   // Flush and close
    NC_CHECK(nc_sync(ncid));
    NC_CHECK(nc_close(ncid));
}

