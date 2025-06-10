#include <netcdf>
#include <string>
#include <iostream>

/**
 * @brief Write a dense 3D array to a NetCDF file with optional compression.
 *
 * @param filename          Output NetCDF file name.
 * @param h_dense           Pointer to flattened 3D float array [system, time, variable].
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
                        int compression_level = 4) {
    try {
        netCDF::NcFile dataFile(filename, netCDF::NcFile::replace);

        // Create dimensions
        auto sysDim = dataFile.addDim("system", num_systems);
        auto timeDim = dataFile.addDim("time", num_queries);
        auto varDim = dataFile.addDim("variable", N_EQ);

        // Add coordinate variables for each dimension
        auto sysVar = dataFile.addVar("system", netCDF::ncInt, sysDim);
        auto timeVar = dataFile.addVar("time", netCDF::ncFloat, timeDim);
        auto varVar = dataFile.addVar("variable", netCDF::ncInt, varDim);

        // Fill coordinate variables with values
        sysVar.putVar(linkid_vals);
        timeVar.putVar(time_vals);
        varVar.putVar(state_vals);

        // Add attributes to coordinate variables
        sysVar.putAtt("long_name", "LinkID");
        timeVar.putAtt("long_name", "Time");
        timeVar.putAtt("units", "minutes since start of simulation");
        varVar.putAtt("long_name", "state variable");
        varVar.putAtt("units", "various units");

        // Create variable: 3D [system, time, variable]
        auto denseVar = dataFile.addVar("outputs", netCDF::ncFloat, {sysDim, timeDim, varDim});

        // Set compression if requested
        if (compression_level > 0) {
            denseVar.setCompression(true, true, compression_level);
        }

        // Write the flattened array
        denseVar.putVar(h_dense);

    } catch (netCDF::exceptions::NcException &e) {
        std::cerr << "NetCDF error: " << e.what() << std::endl;
    }
}



/**
 * @brief Write only the final time step of a dense 3D array to a NetCDF file (no time dimension).
 *
 * @param filename          Output NetCDF file name.
 * @param h_dense           Pointer to flattened 3D float array [system, time, variable].
 * @param linkid_vals       Pointer to array of system/link IDs (length: num_systems).
 * @param state_vals        Pointer to array of state variable IDs (length: N_EQ).
 * @param num_queries       Number of time steps (size of time dimension in h_dense).
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
                        int compression_level) {
    try {
        netCDF::NcFile dataFile(filename, netCDF::NcFile::replace);

        // Create dimensions (no time)
        auto sysDim = dataFile.addDim("system", num_systems);
        auto varDim = dataFile.addDim("variable", N_EQ);

        // Add coordinate variables for each dimension
        auto sysVar = dataFile.addVar("system", netCDF::ncInt, sysDim);
        auto varVar = dataFile.addVar("variable", netCDF::ncInt, varDim);

        // Fill coordinate variables with values
        sysVar.putVar(linkid_vals);
        varVar.putVar(state_vals);

        // Add attributes to coordinate variables
        sysVar.putAtt("long_name", "LinkID");
        varVar.putAtt("long_name", "state variable");
        varVar.putAtt("units", "various units");

        // Create variable: 2D [system, variable]
        auto finalVar = dataFile.addVar("outputs", netCDF::ncFloat, {sysDim, varDim});

        // Set compression if requested
        if (compression_level > 0) {
            finalVar.setCompression(true, true, compression_level);
        }

        // Write the provided 2D array directly
        finalVar.putVar(h_y_final);

    } catch (netCDF::exceptions::NcException &e) {
        std::cerr << "NetCDF error: " << e.what() << std::endl;
    }
}