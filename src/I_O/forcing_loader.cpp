// src/I_O/forcing_loader.cpp

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "forcing_loader.hpp"


/* LookupMapper class implementation */
LookupMapper::LookupMapper(const std::string& filepath)
    : filepath_(filepath) {}

bool LookupMapper::load() {
    std::ifstream file(filepath_);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath_ << std::endl;
        return false;
    }

    std::string line;

    // Skip the header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string field;
        long long stream;
        int lat, lon;

        std::getline(ss, field, ',');
        stream = std::stoll(field);

        std::getline(ss, field, ',');
        lat = std::stoi(field);

        std::getline(ss, field, ',');
        lon = std::stoi(field);

        stream_map_[stream] = {lat, lon};
    }

    return true;
}

bool LookupMapper::hasStream(long long stream_id) const {
    return stream_map_.find(stream_id) != stream_map_.end();
}

std::pair<int, int> LookupMapper::getLatLon(long long stream_id) const {
    auto it = stream_map_.find(stream_id);
    if (it != stream_map_.end()) {
        return it->second;
    }
    return {-1, -1};  // Default/fallback value if not found
}

size_t LookupMapper::size() const {
    return stream_map_.size();
}


// Helper function to check NetCDF errors
// Helper function to check NetCDF errors
void NetCDFLoader::checkError(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        throw std::runtime_error(operation + ": " + nc_strerror(status));
    }
}
 
// Constructor
NetCDFLoader::NetCDFLoader(const std::string& filename, const std::string& varName)
    : ncid(-1), varid(-1), timeSize(0), latSize(0), lonSize(0), fileName(filename), varName(varName) {
    
    // Open the NetCDF file
    int status = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    checkError(status, "Opening file " + filename);
    
    // Get the variable ID
    status = nc_inq_varid(ncid, varName.c_str(), &varid);
    if (status != NC_NOERR) {
        nc_close(ncid);  // Clean up before throwing
        ncid = -1;
        throw std::runtime_error("Variable " + varName + " not found in file");
    }
    
    // Get variable dimensions
    int ndims;
    int dimids[NC_MAX_VAR_DIMS];
    status = nc_inq_var(ncid, varid, NULL, NULL, &ndims, dimids, NULL);
    checkError(status, "Inquiring variable dimensions");
    
    if (ndims != 3) {
        nc_close(ncid);
        ncid = -1;
        throw std::runtime_error("Expected 3D variable (time, lat, lon), got " + std::to_string(ndims) + "D");
    }
    
    // Get dimension sizes (assuming order: time, lat, lon)
    status = nc_inq_dimlen(ncid, dimids[0], &timeSize);
    checkError(status, "Getting time dimension size");
    
    status = nc_inq_dimlen(ncid, dimids[1], &latSize);
    checkError(status, "Getting latitude dimension size");
    
    status = nc_inq_dimlen(ncid, dimids[2], &lonSize);
    checkError(status, "Getting longitude dimension size");
    
    // std::cout << "Dataset dimensions: " << timeSize << " x " << latSize << " x " << lonSize << std::endl;
}
 
// Destructor
NetCDFLoader::~NetCDFLoader() {
    if (ncid >= 0) {
        nc_close(ncid);
        ncid = -1;
    }
}
 
// Move constructor
NetCDFLoader::NetCDFLoader(NetCDFLoader&& other) noexcept
    : ncid(other.ncid), varid(other.varid),
      timeSize(other.timeSize), latSize(other.latSize), lonSize(other.lonSize),
      fileName(std::move(other.fileName)), varName(std::move(other.varName)) {
    other.ncid = -1;  // Mark as moved
    other.varid = -1;
    other.timeSize = 0;
    other.latSize = 0;
    other.lonSize = 0;
}
 
// Move assignment operator
NetCDFLoader& NetCDFLoader::operator=(NetCDFLoader&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        if (ncid >= 0) {
            nc_close(ncid);
        }
        
        // Move resources
        ncid = other.ncid;
        varid = other.varid;
        timeSize = other.timeSize;
        latSize = other.latSize;
        lonSize = other.lonSize;
        fileName = std::move(other.fileName);
        varName = std::move(other.varName);
        
        // Reset other object
        other.ncid = -1;
        other.varid = -1;
        other.timeSize = 0;
        other.latSize = 0;
        other.lonSize = 0;
    }
    return *this;
}
 
// Load data by time chunk into memory
std::unique_ptr<float[]> NetCDFLoader::loadTimeChunk(size_t startTime, size_t numTimeSteps) {
    // Check if time chunk is out of bounds
    if (numTimeSteps == 0) {
        throw std::invalid_argument("Size of time chunk must be greater than zero");
    }
    if (startTime >= timeSize) {
        throw std::out_of_range("Start time index out of range");
    }
    if (startTime + numTimeSteps > timeSize) {
        throw std::out_of_range("Requested time steps exceed available data");
    }
 
    // Calculate the actual number of time steps to read
    size_t actualTimeSteps = std::min(numTimeSteps, timeSize - startTime);
    size_t totalElements = actualTimeSteps * latSize * lonSize;
    
    // Allocate memory
    std::unique_ptr<float[]> data = std::make_unique<float[]>(totalElements);
    
    // Define start and count arrays for subsetting
    size_t start[3] = {startTime, 0, 0};
    size_t count[3] = {actualTimeSteps, latSize, lonSize};
    
    // Read data from NetCDF file using C API
    int status = nc_get_vara_float(ncid, varid, start, count, data.get());
    checkError(status, "Reading variable data");
    
    // std::cout << "Loaded time chunk: steps " << startTime << " to "
    //           << (startTime + actualTimeSteps - 1) << " (" << actualTimeSteps
    //           << " time steps)" << std::endl;
    
    return data;
}
 
// Get a single value from pre-loaded chunk data
float NetCDFLoader::getValueFromChunk(const std::unique_ptr<float[]>& chunkData,
                                     size_t relativeTimeIndex, size_t latIndex, size_t lonIndex,
                                     size_t chunkTimeSize, size_t latSize, size_t lonSize) {
    // ESSENTIAL bounds checking
    if (relativeTimeIndex >= chunkTimeSize ||
        latIndex >= latSize ||
        lonIndex >= lonSize) {
        throw std::out_of_range("Chunk indices out of range");
    }
    
    // Calculate 1D index from 3D coordinates
    // Memory layout: [time][lat][lon] in row-major order
    size_t index = relativeTimeIndex * (latSize * lonSize) + latIndex * lonSize + lonIndex;
    return chunkData[index];
}
 
// Check if data is loaded correctly
bool NetCDFLoader::isDataLoaded() const {
    return ncid >= 0 && timeSize > 0 && latSize > 0 && lonSize > 0;
}

void NCForcing::loadData() {
    size_t totalSystems = systems;
    size_t totalDays    = days;

    h_data.clear();
    nForc = entries.size();
    dt.clear();
    nT.clear();

    // Precompute total size for allocation
    size_t totalSize = 0;
    for (auto& e : entries) {
        size_t stepsPerDay = static_cast<size_t>(24.0 / e.dt_hours);
        dt.push_back(e.dt_hours);
        nT.push_back(stepsPerDay * totalDays);
        totalSize += totalSystems * stepsPerDay * totalDays;
    }
    h_data.reserve(totalSize);

    // ───── Load each forcing ─────
    for (size_t f = 0; f < entries.size(); ++f) {
        std::cout << "[FORCING DEBUG] Loading " << entries[f].var_name
                  << " from " << entries[f].file << std::endl;

        NetCDFLoader loader(entries[f].file, entries[f].var_name);

        size_t stepsPerDay = static_cast<size_t>(24.0 / entries[f].dt_hours);
        size_t numTimeSteps = stepsPerDay * totalDays;

        auto chunkData = loader.loadTimeChunk(offset * stepsPerDay, numTimeSteps);

        size_t latSize = loader.getLatSize();
        size_t lonSize = loader.getLonSize();

        // ───── Flatten data as [f][t][s] ─────
        for (size_t t = 0; t < numTimeSteps; ++t) {
            for (size_t s = 0; s < totalSystems; ++s) {
                auto [ilat, ilon] = mapper.getLatLon(systemIds[s]);
                if (ilat < 0 || ilon < 0) {
                    throw std::runtime_error("Invalid lat/lon mapping for system " +
                                              std::to_string(systemIds[s]));
                }

                size_t idx3d = t * (latSize * lonSize) + ilat * lonSize + ilon;
                h_data.push_back(chunkData[idx3d]);
            }
        }

        std::cout << "[FORCING DEBUG] f=" << f 
                  << " nT=" << numTimeSteps 
                  << " first=" << h_data[h_data.size() - totalSystems] << std::endl;
    }

    std::cout << "[FORCING DEBUG] Final h_data size=" << h_data.size() << std::endl;

    
    // ───────────── DEBUG: Print forcing values for f=0 at t=0 and t=1 ─────────────
    if (!h_data.empty()) {
        size_t s = 0; // First system
        size_t f_precip = 0; // Assuming f=0 is precipitation
        size_t stepsPerDay_precip = static_cast<size_t>(24.0 / entries[f_precip].dt_hours);
        size_t totalSystems = systems;

        // Index for f=0 t=0 s=0
        size_t idx_t0 = (0 * totalSystems) + s;  
        std::cout << "[DEBUG CHECK] Forcing f=0 t=0 s=0 → "
                  << h_data[idx_t0] << std::endl;

        // Index for f=0 t=1 s=0 (next hour)
        size_t idx_t1 = (1 * totalSystems) + s;  
        std::cout << "[DEBUG CHECK] Forcing f=0 t=1 s=0 → "
                  << h_data[idx_t1] << std::endl;
    }
    
}
