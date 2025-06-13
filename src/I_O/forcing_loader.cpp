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
    
    std::cout << "Dataset dimensions: " << timeSize << " x " << latSize << " x " << lonSize << std::endl;
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
    
    std::cout << "Loaded time chunk: steps " << startTime << " to "
              << (startTime + actualTimeSteps - 1) << " (" << actualTimeSteps
              << " time steps)" << std::endl;
    
    return data;
}

// Load data by any chunk: time/lat/lon into memory
std::unique_ptr<float[]> NetCDFLoader::loadChunk(size_t startTime, size_t numTime, 
                                                  size_t startLat, size_t numLat, 
                                                  size_t startLon, size_t numLon) {
    // Check if chunk is out of bounds
    if (numTime == 0 || numLat == 0 || numLon == 0) {
        throw std::invalid_argument("Size of chunk dimensions must be greater than zero");
    }
    if (startTime >= timeSize || startLat >= latSize || startLon >= lonSize) {
        throw std::out_of_range("Start indices out of range");
    }
    if (startTime + numTime > timeSize || startLat + numLat > latSize || startLon + numLon > lonSize) {
        throw std::out_of_range("Requested chunk exceeds available data");
    }
    
    // Calculate total elements in the chunk
    size_t totalElements = numTime * numLat * numLon;
    
    // Allocate memory for the chunk
    std::unique_ptr<float[]> data = std::make_unique<float[]>(totalElements);
    
    // Define start and count arrays for subsetting
    size_t start[3] = {startTime, startLat, startLon};
    size_t count[3] = {numTime, numLat, numLon};
    
    // Read data from NetCDF file using C API
    int status = nc_get_vara_float(ncid, varid, start, count, data.get());
    checkError(status, "Reading variable data");
    
    std::cout << "Loaded chunk for " << varName << ": time steps " << startTime << " to "
              << (startTime + numTime - 1) << ", lat " << startLat << " to "
              << (startLat + numLat - 1) << ", lon " << startLon << " to "
              << (startLon + numLon - 1) << std::endl;
    
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


// Calculate spatial chunks based on latitude and longitude sizes
std::vector<SpatialChunk> NetCDFLoader::calculateSpatialChunks(size_t latSize, size_t lonSize, int numChunks) {
    if (numChunks <= 0) {
        throw std::invalid_argument("Number of chunks must be positive");
    }
    if (latSize == 0 || lonSize == 0) {
        throw std::invalid_argument("Spatial dimensions must be positive");
    }
    
    std::vector<SpatialChunk> chunks(numChunks);
    
    // Divide by the smaller dimension for better load balancing
    if (latSize <= lonSize) {
        // Divide latitude dimension
        size_t latChunkSize = latSize / numChunks;
        size_t remainder = latSize % numChunks;
        
        for (int r = 0; r < numChunks; ++r) {
            chunks[r].startLat = r * latChunkSize + std::min(static_cast<size_t>(r), remainder);
            chunks[r].numLat = latChunkSize + (r < remainder ? 1 : 0);
            chunks[r].startLon = 0;
            chunks[r].numLon = lonSize;  // Full longitude range
        }
    } else {
        // Divide longitude dimension
        size_t lonChunkSize = lonSize / numChunks;
        size_t remainder = lonSize % numChunks;
        
        for (int r = 0; r < numChunks; ++r) {
            chunks[r].startLat = 0;
            chunks[r].numLat = latSize;  // Full latitude range
            chunks[r].startLon = r * lonChunkSize + std::min(static_cast<size_t>(r), remainder);
            chunks[r].numLon = lonChunkSize + (r < remainder ? 1 : 0);
        }
    }
    
    return chunks;
}
