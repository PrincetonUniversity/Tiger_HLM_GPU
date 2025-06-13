// src/I_O/forcing_loader.hpp
#pragma once

#include <netcdf.h>
//#include <netcdf_par.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>  // for std::pair
#include <vector>

// Structure to hold spatial chunk information
// Current logic is simple: divide the lat/lon grid into rectangular chunks based on # of processes (ranks)
struct SpatialChunk {
    size_t startLat, numLat;    // Start index and number of latitudes in the chunk
    size_t startLon, numLon;    
    
    // Constructor for convenience
    SpatialChunk(size_t sLat = 0, size_t nLat = 0, size_t sLon = 0, size_t nLon = 0)
        : startLat(sLat), numLat(nLat), startLon(sLon), numLon(nLon) {}
};


class LookupMapper {
public:
    explicit LookupMapper(const std::string& filepath);

    // Load the CSV into the map
    bool load();

    // Check if a stream ID exists
    bool hasStream(long long stream_id) const;

    // Get the (lat_index, lon_index) for a given stream
    std::pair<int, int> getLatLon(long long stream_id) const;

    // Get size of the map
    size_t size() const;

private:
    std::string filepath_;
    std::unordered_map<long long, std::pair<int, int>> stream_map_;
};

class NetCDFLoader {
private:
    int ncid;           // NetCDF file ID
    int varid;          // Variable ID
    size_t timeSize, latSize, lonSize;
    std::string fileName;
    std::string varName;
 
    // Helper function to check NetCDF errors
    void checkError(int status, const std::string& operation);
 
public:
    // Constructor
    NetCDFLoader(const std::string& filename, const std::string& varName);
 
    // Destructor
    ~NetCDFLoader();
 
    // Disable copy constructor and assignment operator
    NetCDFLoader(const NetCDFLoader&) = delete;
    NetCDFLoader& operator=(const NetCDFLoader&) = delete;
 
    // Enable move constructor and assignment operator
    NetCDFLoader(NetCDFLoader&& other) noexcept;
    NetCDFLoader& operator=(NetCDFLoader&& other) noexcept;
 
    // Load data by time chunk into memory
    std::unique_ptr<float[]> loadTimeChunk(size_t startTime, size_t numTimeSteps);

    // Load data by any chunk: time/lat/lon into memory
    std::unique_ptr<float[]> loadChunk(size_t startTime, size_t numTime,    // start index of time dimension, number of time steps in loading chunk
                                       size_t startLat, size_t numLat,
                                       size_t startLon, size_t numLon);
 
    // Getters
    size_t getTimeSize() const { return timeSize; }
    size_t getLatSize() const { return latSize; }
    size_t getLonSize() const { return lonSize; }
    std::string getVariableName() const { return varName; }
    std::string getFileName() const { return fileName; }
 
    // Get value at a specific time, latitude, and longitude, in the loaded time chunk
    // static float getValueFromChunk(const std::unique_ptr<float[]>& chunkData,
    //                           size_t relativeTimeIndex,  
    //                           size_t latIndex, size_t lonIndex,
    //                           size_t latSize, size_t lonSize);
                              
    float getValueFromChunk(const std::unique_ptr<float[]>& chunkData,
                       size_t relativeTimeIndex, // 0 to chunkSize-1: position within the loaded chunk
                       size_t latIndex, size_t lonIndex,
                       size_t chunkTimeSize, size_t latSize, size_t lonSize);
    
    // Return NetCDF file and variable IDs for advanced operations: maybe needed in the future
    int getFileId() const { return ncid; }
    int getVariableId() const { return varid; }  
    
    // Calculate chunks for this NetCDF file
    std::vector<SpatialChunk> calculateSpatialChunks(size_t latSize, size_t lonSize, int numChunks);
};


