// src/I_O/forcing_loader.hpp
#pragma once

#include <netcdf.h>
#include <memory>
#include <string>

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

    // Check if data is loaded correctly
    bool isDataLoaded() const;

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
};
