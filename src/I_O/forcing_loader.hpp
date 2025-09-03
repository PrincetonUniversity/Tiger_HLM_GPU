#pragma once

#include <netcdf.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>  
#include <vector>
#include <filesystem>

#include "../stream.hpp"  // Stream<Runoff5> is defined in stream.hpp

namespace fs = std::filesystem;

// ─────────── File picking helper ───────────
std::vector<std::string>
select_ranged_files_overlapping(const fs::path& dir,
                                const std::string& prefix,
                                const std::string& suffix,
                                int Ys,int Ms,int Ds,
                                int Ye,int Me,int De);

// ─────────── LookupMapper ───────────
class LookupMapper {
public:
    explicit LookupMapper(const std::string& filepath);

    bool load();
    bool hasStream(long long stream_id) const;
    std::pair<int, int> getLatLon(long long stream_id) const;
    size_t size() const;

private:
    std::string filepath_;
    std::unordered_map<long long, std::pair<int, int>> stream_map_;
};

// ─────────── NetCDFLoader (simple 3D var reader) ───────────
class NetCDFLoader {
private:
    int ncid;           // NetCDF file ID
    int varid;          // Variable ID
    size_t timeSize, latSize, lonSize;
    std::string fileName;
    std::string varName;

    void checkError(int status, const std::string& operation);

public:
    NetCDFLoader(const std::string& filename, const std::string& varName);
    ~NetCDFLoader();

    NetCDFLoader(const NetCDFLoader&) = delete;
    NetCDFLoader& operator=(const NetCDFLoader&) = delete;

    NetCDFLoader(NetCDFLoader&& other) noexcept;
    NetCDFLoader& operator=(NetCDFLoader&& other) noexcept;

    std::unique_ptr<float[]> loadTimeChunk(size_t startTime, size_t numTimeSteps);
    bool isDataLoaded() const;

    size_t getTimeSize() const { return timeSize; }
    size_t getLatSize()  const { return latSize; }
    size_t getLonSize()  const { return lonSize; }
    std::string getVariableName() const { return varName; }
    std::string getFileName()     const { return fileName; }

    float getValueFromChunk(const std::unique_ptr<float[]>& chunkData,
                            size_t relativeTimeIndex,
                            size_t latIndex, size_t lonIndex,
                            size_t chunkTimeSize, size_t latSize, size_t lonSize);

    int getFileId() const { return ncid; }
    int getVariableId() const { return varid; }
};

// ─────────── Forcing configuration ───────────
struct ForcingEntry {
    std::vector<std::string> files; // files for this chunk (len=1 for yearly, N for ranged)
    std::string var_name;           // variable name inside the NetCDF
    double dt_hours;                // time resolution in hours

    // Dimension/coordinate names coming from config
    std::string dims_time;   // default "time"
    std::string dims_lat;    // default "latitude"
    std::string dims_lon;    // default "longitude"

    std::string lookup_csv;         // CSV mapping stream -> (lat_index, lon_index) for this forcing
};

// ─────────── NCForcing ───────────
class NCForcing {
public:
    std::vector<ForcingEntry> entries;
    std::vector<float> h_data;

    LookupMapper mapper;
    std::vector<long long> systemIds;

    size_t systems = 0;
    size_t days    = 0;
    size_t offset  = 0;

    int Ys=0, Ms=0, Ds=0;  // chunk start date
    int Ye=0, Me=0, De=0;  // chunk end   date

    int nForc = 0;
    std::vector<double> dt;
    std::vector<size_t> nT;

    NCForcing() : mapper("") {}
    explicit NCForcing(const std::string& mapperPath) : mapper(mapperPath) {}

    void loadData();
};
