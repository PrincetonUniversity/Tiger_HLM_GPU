// I_O/parameters_loader.cpp
#include "parameters_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

std::vector<SpatialParams> loadSpatialParams(const std::string& csv_path) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open parameter file: " + csv_path);
    }

    std::string line;
    // Read header
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty parameter file: " + csv_path);
    }
    std::vector<std::string> headers;
    {
        std::istringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ',')) {
            headers.push_back(col);
        }
    }

    // Map header name → column index
    std::unordered_map<std::string,int> idx;
    for (int i = 0; i < (int)headers.size(); ++i) {
        idx[headers[i]] = i;
    }

    // Required columns: match CSV's names
    const std::vector<std::string> required = {
        "stream", // "next_stream",
        "i2", "i3",               // infiltration/percolation fractions
        "hu",                     // → SpatialParams::Hu
        "centroid_lon",           // → SpatialParams::lon
        "centroid_lat",           // → SpatialParams::lat
        "sw", "ss",
        "n",                      // → SpatialParams::n_mann
        "slope",
        "length_km",              // → SpatialParams::L
        // "drainage_area_km2",      // → SpatialParams::A_i
        "area_sqkm",              // → SpatialParams::A_h
        "melt",                   // → SpatialParams::melt_f
        "t_thres",                // → SpatialParams::temp_thr
        "res_ss",                 // → to compute alpha3
        "res_gw"                  // → to compute alpha4
    };
    for (auto& name : required) {
        if (idx.find(name) == idx.end()) {
            throw std::runtime_error("Missing column '" + name + "' in " + csv_path);
        }
    }

    // Conversion constant: mm/hr → m/min
    constexpr double c_1 = 0.001 / 60.0;
    // We no longer read c1 from CSV; use c_1 as a fixed conversion factor.

    std::vector<SpatialParams> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<std::string> fields;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            fields.push_back(cell);
        }
        if ((int)fields.size() < (int)headers.size())
            throw std::runtime_error("Bad row with too few fields in " + csv_path);

        SpatialParams p;
        p.stream      = std::stol(fields[idx["stream"]]);
        // Optional next_stream: default to -1 when the column is absent or blank
        auto it_next = idx.find("next_stream");
        if (it_next != idx.end()) {
            const std::string& v = fields[it_next->second];
            if (!v.empty())
                p.next_stream = std::stol(v);
            else
                p.next_stream = -1;
        } else {
            p.next_stream = -1;
        }

        // store conversion factor for downstream code
        p.c1 = 0.001/60.0;

        // direct mappings
        p.Hu       = std::stod(fields[idx["hu"]]); // depth of static tank [mm] 
        p.lat      = std::stod(fields[idx["centroid_lat"]]); // hillslope latitude in degrees for PET calculation [unitless]
        // pass longitude here from the params file
        p.lon      = std::stod(fields[idx["centroid_lon"]]); // hillslope longitude in degrees for PET calculation [unitless]
        p.sw       = std::stod(fields[idx["sw"]]); // relative soil moisture wilting point [unitless]
        p.ss       = std::stod(fields[idx["ss"]]); // relative soil moisture point of stomatal closure [unitless]
        p.n_mann   = std::stod(fields[idx["n"]]); // Manning's n coefficient [unitless]
        p.slope    = std::stod(fields[idx["slope"]]); // average slope of hillslope [m/m]
        p.L        = std::stod(fields[idx["length_km"]]);   // length of channel [km] 
        p.A_h      = std::stod(fields[idx["area_sqkm"]]); // area of hillslopes in [km²]
        // p.A_i = std::stod(fields[idx["drainage_area_km2"]]); // drainage area in [km²], don't need this parameter for runoff !!! remove it
        p.melt_f   = std::stod(fields[idx["melt"]]); 
        p.temp_thr = std::stod(fields[idx["t_thres"]]); // temperature threshold for snowmelt [°C]

        double res_ss = std::stod(fields[idx["res_ss"]]); // residence time in subsurface tank [days]
        double res_gw = std::stod(fields[idx["res_gw"]]); // residence time in groundwater tank [days]

        // compute infiltration & percolation rates
        double i2 = std::stod(fields[idx["i2"]]); // infiltration fraction [mm/hr]
        double i3 = std::stod(fields[idx["i3"]]); // percolation fraction [mm/hr]
        p.infil   = i2 * p.c1;  // infiltration rate [m/min]
        p.perco   = i3 * p.c1;  // percolation rate to aquifer [m/min]

        // compute residence-time parameters in minutes
        p.alpha3 = res_ss * 24.0 * 60.0;  // days → minutes
        p.alpha4 = res_gw * 24.0 * 60.0;  // days → minutes
        
        // convert melt factor from mm/day/degree to m/min/degree
        p.melt_f = p.melt_f *(1/(24*60.0))*(1/1000.0);  // mm/day/degree to m/min/degree

        // convert units
        p.Hu  = p.Hu / 1000.0; // convert Hu from mm to m
        p.L   = p.L * 1000;   // length of channel [km] -> [m]
        p.A_h = p.A_h * 1000000; // area of hillslopes in [km²] -> [m²]

        out.push_back(p);
    }

    return out;
}
