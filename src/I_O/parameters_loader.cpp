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

    // Required columns: match your CSV's names
    const std::vector<std::string> required = {
        "stream", "next_stream",
        "i2", "i3",               // infiltration/percolation fractions
        "hu",                     // → SpatialParams::Hu
        "centroid_lat",           // → SpatialParams::lat
        "sw", "ss",
        "n",                      // → SpatialParams::n_mann
        "slope",
        "length_km",              // → SpatialParams::L
        "drainage_area_km2",      // → SpatialParams::A_h
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
        p.next_stream = std::stol(fields[idx["next_stream"]]);

        // store conversion factor for downstream code
        p.c1 = c_1;

        // direct mappings
        p.Hu       = std::stod(fields[idx["hu"]]);
        p.lat      = std::stod(fields[idx["centroid_lat"]]);
        p.sw       = std::stod(fields[idx["sw"]]);
        p.ss       = std::stod(fields[idx["ss"]]);
        p.n_mann   = std::stod(fields[idx["n"]]);
        p.slope    = std::stod(fields[idx["slope"]]);
        p.L        = std::stod(fields[idx["length_km"]]);
        p.A_h      = std::stod(fields[idx["drainage_area_km2"]]);
        p.melt_f   = std::stod(fields[idx["melt"]]);
        p.temp_thr = std::stod(fields[idx["t_thres"]]);

        // compute infiltration & percolation rates
        double i2 = std::stod(fields[idx["i2"]]);
        double i3 = std::stod(fields[idx["i3"]]);
        p.infil = i2 * c_1;  // infiltration rate [m/min]
        p.perco = i3 * c_1;  // percolation rate to aquifer [m/min]

        // compute residence-time parameters in minutes
        double res_ss = std::stod(fields[idx["res_ss"]]);
        double res_gw = std::stod(fields[idx["res_gw"]]);
        p.alpha3 = res_ss * 24.0 * 60.0;  // days → minutes
        p.alpha4 = res_gw * 24.0 * 60.0;  // days → minutes

        out.push_back(p);
    }

    return out;
}
