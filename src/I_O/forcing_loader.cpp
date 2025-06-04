// src/I_O/forcing_loader.cpp

#include "forcing_loader.hpp"

// (No real implementation for the dummy example; these return false or empty.)

bool loadCsvForcing(const std::string& /*filename*/, std::vector<double>& /*times*/, std::vector<double>& /*values*/) {
    return false;
}

bool loadNetcdfForcing(const std::string& /*filename*/, std::vector<double>& /*times*/, std::vector<double>& /*values*/) {
    return false;
}
