// src/I_O/forcing_loader.hpp
#pragma once

#include <vector>
#include <string>

/**
 * (Optional) A stub for loading forcing data from CSV or NetCDF.
 * For the dummy model, we are not using any forcing, so these functions can be left unimplemented.
 */

bool loadCsvForcing(const std::string& filename, std::vector<double>& times, std::vector<double>& values);
bool loadNetcdfForcing(const std::string& filename, std::vector<double>& times, std::vector<double>& values);
