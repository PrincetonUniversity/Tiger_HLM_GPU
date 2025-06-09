#pragma once
// models/soiltemp.hpp

#include <cmath>

namespace SoilTemp {

// ───────── soiltemp ─────────
// Computes daily‐scale soil temperature via Rankinen et al. 2002.
//
// @param Tair  Previous daily air temperature [°C]
// @param Tz    Previous daily ground temperature [°C]
// @param Ds    Snow depth [m]
// @returns     Updated soil temperature [°C]
__host__ __device__
double soiltemp(double Tair,
                double Tz,
                double Ds);

} // namespace SoilTemp
