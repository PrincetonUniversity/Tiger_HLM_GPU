#pragma once
// models/ETMethods.hpp

#include <cmath>

namespace ETMethods {

// ───────── HamonPET ─────────
// Estimates daily PET [m/min] via the Hamon (CBM) model.
//
// @param temperature  Daily average air temperature [°C]
// @param latitude     Hillslope latitude [°]
// @param doy          Day‐of‐year [1–365]
// @returns            PET in m/min
__host__ __device__
double HamonPET(double temperature,
                double latitude,
                double doy);

// ───────── ETactual ─────────
// Computes actual evapotranspiration based on Amilcare textbook.
// Linear decrease between wilting (sw) and stomatal closure (ss).
//
// @param Emax  Maximum evaporation [m/min]
// @param s     Relative soil moisture [-]
// @param sw    Wilting‐point threshold [-]
// @param ss    Stomatal‐closure threshold [-]
// @returns     Actual ET [m/min]
__host__ __device__
double ETactual(double Emax,
                double s,
                double sw,
                double ss);

} // namespace ETMethods
