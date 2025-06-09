// models/ETmethods.cpp

#include "models/ETmethods.hpp"
#include <cmath>

namespace ETMethods {

// ───────── HamonPET ─────────
// Implements Hamon PET estimation.
// See header for documentation.
double HamonPET(double temperature,
                double latitude,
                double doy)
{
    double PET = 0.0;
    if (temperature > 0.0) {
        // Saturation vapor pressure (mb)
        double esat = 6.108 * std::exp((17.26939 * temperature) / (temperature + 237.3));
        // Saturated water vapor (g/m³)
        double Wt   = 216.7 * esat / (temperature + 273.3);

        // Daylight fraction (per 12 h) via CBM model
        double theta = 0.2163108 + 2.0 * std::atan(0.9671396 * std::tan(0.00860 * (doy - 186.0)));
        double phi   = std::asin(0.39795 * std::cos(theta));
        const double PI = 3.14159265358979323846;
        double D = (24.0 - (24.0/PI) * std::acos((std::sin(0.8333 * PI/180.0)
                     + std::sin(latitude * PI/180.0) * std::sin(phi))
                     /(std::cos(latitude * PI/180.0) * std::cos(phi)))) / 12.0;

        // Arctic handling
        if (std::isnan(D)) {
            D = 0.0;
            if ((phi > 0.0 && latitude > 0.0) || (phi < 0.0 && latitude < 0.0)) {
                D = 2.0;
            }
        }

        // PET [m/min]
        PET = 1.6169e-6 * D * D * Wt * 60.0 / 1000.0;
    }
    return PET;
}

// ───────── ETactual ─────────
// Implements actual ET estimation.
// See header for documentation.
double ETactual(double Emax,
                double s,
                double sw,
                double ss)
{
    double ETa = 0.0;
    if (s > sw && s <= ss) {
        ETa = Emax * (s - sw) / (ss - sw);
    } else if (s > ss) {
        ETa = Emax;
    }
    return ETa;
}

} // namespace ETMethods
