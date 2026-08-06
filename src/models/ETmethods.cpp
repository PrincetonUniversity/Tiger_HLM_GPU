// models/ETmethods.cpp

#include "models/ETmethods.hpp"
#include <cmath>

namespace ETMethods {

// ───────── OudinPET ─────────
// Implements Oudin PET estimation
// See header for documentation
double OudinPET(double temperature,
                double latitude,
                double doy)
{
    // Equation parameters optimized across different models and basins (Make as model parameters???)
    const double K1 = 100; // Calibration Parameter from study
    const double K2 = 5; // Temperature offset (c) calibration parameter from study 

    // PET Computation
    double PET = 0.0;
    if (temperature + K2 > 0){
        // Extraterrestrial radiation (MJ m-2 day-1)
        const double PI = 3.14159265358979323846;
        const double Gsc = 0.0820; // solar constant MJ m-2 min-1
        double dr = 1 + 0.033 * std::cos(2 * PI * doy / 365); // inverse realtive distance earth-sun 
        double delta = 0.409 * std::sin((2 * PI * doy / 365) - 1.39); // solar decimation
        double phi = latitude * (PI / 180.0); // Latitude in degrees    
        double ws = std::acos(-std::tan(phi) * std::tan(delta));
        double Re = (24 * 60 / PI) * Gsc * dr * (ws * std::sin(phi) * std::sin(delta) + std::cos(phi) * std::cos(delta) * std::sin(ws)); //extraterrestial radation
        // PET computation
        const double lhf = 2.45; // latent heat flux MJ kg-1
        const double rho = 1000; // density of water kg/m3
        PET = (Re / (lhf * rho)) * (temperature + K2) / K1; // m/day
        PET = PET / (24 * 60); //convert from m/day -> m/min
    }
    return PET;
}

// ───────── HamonPET ─────────
// Implements Hamon PET estimation
// See header for documentation
__host__ __device__
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
// Implements actual ET estimation
// See header for documentation
__host__ __device__
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
