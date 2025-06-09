// models/soiltemp.cpp

#include "models/soiltemp.hpp"
#include <cmath>

namespace SoilTemp {

// ───────── soiltemp ─────────
// Implements soil temperature change at daily scale.
// See header for documentation.
double soiltemp(double Tair,
                double Tz,
                double Ds)
{
    // Parameters from Rankinen et al. 2002
    const double Cs   = 1e6;    // J/m³/°C
    const double Kt   = 0.516;  // W/m/°C
    const double Cice = 8.93e6; // J/m³/°C
    const double fs   = -2.7;   // m⁻¹
    const double Zs   = 3.5e-2; // m (middle of 0–7 cm)
    const double dt   = 3600.0 * 24.0; // 1 day [s]
    const double CA   = Cs + Cice;
    const double f    = dt * Kt / (CA * (2*Zs)*(2*Zs)); // intermediate factor

    // Compute new temperature
    double T_star = Tz + f * (Tair - Tz);
    double Tz_out = T_star * std::exp(-fs * Ds);
    return Tz_out;
}

} // namespace SoilTemp
