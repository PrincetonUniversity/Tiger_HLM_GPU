#pragma once

#include "solver/rk45.h"
#include <cmath>
#include "ETmethods.hpp"          // for ETMethods::HamonPET, ETMethods::ETactual
#include "soiltemp.hpp"           // for SoilTemp::soiltemp
#include "parameters_loader.hpp"  // for SpatialParams

// bring them into the local namespace so we can write HamonPET(...) etc.
using ETMethods::HamonPET;
using ETMethods::ETactual;
using SoilTemp::soiltemp;

/// Model204: 5‐equation “snow, static, surface, grav, aquifer” runoff model.
struct Model204
{
    using SP_TYPE = SpatialParams;    ///< alias for the spatial‐params struct
    static constexpr unsigned short UID = 204;
    static constexpr int N_EQ = 5;    ///< number of equations

    /// RK45‐tolerance parameters (populated at runtime via model_registry)
    struct Parameters
    {
        double initialStep = 0.01;
        double rtol        = 1e-6;
        double atol        = 1e-9;
        double safety      = 0.9;
        double minScale    = 0.2;
        double maxScale    = 10.0;
    };

    /**
     * RHS of the ODE system y' = f(t,y).
     *
     * y[0] = h_snow
     * y[1] = h_static
     * y[2] = h_surface
     * y[3] = h_grav
     * y[4] = h_aquifer
     *
     * Now pulls per‐stream parameters from sp_ptr[sys].
     */
    __host__ __device__
    static void rhs(double t,
                    const double *y,
                    double *dydt,
                    int /*n*/,
                    int sys,   // which stream index we’re on
                    const SpatialParams* sp_ptr)
    {
        const SpatialParams &P = sp_ptr[sys];

        // — unpack the state vector y —
        double h_snow = y[0];
        double h_stat = y[1];
        double h_surf = y[2];
        double h_grav = y[3];
        double h_aq   = y[4];

        // — real spatial parameters —
        double c1     = P.c1;
        double infil  = P.infil;
        double perco  = P.perco;
        double Hu     = P.Hu;
        double lat    = P.lat;
        double sw     = P.sw, ss = P.ss;
        double n_mann = P.n_mann, slope = P.slope;
        double L      = P.L,    A_h = P.A_h;
        double alpha3 = P.alpha3, alpha4 = P.alpha4;
        double melt_f = P.melt_f,  temp_thr = P.temp_thr;

        // — stub forcings (for now) —
        double rainfall    = 0.001;        // [m/min]
        double temperature = 1.0;          // [°C]
        double doy         = 1.0 + t/1440; // day‐of‐year 

        // 1) Snow
        double snowmelt = (temperature >= temp_thr)
                              ? fmin(h_snow, temperature * melt_f)
                              : 0.0;
        double x1       = rainfall + snowmelt;
        dydt[0]         = rainfall - snowmelt;

        // 2) Static
        double x2  = fmax(0.0, x1 + h_stat - Hu);
        double d1  = x1 - x2;
        double Emax= fmin(0.1 * temperature, h_stat);
        double s   = h_stat / Hu;
        dydt[1]    = d1 - s * Emax;

        // 3) Surface
        double x3    = fmin(x2, infil);
        double d2    = x2 - x3;
        double alfa2 = (1.0 / n_mann) * pow(h_surf, 2.0/3.0) * sqrt(slope);
        double w     = fmin(1.0, alfa2 * L / A_h * 60.0);
        dydt[2]      = d2 - h_surf * w;

        // 4) Gravitational (interflow)
        double x4    = fmin(x3, perco);
        double d3    = x3 - x4;
        dydt[3]      = d3 - (alpha3 >= 1.0 ? h_grav / alpha3 : 0.0);

        // 5) Aquifer (baseflow)
        dydt[4]      = x4 - (alpha4 >= 1.0 ? h_aq / alpha4 : 0.0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// These symbols are defined in models/model_204_global.cu
// ─────────────────────────────────────────────────────────────────────────────
extern __constant__ Model204::Parameters devParams;
extern __constant__ SpatialParams*      devSpatialParamsPtr;

// ─────────────────────────────────────────────────────────────────────────────
// Optional kernel to inspect devParams on the device
// ─────────────────────────────────────────────────────────────────────────────
__global__ void checkDevParamsKernel204();
