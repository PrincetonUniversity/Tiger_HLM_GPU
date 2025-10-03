#pragma once

#include "../solver/rk45.h"
#include <cmath>
// Portable finite check for host + device
__host__ __device__ inline bool finite_val(double x) {
#if defined(__CUDA_ARCH__)
    // device build: use CUDA's global ::isfinite
    return ::isfinite(x);
#else
    // host build: use C++ <cmath>
    return std::isfinite(x);
#endif
}

#include "ETmethods.hpp"          // for ETMethods::HamonPET, ETMethods::ETactual
#include "soiltemp.hpp"           // for SoilTemp::soiltemp
#include "../I_O/parameters_loader.hpp"  // for SpatialParams
#include "../I_O/forcing_data.h"    // for d_forc_data, nForc

// bring them into the local namespace so we can write HamonPET(...) etc.
using ETMethods::HamonPET;
using ETMethods::ETactual;
using SoilTemp::soiltemp;

// Runoff5: 5‐equation “snow, static, surface, grav, aquifer” runoff model.
struct Runoff5
{
    using SP_TYPE = SpatialParams;    // alias for the spatial‐params struct
    static constexpr unsigned short UID = 5;
    //static constexpr int N_EQ = 7;    // number of equations in the ODE system
    static constexpr int N_EQ = 9;    // added surface runoff & total runoff
  
    // RK45‐tolerance parameters (populated at runtime via model_registry)
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
     * {0.01, 0.1, 0.0, 0.0, 0.01, 1, 1, 0, 0} 
     * 
     * y[0] = h_snow 0.01
     * y[1] = h_static 0.1
     * y[2] = h_surface 0.0
     * y[3] = h_grav 0.0
     * y[4] = h_aquifer 0.01
     * y[5] = T_air_prev 1
     * y[6] = T_soil_prev 1
     *
     * Now pulls per‐stream parameters from sp_ptr[sys].
     */

    // State indices for the ODE system
    enum State : int {
        STATE_SNOW        = 0,
        STATE_STATIC      = 1,
        STATE_SURFACE     = 2,
        STATE_GRAV        = 3,
        STATE_AQUIFER     = 4,
        STATE_TEMP_AIR    = 5,
        STATE_TEMP_SOIL   = 6,
        STATE_SURF_RUNOFF = 7, // surface runoff
        // STATE_SUBSURF_RUNOFF = 8 // subsurface runoff
        STATE_TOTAL_RUNOFF = 8 // total runoff
    };

    // models/model_Runoff5.hpp  (inside struct Runoff5)
    __host__ __device__ static inline void project_nonnegative(double* y) {
        y[STATE_SNOW]    = fmax(0.0, y[STATE_SNOW]);
        y[STATE_STATIC]  = fmax(0.0, y[STATE_STATIC]);
        y[STATE_SURFACE] = fmax(0.0, y[STATE_SURFACE]);
        y[STATE_GRAV]    = fmax(0.0, y[STATE_GRAV]);
        y[STATE_AQUIFER] = fmax(0.0, y[STATE_AQUIFER]);
        // temperatures & runoff states are not projected
    }

    // ---- my way to ensure non-negative states ----
    __host__ __device__ static inline double ensure_non_negative(double x) {
        return (x < 1e-12 ? 0.0 : x);
    }
    // -----------------------------------------------

    __host__ __device__
    static void rhs(double t,
                    const double *y,
                    double *dydt,
                    int /*n*/,
                    int sys,
                    const SpatialParams* sp_ptr,
                    const float*  F,
                    int nForc)
    {
    

        // ── 0) unpack & set states ──────────────────────────────
        // ensure non-negative states 
        double h_snow = ensure_non_negative(y[STATE_SNOW]);
        double h_stat = ensure_non_negative(y[STATE_STATIC]);
        double h_surf = ensure_non_negative(y[STATE_SURFACE]);
        double h_grav = ensure_non_negative(y[STATE_GRAV]);
        double h_aq   = ensure_non_negative(y[STATE_AQUIFER]);
        // ───────────────────────────────────────────────────────

        // ── 1) unpack previous temperatures ───────────────────────
        double T_air_prev  = y[STATE_TEMP_AIR];
        double T_soil_prev = y[STATE_TEMP_SOIL];

        // ── 2) spatial params ────────────────────────────────────
        const auto &P   = sp_ptr[sys];
        double c1       = P.c1;      // mm/hr → m/min
        double Hu       = P.Hu; // depth of static tank [m], converted from mm to m in parameters_loader.cpp
        double infil    = P.infil;//4*0.001/60.0; //P.infil; // i2 * p.c1;  // infiltration rate [m/min]  4*0.001/60.0 doesnt work, divide by 10 one more time
        double perco    = P.perco;//2.0*0.001/60.0; //P.perco; // i3 * p.c1;  // percolation rate to aquifer [m/min] 2.0*0.001/60.0 doesnt work, divide by 10 one more time
        double lat      = P.lat; // hillslope latitude in degrees for PET calculation [unitless]
        double sw       = P.sw; // relative soil moisture wilting point [unitless]
        double ss       = P.ss; // relative soil moisture point of stomatal closure [unitless]
        double n_mann   = P.n_mann; // Manning's n coefficient [unitless]
        double slope    = P.slope; // average slope of hillslope [m/m]
        double L        = P.L;  // length of channel [m]
        double A_i      = P.A_i;  // drainage area in [m²], don't need this parameter for runoff !!! remove it
        double A_h      = P.A_h; // area of hillslopes in [m²]
        double alpha3   = P.alpha3; // res_ss * 24.0 * 60.0; //days → minutes
        double alpha4   = P.alpha4; // res_gw * 24.0 * 60.0; //days → minutes
        double melt_f   = P.melt_f; // melt factor [m/day/°C] is what we passed to the model
        double temp_thr = P.temp_thr; // temperature threshold for snowmelt [°C]

        // ── 3) forcings  ─────────────────────────────────────────
        //double c1         = 0.001/60.0;                   // mm/hr → m/min
        double rainfall    = (nForc>0 ? F[0]*c1 : 0.0); // rainfall rate [m/min], convert from mm/hr to m/min
        double temperature = (nForc>1 ? F[1]    : 0.0);
        if (!finite_val(rainfall))    rainfall = 0.0; // only NaN/±Inf → 0
        if (!finite_val(temperature)) temperature = 0.0; // only NaN/±Inf → 0
        double doy        = 1.0 + t/1440.0;

        // ── 4) compute ET for static tank─────────────────────
        double pet    = HamonPET(temperature, lat, doy); // potential evapotranspiration [m/min]
        // double Emax   = fmin(pet, h_stat); // maximum possible evapotranspiration [m/min] from static tank, cannot be more than h1 [m]
        double pet_clamped = fmax(0.0, pet); // !!! clamp PET to non-negative
        double Emax = fmin(h_stat, fmax(0.0, pet_clamped));   // clamp PET, cap by storage

        //double s_stat = h_stat/Hu; // relative soil moisture [unitless] !!!
        const double Hu_safe = fmax(1e-9, Hu);      // avoid divide-by-zero
        double s_stat = h_stat / Hu_safe;           // relative soil moisture [unitless]
        s_stat = fmin(1.0, fmax(0.0, s_stat));      // bound to [0,1]
        
        double out1   = ETactual(Emax, s_stat, sw, ss); // actual evapotranspiration [m/min] based on wilting point and stress factor

        // ── 5) soil‐temperature & freeze flag ────────────────────
        double soil_temp = T_soil_prev;
        // if (temperature != T_air_prev) // temperature has changed
        // Use a tolerance to avoid float-equality traps
        constexpr double TEMP_EPS = 1e-6;
        if (fabs(temperature - T_air_prev) > TEMP_EPS) {
            soil_temp = soiltemp(temperature, T_soil_prev, h_snow);
        }
        bool frozen_ground = (soil_temp <= 0.0);

        // ── 6) temperature‐state derivatives ────────────────────
        dydt[STATE_TEMP_AIR ] = - T_air_prev + temperature;
        dydt[STATE_TEMP_SOIL] = - T_soil_prev + soil_temp;

        // ── 7) snow tank new ────────────────────────────────────── 
        double x1 = 0.0;  
        double melt_thr = 0.0; // melt threshold
    
        if (temperature < temp_thr) // temp_thr is the accumulation threshold
        { 
            // If temperature is below accumulation threshold, all rainfall accumulates as snow  
            x1 = 0.0; 
            dydt[STATE_SNOW] = rainfall;
            // Accumulate precip as snow and melt out
            if (temperature > melt_thr){
                double snowmelt = fmin(h_snow, temperature * melt_f);
                x1 = snowmelt;
                dydt[STATE_SNOW] = rainfall-snowmelt; // snowfall minus snowmelt
            }                     
        }
        else if (temperature >= temp_thr){ 
            // If temperature is above or equal to accumulation threshold, all rainfall is rain
            x1 = rainfall;
            dydt[STATE_SNOW] = 0;
            // Melt snow if temperature is above melt threshold
            if (temperature > melt_thr){
                double snowmelt = fmin(h_snow, temperature * melt_f);
                x1 = rainfall + snowmelt;
                dydt[STATE_SNOW] = -snowmelt;
            }
        }

        // ── 8) static tank ─────────────────────────────────────
        double x2 = fmax(0.0, x1 + h_stat - Hu); // water that enters second storage (surface) tank [m/min]
        if (frozen_ground) x2 = x1; // if frozen, all water goes to surface tank
        double d1 = x1 - x2; // input to static tank [m/min]
        dydt[STATE_STATIC] = d1 - out1;

        // POSITIVITY GUARD (add this line right here)
        if (h_stat <= 0.0 && dydt[STATE_STATIC] < 0.0) {
            dydt[STATE_STATIC] = 0.0;
        }

        // ── 9) surface tank ────────────────────────────────────
        double infil_eff = (frozen_ground ? 0.0 : infil);

        double x3        = fmin(x2, infil_eff); // water that infiltrates to gravitational storage [m/min]
        double d2        = x2 - x3; // input to surface tank [m/min]
        double alfa2     = (1.0/n_mann) * pow(h_surf,2.0/3.0)*sqrt(slope);
        // double out2      = h_surf * fmin(1.0, alfa2*L/A_h*60.0);
        // Hardened outflow: limit to storage, check finite & non-negative
        double out2 = h_surf * fmin(1.0, alfa2 * L / A_h * 60.0);
        out2 = fmin(out2, h_surf);
        if (!finite_val(out2) || out2 < 0.0) out2 = 0.0;

        dydt[STATE_SURFACE] = d2 - out2;

        // POSITIVITY GUARD
        if (h_surf <= 0.0 && dydt[STATE_SURFACE] < 0.0) dydt[STATE_SURFACE] = 0.0;

        // ── 10) subsurface (gravitational) ──────────────────────
        // Previous verison:
        double x4   = fmin(x3, perco); // water that percolates to aquifer [m/min]
        double d3   = x3 - x4; 

        // linear reservoir outflow (min 0); cap to available storage
        double out3 = (alpha3 >= 1.0 ? h_grav / alpha3 : 0.0);
        out3 = fmin(out3, h_grav);
        if (!finite_val(out3) || out3 < 0.0) out3 = 0.0;

        dydt[STATE_GRAV] = d3 - out3; 

        // POSITIVITY GUARD
        if (h_grav <= 0.0 && dydt[STATE_GRAV] < 0.0) dydt[STATE_GRAV] = 0.0;

        // ── 11) aquifer (groundwater) ─────────────────────────────
        double d4   = x4;
        // linear reservoir outflow (min 0); cap to available storage
        double out4 = (alpha4 >= 1.0 ? h_aq / alpha4 : 0.0);
        out4 = fmin(out4, h_aq);
        if (!finite_val(out4) || out4 < 0.0) out4 = 0.0;

        dydt[STATE_AQUIFER] = d4 - out4;

        // POSITIVITY GUARD
        if (h_aq <= 0.0 && dydt[STATE_AQUIFER] < 0.0) dydt[STATE_AQUIFER] = 0.0;

        // ── 12) runoff diagnostics as CUMULATIVE depths (mm) ──────────────────
        dydt[STATE_SURF_RUNOFF]  = (c1 > 0.0 ? out2 : 0.0);               // [m/min]
        dydt[STATE_TOTAL_RUNOFF] = (c1 > 0.0 ? (out2 + out3 + out4): 0.0);  // [m/min]




    }


};

// ─────────────────────────────────────────────────────────────────────────────
// These symbols are defined in models/model_Runoff5_global.cu
// ─────────────────────────────────────────────────────────────────────────────
extern __constant__ Runoff5::Parameters devParams;
extern __constant__ SpatialParams*      devSpatialParamsPtr;



// ─────────────────────────────────────────────────────────────────────────────
// Optional kernel to inspect devParams on the device
// ─────────────────────────────────────────────────────────────────────────────
// __global__ void checkDevParamsKernelRunoff5();
