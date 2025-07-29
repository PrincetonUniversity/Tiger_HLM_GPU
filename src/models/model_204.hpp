#pragma once

#include "solver/rk45.h"
#include <cmath>
#include "ETmethods.hpp"          // for ETMethods::HamonPET, ETMethods::ETactual
#include "soiltemp.hpp"           // for SoilTemp::soiltemp
#include "parameters_loader.hpp"  // for SpatialParams
#include "I_O/forcing_data.h"    // for d_forc_data, nForc

// bring them into the local namespace so we can write HamonPET(...) etc.
using ETMethods::HamonPET;
using ETMethods::ETactual;
using SoilTemp::soiltemp;

// Model204: 5‐equation “snow, static, surface, grav, aquifer” runoff model.
struct Model204
{
    using SP_TYPE = SpatialParams;    // alias for the spatial‐params struct
    static constexpr unsigned short UID = 204;
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
        STATE_SURF_RUNOFF = 7, // new: surface runoff
        // STATE_SUBSURF_RUNOFF = 8 // new: subsurface runoff
        STATE_TOTAL_RUNOFF = 8 // new: total runoff
    };


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
    

        // ── 0) unpack & set states ─────────────────────────────
        double h_snow      = fmax(0.0, y[STATE_SNOW]);
        double h_stat      = fmax(0.0, y[STATE_STATIC]);
        double h_surf      = fmax(0.0, y[STATE_SURFACE]);
        double h_grav      = fmax(0.0, y[STATE_GRAV]);
        double h_aq        = fmax(0.0, y[STATE_AQUIFER]); //update states when reach 0 !!!

        // ── 1) unpack previous temperatures ───────────────────────
        double T_air_prev  = y[STATE_TEMP_AIR];
        double T_soil_prev = y[STATE_TEMP_SOIL];

        // ── 2) spatial params ────────────────────────────────────
        const auto &P   = sp_ptr[sys];
        double c1       = P.c1;      // mm/hr → m/min
        double Hu       = P.Hu;
        double infil    = P.infil;
        double perco    = P.perco;
        double lat      = P.lat;
        double sw       = P.sw;
        double ss       = P.ss;
        double n_mann   = P.n_mann;
        double slope    = P.slope;
        double L        = P.L;
        double A_i      = P.A_i;     
        double A_h      = P.A_h;
        double alpha3   = P.alpha3;
        double alpha4   = P.alpha4;
        double melt_f   = P.melt_f;
        double temp_thr = P.temp_thr;

        // ── 3) forcings  ─────────────────────────────────────────
        //double c1         = 0.001/60.0;                   // mm/hr → m/min
        double rainfall   = (nForc>0 ? F[0]*c1 : 0.0);
        double temperature= (nForc>1 ? F[1]    : 0.0);
        double doy        = 1.0 + t/1440.0;

        // ── 4) compute ET for static tank─────────────────────
        double pet    = HamonPET(temperature, lat, doy); // potential evapotranspiration [m/min]
        double Emax   = fmin(pet, h_stat); // maximum possible evapotranspiration [m/min] from static tank, cannot be more than h1 [m]
        double s_stat = h_stat/Hu; // relative soil moisture [unitless]
        double out1   = ETactual(Emax, s_stat, sw, ss); // actual evapotranspiration [m/min] based on wilting point and stress factor

        // ── 5) soil‐temperature & freeze flag ────────────────────
        double soil_temp = T_soil_prev;
        if (temperature != T_air_prev) {
            soil_temp = soiltemp(temperature, T_soil_prev, h_snow);
            
        }
        bool frozen_ground = (soil_temp <= 0.0);

        // ── 6) temperature‐state derivatives ────────────────────
        dydt[STATE_TEMP_AIR ] = - T_air_prev + temperature;
        dydt[STATE_TEMP_SOIL] = - T_soil_prev + soil_temp;

        // ── 7) snow tank ──────────────────────────────────────
        double x1 = 0.0;
        if (temperature == 0.0) {
            x1 = rainfall;
            dydt[STATE_SNOW] = 0.0;
        }
        else if (temperature < temp_thr) {
            x1 = 0.0;
            dydt[STATE_SNOW] = rainfall;
        }
        else {
            double snowmelt = fmin(h_snow, temperature * melt_f);
            x1 = rainfall + snowmelt;
            dydt[STATE_SNOW] = -snowmelt;
        }

        // ── 7) snow tank new ────────────────────────────────────── 
        // double x1 = 0.0;
        // double melt_thr = 0.0; // melt threshold
        // if (temperature < temp_thr)//temp_thr is the accumulation threshold
        // { 
        //     // If temperature is below accumulation threshold, all rainfall accumulates as snow  
        //     x1 = 0.0;
        //     dydt[STATE_SNOW] = rainfall;
        //     //Accumulate precip as snow and melt out
        //     if (temperature > melt_thr){
        //         double snowmelt = fmin(h_snow, temperature * melt_f);
        //         x1 = snowmelt;
        //         dydt[STATE_SNOW] = -snowmelt; 
        //     }                     
        // }
        // else if (temperature >= temp_thr){ 
        //     //default for tmp< mlt_thr
        //     double x1 = rainfall;
        //     dydt[STATE_SNOW] = 0;
        //     //default for tmp>= mlt_thr
        //     if (temperature > melt_thr){
        //         double snowmelt = fmin(h_snow, temperature * melt_f);
        //         x1 = rainfall + snowmelt;
        //         dydt[STATE_SNOW] = -snowmelt;
        //     }
        // }

        // ── 8) static tank ─────────────────────────────────────
        double x2 = fmax(0.0, x1 + h_stat - Hu); // water that enters second storage (surface) tank [m/min]
        if (frozen_ground) x2 = x1; // if frozen, all water goes to surface tank
        double d1 = x1 - x2; // input to static tank [m/min]
        dydt[STATE_STATIC] = d1 - out1;

        // ── 9) surface tank ────────────────────────────────────
        double infil_eff = (frozen_ground ? 0.0 : infil);
        double x3        = fmin(x2, infil_eff); // water that infiltrates to gravitational storage [m/min]
        double d2        = x2 - x3; // input to surface tank [m/min]
        double alfa2     = (1.0/n_mann) * pow(h_surf,2.0/3.0)*sqrt(slope);
        double out2      = h_surf * fmin(1.0, alfa2*L/A_h*60.0);
        dydt[STATE_SURFACE] = d2 - out2;

        // ── 10) subsurface (gravitational) ──────────────────────
        double x4   = fmin(x3, perco);
        double d3   = x3 - x4;
        double out3 = (alpha3>=1.0 ? h_grav/alpha3 : 0.0);
        dydt[STATE_GRAV] = d3 - out3;

        // ── 11) aquifer (groundwater) ─────────────────────────────
        double d4   = x4;
        double out4 = (alpha4>=1.0 ? h_aq/alpha4 : 0.0);
        dydt[STATE_AQUIFER] = d4 - out4;

        // ── 12) surface and subsurface runoff ─────────────────────────────────
        dydt[STATE_SURF_RUNOFF]    = - y[STATE_SURF_RUNOFF] + out2/c1; // m/min to mm/hr
        // dydt[STATE_SUBSURF_RUNOFF] = out3 + out4; // instead save total runoff
        double out_total = (out2 + out3 + out4) / c1; // instantaneous total runoff [mm/hr]
        dydt[STATE_TOTAL_RUNOFF] = - y[STATE_TOTAL_RUNOFF] + out_total; // m/min to mm/hr
    


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
// __global__ void checkDevParamsKernel204();
