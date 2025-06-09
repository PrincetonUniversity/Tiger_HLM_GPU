//models/model_dummy.hpp
#pragma once
#include "solver/rk45.h"

struct DummyModel {
    static constexpr unsigned short UID = 10;
    static constexpr int        N_EQ = 5;

    struct Parameters {
        double initialStep = 0.01;  // default if the user never sets it
        double rtol        = 1e-6;  // default relative tolerance
        double atol        = 1e-9;  // default absolute tolerance
        double safety      = 0.9;   // default safety factor
        double minScale    = 0.2;   // default minimum scale
        double maxScale    = 10.0;  // default maximum scale
    };

    __host__ __device__ static void rhs(double /*t*/, const double* y, double* dydt, int /*n*/, int /*sys*/) {
        double H0 = y[0];
        double H1 = y[1];
        double H2 = y[2];  
        double H3 = y[3];
        double H4 = y[4];  

        // RHS code for the dummy model (same as the python version)
        double X0 = 1.0;
        double Y0 = 0.5 * H0;
        double X1 = 1.2;
        double X2 = 0.3 * H1;
        double Y1 = 0.4;
        double Y2 = 0.2;
        double Y3 = 0.3;
        double Y4 = 0.1;
        double I2 = 0.6 * H1;
        double I3 = 0.4 * H3;
        
        // Compute the derivatives
        // dH0/dt = X0 - Y0
        dydt[0] = X0 - Y0;
        // dH1/dt = X1 + Y0(H0) - X2 - Y1 - I2
        dydt[1] = X1 + Y0 - X2 - Y1 - I2;
        // dH2/dt = X2(H1) - Y2
        dydt[2] = X2 - Y2;
        // dH3/dt = I2(H1) - I3 - Y3
        dydt[3] = I2 - I3 - Y3;
        // dH4/dt = I3(H3) - Y4
        dydt[4] = I3 - Y4;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Declare `devParams` in constant memory.  No definition/initializer here.
// Definition lives in model_dummy_global.cu.
// ─────────────────────────────────────────────────────────────────────────────
//extern __constant__ DummyModel::Parameters devParams;
// only if USE_DUMMY_MODEL is defined do we pull in the extern:
//extern __constant__ ActiveModel::Parameters devParams;


// ─────────────────────────────────────────────────────────────────────────────
// Declare the kernel function (optional, for test launches).
// ─────────────────────────────────────────────────────────────────────────────
extern __global__ void checkDevParamsKernel();
