#pragma once
#include "solver/rk45.h"

struct DummyModel {
    static constexpr unsigned short UID = 10;
    static constexpr int        N_EQ = 5;

    struct Parameters {
        double initialStep;
        double safety;
        double minScale;
        double maxScale;
    };

    __device__ static void rhs(double /*t*/, const double* y, double* dydt, int /*n*/) {
        double H0 = y[0];
        double H1 = y[1];
        double H2 = y[2];  // (unused but kept for size)
        double H3 = y[3];
        double H4 = y[4];  // (unused but kept for size)

        // … same RHS code as before …
        double X0 = 1.0;
        double Y0 = 0.5 * H0;
        double X1 = 1.2;
        double X2 = 0.3 * H1;
        double Y1 = 0.4;
        double Y2 = 0.2;
        double Y3 = 0.3;
        double Y4 = 0.1;
        double I2 = 0.6;
        double I3 = 0.4 * H3;

        dydt[0] = X0 - Y0;
        dydt[1] = X1 + Y0 - X2 - Y1 - I2;
        dydt[2] = X2 - Y2;
        dydt[3] = I2 - I3 - Y3;
        dydt[4] = I3 - Y4;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Declare `devParams` in constant memory.  No definition/initializer here.
// ─────────────────────────────────────────────────────────────────────────────
extern __constant__ DummyModel::Parameters devParams;

// Declare the kernel function
extern __global__ void checkDevParamsKernel();