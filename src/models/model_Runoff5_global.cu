// models/model_Runoff5_global.cu

#include "../models/model_Runoff5.hpp"
#include "../I_O/parameters_loader.hpp"   // for the free‐standing SpatialParams
#include <stdio.h>

#ifdef USE_MODEL_5

// ─────────────────────────────────────────────────────────────────────────────
// RK45 tolerance parameters (populated at runtime via setModelParameters<>() / cudaMemcpyToSymbol):
// ─────────────────────────────────────────────────────────────────────────────
__constant__ Runoff5::Parameters devParams;

// ─────────────────────────────────────────────────────────────────────────────
// Device‐constant pointer to the SpatialParams array in global memory
// (populated at runtime via cudaMemcpyToSymbol of a SpatialParams*):
// ─────────────────────────────────────────────────────────────────────────────
__constant__ SpatialParams* devSpatialParamsPtr;



// ─────────────────────────────────────────────────────────────────────────────
// Optional kernel to verify devParams on the device.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void checkDevParamsKernelRunoff5() {
    printf("Runoff5.devParams.initialStep = %g\n", devParams.initialStep);
    printf("Runoff5.devParams.rtol        = %g\n", devParams.rtol);
    printf("Runoff5.devParams.atol        = %g\n", devParams.atol);
    printf("Runoff5.devParams.safety      = %g\n", devParams.safety);
    printf("Runoff5.devParams.minScale    = %g\n", devParams.minScale);
    printf("Runoff5.devParams.maxScale    = %g\n", devParams.maxScale);
    printf("devSpatialParamsPtr = %p\n", (void*)devSpatialParamsPtr);
    printf("Runoff5.devParams.initialStep = %g\n", devParams.initialStep);
}

#endif  // USE_MODEL_5
