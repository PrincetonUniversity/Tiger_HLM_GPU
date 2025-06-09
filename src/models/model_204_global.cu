// // models/model_204_global.cu

// #include "models/model_204.hpp"
// #include <stdio.h>

// // ─────────────────────────────────────────────────────────────────────────────
// // Define & initialize the constant devParams for Model204:
// //
// // This provides the device‐side storage that model_registry.cpp will populate.
// // ─────────────────────────────────────────────────────────────────────────────
// __constant__ Model204::Parameters devParams;
// __constant__ Model204::SpatialParams devSpatialParams[];

// // ─────────────────────────────────────────────────────────────────────────────
// // Optional kernel to verify devParams on the device.
// // ─────────────────────────────────────────────────────────────────────────────
// __global__ void checkDevParamsKernel204() {
//     printf("Model204.devParams.initialStep = %g\n", devParams.initialStep);
//     printf("Model204.devParams.rtol        = %g\n", devParams.rtol);
//     printf("Model204.devParams.atol        = %g\n", devParams.atol);
//     printf("Model204.devParams.safety      = %g\n", devParams.safety);
//     printf("Model204.devParams.minScale    = %g\n", devParams.minScale);
//     printf("Model204.devParams.maxScale    = %g\n", devParams.maxScale);
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // **Do not** explicitly instantiate rk45_kernel_multi<Model204> here.
// // The template definition lives in rk45_kernel.cu which is compiled
// // with -dc and will generate the instantiation when you call it.
// // ─────────────────────────────────────────────────────────────────────────────
// models/model_204_global.cu

#include "models/model_204.hpp"
#include "I_O/parameters_loader.hpp"   // for the free‐standing SpatialParams
#include <stdio.h>

#ifdef USE_MODEL_204

// ─────────────────────────────────────────────────────────────────────────────
// RK45 tolerance parameters (populated at runtime via setModelParameters<>() / cudaMemcpyToSymbol):
// ─────────────────────────────────────────────────────────────────────────────
__constant__ Model204::Parameters devParams;

// ─────────────────────────────────────────────────────────────────────────────
// Device‐constant pointer to the SpatialParams array in global memory
// (populated at runtime via cudaMemcpyToSymbol of a SpatialParams*):
// ─────────────────────────────────────────────────────────────────────────────
__constant__ SpatialParams* devSpatialParamsPtr;

// ─────────────────────────────────────────────────────────────────────────────
// Optional kernel to verify devParams on the device.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void checkDevParamsKernel204() {
    printf("Model204.devParams.initialStep = %g\n", devParams.initialStep);
    printf("Model204.devParams.rtol        = %g\n", devParams.rtol);
    printf("Model204.devParams.atol        = %g\n", devParams.atol);
    printf("Model204.devParams.safety      = %g\n", devParams.safety);
    printf("Model204.devParams.minScale    = %g\n", devParams.minScale);
    printf("Model204.devParams.maxScale    = %g\n", devParams.maxScale);
    printf("devSpatialParamsPtr = %p\n", (void*)devSpatialParamsPtr);
    printf("Model204.devParams.initialStep = %g\n", devParams.initialStep);
}

#endif  // USE_MODEL_204
