// models/model_dummy_global.cu

#include "models/model_dummy.hpp"
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef USE_MODEL_DUMMY

// ─────────────────────────────────────────────────────────────────────────────
// Define + initialize devParams in constant memory once.
// This provides the device‐side storage that model_registry.cpp will populate.
// ─────────────────────────────────────────────────────────────────────────────
__constant__ DummyModel::Parameters devParams;

// ─────────────────────────────────────────────────────────────────────────────
// (Optional) Kernel to check devParams on the device.
// ─────────────────────────────────────────────────────────────────────────────
// __global__ void checkDevParamsKernelDummy() {
//     printf("DummyModel.devParams.initialStep = %g\n", devParams.initialStep);
//     printf("DummyModel.devParams.rtol        = %g\n", devParams.rtol);
//     printf("DummyModel.devParams.atol        = %g\n", devParams.atol);
//     printf("DummyModel.devParams.safety      = %g\n", devParams.safety);
//     printf("DummyModel.devParams.minScale    = %g\n", devParams.minScale);
//     printf("DummyModel.devParams.maxScale    = %g\n", devParams.maxScale);
// }

#endif  // USE_MODEL_DUMMY
