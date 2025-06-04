#include "models/model_dummy.hpp"
#include <cuda_runtime.h>
#include <stdio.h> // Required for printf

// ─────────────────────────────────────────────────────────────────────────────
// Define + initialize `devParams` in constant memory once.
// ─────────────────────────────────────────────────────────────────────────────
__constant__ DummyModel::Parameters devParams = {
    /* initialStep = */ 0.01,
    /* safety      = */ 0.9,
    /* minScale    = */ 0.2,
    /* maxScale    = */ 5.0
};

// Kernel to check devParams on the device
__global__ void checkDevParamsKernel() {
    printf("Device-side devParams.initialStep: %f\n", devParams.initialStep);
}
