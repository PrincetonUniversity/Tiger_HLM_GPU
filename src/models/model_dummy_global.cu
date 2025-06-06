#include "models/model_dummy.hpp"
#include <cuda_runtime.h>
#include <stdio.h> 

// Define + initialize devParams in constant memory once.
__constant__ DummyModel::Parameters devParams;

// Kernel to check devParams on the device
// __global__ void checkDevParamsKernel() {
//     printf("Device-side devParams.initialStep: %f\n", devParams.initialStep);
//     printf("Device-side devParams.rtol:        %g\n", devParams.rtol);
//     printf("Device-side devParams.atol:        %g\n", devParams.atol);
//     printf("Device-side devParams.safety:      %g\n", devParams.safety);
//     printf("Device-side devParams.minScale:    %g\n", devParams.minScale);
//     printf("Device-side devParams.maxScale:    %g\n", devParams.maxScale);
// }
