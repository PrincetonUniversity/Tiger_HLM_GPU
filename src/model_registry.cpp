// src/model_registry.cpp

// ─────────────────────────────────────────────────────────────────────────────
// model_registry.cpp
//
// This file provides the runtime hookup for copying host‐side model parameters
// into the device‐side symbol `devParams`.  It defines:
//   1) A non‐templated helper `setModelParameters(const DummyModel::Parameters&)`
//      which calls cudaMemcpyToSymbol to copy into the single device symbol.
//   2) A template specialization for `rk45_api::setModelParameters<DummyModel>`
//      that simply calls the helper above.
//   3) An explicit instantiation of that specialization so the linker emits it.
//
// The header `model_registry.hpp` declares these functions, and also pulls in
// `DummyModel` (and its `Parameters` struct and `devParams` symbol) via
// `models/model_dummy.hpp`.
//
// Because `devParams` lives in constant/global device memory (defined once in
// model_dummy_global.cpp), this ensures that the host can update it before
// launching any RK45 solver kernels.
// ─────────────────────────────────────────────────────────────────────────────

#include "model_registry.hpp"    // Declares setModelParameters(...) and the template<> prototype
#include <cuda_runtime.h>        // For cudaMemcpyToSymbol and cudaGetErrorString
#include <stdexcept>             // For std::runtime_error
#include <string>                // For std::string
#include <iostream>              // For std::cout and std::cerr

// ─────────────────────────────────────────────────────────────────────────────
// Non‐templated helper that actually performs cudaMemcpyToSymbol.
// Copies the contents of 'p' into the single device symbol 'devParams',
// which was defined in model_dummy_global.cpp as:
//   __constant__ DummyModel::Parameters devParams = { ... };
// ─────────────────────────────────────────────────────────────────────────────
// void setModelParameters(const DummyModel::Parameters& p) {
//     std::cout << "Size of DummyModel::Parameters: " << sizeof(DummyModel::Parameters) << " bytes" << std::endl;
//     // Attempt to copy host 'p' → device symbol 'devParams'
//     cudaError_t err = cudaMemcpyToSymbol(devParams, &p, sizeof(p));
//     if (err != cudaSuccess) {
//         // If copy fails, throw a runtime error with the CUDA error string
//         throw std::runtime_error(
//             std::string("cudaMemcpyToSymbol failed: ") +
//             cudaGetErrorString(err)
//         );
//     }
// }
// 
void setModelParameters(const DummyModel::Parameters& p) {
    std::cout << "Size of DummyModel::Parameters: " << sizeof(DummyModel::Parameters) << " bytes" << std::endl;

    // Ensure the device is ready
    cudaDeviceSynchronize();

    // Attempt to copy host 'p' → device symbol 'devParams'
    cudaError_t err = cudaMemcpyToSymbol(devParams, &p, sizeof(p));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed with error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpyToSymbol failed in setModelParameters");
    }
    std::cout << "cudaMemcpyToSymbol succeeded for devParams." << std::endl;

    // Launch kernel to check devParams on the device
    checkDevParamsKernel<<<1, 1>>>();
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel launch failed with error: " << cudaGetErrorString(kernelErr) << std::endl;
        throw std::runtime_error("Kernel launch failed in setModelParameters");
    }
    cudaDeviceSynchronize();
}
// ─────────────────────────────────────────────────────────────────────────────
// Template specialization for rk45_api::setModelParameters<DummyModel>.
// 
// The primary template is declared in solver/rk45_api.hpp as:
//   namespace rk45_api {
//       template <class Model>
//       void setModelParameters(const typename Model::Parameters& hostParams) { ... }
//   }
// 
// Here, we provide the body of the specialization for DummyModel, which
// simply calls our non‐templated helper above.
// ─────────────────────────────────────────────────────────────────────────────
namespace rk45_api {
    template<>
    void setModelParameters<DummyModel>(const DummyModel::Parameters& p) {
        // Forward to our helper that does the actual cudaMemcpyToSymbol
        ::setModelParameters(p);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiation of the DummyModel specialization.
//
// This forces the compiler to emit code for
//   rk45_api::setModelParameters<DummyModel>(const DummyModel::Parameters&)
//
// Without this line, the specialization might never be emitted into the object
// file, leading to undefined‐reference errors if `main.cpp` calls it.
// ─────────────────────────────────────────────────────────────────────────────
template void rk45_api::setModelParameters<DummyModel>(const DummyModel::Parameters&);

