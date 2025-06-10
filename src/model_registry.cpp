// This file provides the host‐side hookup for copying model parameters into
// the device‐side symbol `devParams`.  It defines the non‐templated helper
// and the explicit template specialization for Model204.

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "models/model_204.hpp"    // Model204 & extern __constant__ devParams
#include "model_registry.hpp"

using namespace rk45_api;

// ─────────────────────────────────────────────────────────────────────────────
// Helper that copies a Model204::Parameters struct from host memory
// into the device‐side constant symbol `devParams` (defined in model_204_global.cu).
// Throws std::runtime_error if cudaMemcpyToSymbol fails.
// ─────────────────────────────────────────────────────────────────────────────
void setModelParameters(const Model204::Parameters& p) {
    std::cout << "Size of Model204::Parameters: "
              << sizeof(Model204::Parameters) << " bytes\n";

    // Ensure any previous CUDA calls have completed
    cudaDeviceSynchronize();

    // Copy host struct 'p' into device symbol 'devParams'
    cudaError_t err = cudaMemcpyToSymbol(
        &devParams,                              // device constant symbol
        &p,                                     // address of host struct
        sizeof(Model204::Parameters),           // byte count
        0,                                      // offset
        cudaMemcpyHostToDevice                  // kind
    );
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyToSymbol(devParams) failed: ")
            + cudaGetErrorString(err)
        );
    }
    std::cout << "cudaMemcpyToSymbol succeeded for Model204 devParams.\n";

    // Wait for the copy to finish before returning
    cudaDeviceSynchronize();
}

namespace rk45_api {

// ─────────────────────────────────────────────────────────────────────────────
// Template specialization forwarding to the above helper.
// ─────────────────────────────────────────────────────────────────────────────
template <>
void setModelParameters<Model204>(const Model204::Parameters& p) {
    ::setModelParameters(p);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiation so the specialization is emitted.
// ─────────────────────────────────────────────────────────────────────────────
template void setModelParameters<Model204>(
    const Model204::Parameters& p
);

} // namespace rk45_api
