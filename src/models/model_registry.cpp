// This file provides the host‐side hookup for copying model parameters into
// the device‐side symbol `devParams`.  It defines the non‐templated helper
// and the explicit template specialization for Runoff5.

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "model_Runoff5.hpp"    // Runoff5 & extern __constant__ devParams
#include "model_registry.hpp"

using namespace rk45_api;

// ─────────────────────────────────────────────────────────────────────────────
// Helper that copies a Runoff5::Parameters struct from host memory
// into the device‐side constant symbol `devParams` (defined in model_Runoff5_global.cu).
// Throws std::runtime_error if cudaMemcpyToSymbol fails.
// ─────────────────────────────────────────────────────────────────────────────
void setModelParameters(const Runoff5::Parameters& p) {
    std::cout << "Size of Runoff5::Parameters: "
              << sizeof(Runoff5::Parameters) << " bytes\n";

    // Ensure any previous CUDA calls have completed
    cudaDeviceSynchronize();

    // Copy host struct 'p' into device symbol 'devParams'
    cudaError_t err = cudaMemcpyToSymbol(
        &devParams,                              // device constant symbol
        &p,                                     // address of host struct
        sizeof(Runoff5::Parameters),           // byte count
        0,                                      // offset
        cudaMemcpyHostToDevice                  // kind
    );
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyToSymbol(devParams) failed: ")
            + cudaGetErrorString(err)
        );
    }
    std::cout << "cudaMemcpyToSymbol succeeded for Runoff5 devParams.\n";

    // Wait for the copy to finish before returning
    cudaDeviceSynchronize();
}

namespace rk45_api {

// ─────────────────────────────────────────────────────────────────────────────
// Template specialization forwarding to the above helper.
// ─────────────────────────────────────────────────────────────────────────────
template <>
void setModelParameters<Runoff5>(const Runoff5::Parameters& p) {
    ::setModelParameters(p);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiation so the specialization is emitted.
// ─────────────────────────────────────────────────────────────────────────────
template void setModelParameters<Runoff5>(
    const Runoff5::Parameters& p
);

} // namespace rk45_api
