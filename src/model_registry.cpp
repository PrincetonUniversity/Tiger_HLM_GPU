// model_registry.cpp
// This file provides the host-side hookup for copying model parameters into
// the device-side symbol `devParams`.  In particular, it defines:
//   1) A non-templated helper `setModelParameters(const ActiveModel::Parameters&)`
//      that calls cudaMemcpyToSymbol to copy the host struct into the device symbol.
//   2) A template specialization `rk45_api::setModelParameters<ActiveModel>`
//      which forwards to the helper above.
//   3) An explicit instantiation so the linker emits it.
//
// Because `devParams` resides in constant device memory (defined in the
// corresponding model_global.cu), this mechanism ensures that the host can
// update device parameters before launching any RK45 kernels.

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include "model_registry.hpp"    // Declares helper + template prototype
#include "models/active_model.hpp"// Brings in ActiveModel and extern __constant__ devParams

using namespace rk45_api;

// ─────────────────────────────────────────────────────────────────────────────
// Helper that copies an ActiveModel::Parameters struct from host memory
// into the device‐side constant symbol `devParams` (defined in model_global.cu).
// Throws std::runtime_error if cudaMemcpyToSymbol fails.
// ─────────────────────────────────────────────────────────────────────────────
void setModelParameters(const ActiveModel::Parameters& p) {
    std::cout << "Size of ActiveModel::Parameters: "
              << sizeof(ActiveModel::Parameters) << " bytes" << std::endl;

    // Ensure any previous CUDA calls have completed
    cudaDeviceSynchronize();

    // Copy host struct 'p' into device symbol 'devParams'
    cudaError_t err = cudaMemcpyToSymbol(
        devParams,                                // device constant symbol
        &p,                                       // address of host struct
        sizeof(ActiveModel::Parameters),          // byte count
        0,                                        // offset
        cudaMemcpyHostToDevice                    // kind
    );
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol(ActiveModel) failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(
            "cudaMemcpyToSymbol failed in setModelParameters(ActiveModel)"
        );
    }
    std::cout << "cudaMemcpyToSymbol succeeded for ActiveModel devParams." << std::endl;

    // Wait for the copy to finish before returning
    cudaDeviceSynchronize();
}

namespace rk45_api {
// ─────────────────────────────────────────────────────────────────────────────
// Template specialization forwarding to the above helper.
// ─────────────────────────────────────────────────────────────────────────────
template <>
void setModelParameters<ActiveModel>(const ActiveModel::Parameters& p) {
    ::setModelParameters(p);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiation of the specialization.
// ─────────────────────────────────────────────────────────────────────────────
template void setModelParameters<ActiveModel>(
    const ActiveModel::Parameters& p
);
} // namespace rk45_api
