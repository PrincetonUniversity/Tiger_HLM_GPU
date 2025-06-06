
// This file provides the host‐side hookup for copying model parameters into
// the device‐side symbol `devParams`.  In particular, it defines:
//   1) A non‐templated helper `setModelParameters(const DummyModel::Parameters&)`
//      that calls cudaMemcpyToSymbol to copy the host struct into the device symbol.
//   2) A template specialization `rk45_api::setModelParameters<DummyModel>`
//      which forwards to the helper above.
//   3) An explicit instantiation of that specialization so the linker emits it.
//
// The header `model_registry.hpp` declares these functions and pulls in
// `DummyModel` (and its `Parameters` struct and `devParams` symbol) via
// `models/model_dummy.hpp`.
//
// Because `devParams` resides in constant device memory (defined in
// model_dummy_global.cpp), this mechanism ensures that the host can update
// device parameters before launching any RK45 kernels.

#include <cuda_runtime.h>
#include <stdexcept>            
#include <string>                
#include <iostream>
#include "model_registry.hpp"    // Declares setModelParameters(...) and its template prototype             

// ─────────────────────────────────────────────────────────────────────────────
// Non‐templated helper that copies a DummyModel::Parameters struct from host
// memory into the device‐side constant symbol `devParams` (defined in
// model_dummy_global.cpp).
//
// On entry, the device must be ready.  If the cudaMemcpyToSymbol call fails,
// this function throws a runtime_error containing the CUDA error message.
// ─────────────────────────────────────────────────────────────────────────────
void setModelParameters(const DummyModel::Parameters& p) {
    std::cout << "Size of DummyModel::Parameters: "
              << sizeof(DummyModel::Parameters) << " bytes" << std::endl;

    // Ensure any previous CUDA calls have completed
    cudaDeviceSynchronize();

    // Copy host struct 'p' into device symbol 'devParams'
    cudaError_t err = cudaMemcpyToSymbol(devParams, &p, sizeof(p));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol failed: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpyToSymbol failed in setModelParameters");
    }
    std::cout << "cudaMemcpyToSymbol succeeded for devParams." << std::endl;

    // Optional: launch a small kernel to verify that devParams was set correctly
    // checkDevParamsKernel<<<1, 1>>>();
    // cudaError_t kernelErr = cudaGetLastError();
    // if (kernelErr != cudaSuccess) {
    //     std::cerr << "Kernel launch failed: "
    //               << cudaGetErrorString(kernelErr) << std::endl;
    //     throw std::runtime_error("Kernel launch failed in setModelParameters");
    // }
    cudaDeviceSynchronize();
}

// ─────────────────────────────────────────────────────────────────────────────
// Template specialization for rk45_api::setModelParameters<DummyModel>.
//
// The primary template is declared in solver/rk45_api.hpp as:
//   namespace rk45_api {
//       template <class Model>
//       void setModelParameters(const typename Model::Parameters& hostParams);
//   }
//
// Here, we implement the specialization for DummyModel by forwarding to our
// non‐templated helper above.
// ─────────────────────────────────────────────────────────────────────────────
namespace rk45_api {
    template<>
    void setModelParameters<DummyModel>(const DummyModel::Parameters& p) {
        ::setModelParameters(p);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiation of the DummyModel specialization.
//
// This forces the compiler to emit code for
//   rk45_api::setModelParameters<DummyModel>(const DummyModel::Parameters&)
//
// Without this, the specialization might not be generated, leading to
// undefined‐reference errors at link time when main.cpp calls it.
// ─────────────────────────────────────────────────────────────────────────────
template void rk45_api::setModelParameters<DummyModel>(const DummyModel::Parameters&);
