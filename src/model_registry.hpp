#pragma once

#include "solver/rk45_api.hpp"     // For namespace rk45_api
#include "models/model_dummy.hpp" // For DummyModel::Parameters and extern devParams

// ─────────────────────────────────────────────────────────────────────────────
// Define a fully templated version of setModelParameters.
// Copies host‐side Model::Parameters into the device constant symbol devParams.
// Throws std::runtime_error if cudaMemcpyToSymbol fails.
// ─────────────────────────────────────────────────────────────────────────────
namespace rk45_api {
    template <typename Model>
    void setModelParameters(const typename Model::Parameters& p) {
        cudaError_t err = cudaMemcpyToSymbol(devParams, &p, sizeof(typename Model::Parameters));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpyToSymbol failed: " + std::string(cudaGetErrorString(err)));
        }
    }
}