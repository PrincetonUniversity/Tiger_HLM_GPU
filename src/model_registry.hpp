#pragma once

#include "solver/rk45_api.hpp"     // Brings in: namespace rk45_api
#include "models/model_dummy.hpp" // Brings in DummyModel::Parameters and extern devParams

// ─────────────────────────────────────────────────────────────────────────────
// Define a fully templated version of setModelParameters.
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