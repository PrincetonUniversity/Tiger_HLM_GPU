#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include "model_Runoff5.hpp"  // brings in Runoff5 & extern devParams

namespace rk45_api {

/// Non‐templated helper: copies a Runoff5::Parameters into device constant memory.
void setModelParameters(const Runoff5::Parameters &p);

/// Primary template declaration (no definition here; specializations provided in .cpp).
template <typename Model>
void setModelParameters(const typename Model::Parameters &p);

} // namespace rk45_api
