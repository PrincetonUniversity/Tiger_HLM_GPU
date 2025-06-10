#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include "models/model_204.hpp"  // brings in Model204 & extern devParams

namespace rk45_api {

/// Non‐templated helper: copies a Model204::Parameters into device constant memory.
void setModelParameters(const Model204::Parameters &p);

/// Primary template declaration (no definition here; specializations provided in .cpp).
template <typename Model>
void setModelParameters(const typename Model::Parameters &p);

} // namespace rk45_api
