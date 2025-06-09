// model_registry.hpp
// This file provides the host-side hookup for copying model parameters into
// the device-side symbol `devParams`.  In particular, it declares:
//   1) A non-templated helper `setModelParameters(const DummyModel::Parameters&)`
//      that calls cudaMemcpyToSymbol to copy the host struct into the device symbol.
//   2) A non-templated helper `setModelParameters(const Model204::Parameters&)`
//      that does the same for Model204.
//   3) A template declaration `rk45_api::setModelParameters<Model>`
//      which is only defined via explicit specializations.
//   4) Explicit specialization prototypes for DummyModel and Model204.
//
// The header `model_registry.hpp` pulls in each model’s Parameters struct
// and its `extern __constant__ devParams` symbol via the model header.

#pragma once
#include "models/active_model.hpp"  // brings in ActiveModel and its devParams
#include <cuda_runtime.h>

namespace rk45_api {

/// 1) Non-templated helper that copies a DummyModel::Parameters struct from host
///    memory into the device-side constant symbol `devParams` (defined in model_dummy_global.cu).
///    Throws std::runtime_error if cudaMemcpyToSymbol fails.
#if defined(USE_DUMMY_MODEL)
void setModelParameters(const DummyModel::Parameters& p);
#endif

/// 2) Non-templated helper that copies a Model204::Parameters struct from host
///    memory into the device-side constant symbol `devParams` (defined in model_204_global.cu).
///    Throws std::runtime_error if cudaMemcpyToSymbol fails.
#if defined(USE_MODEL_204)
void setModelParameters(const Model204::Parameters& p);
#endif

/// 3) Primary template declaration – only defined via explicit specializations.
template <typename Model>
void setModelParameters(const typename Model::Parameters& p);

/// 4) Explicit specialization prototypes:
#if defined(USE_DUMMY_MODEL)
template <> void setModelParameters<DummyModel>(const DummyModel::Parameters& p);
#elif defined(USE_MODEL_204)
template <> void setModelParameters<Model204>(const Model204::Parameters& p);
#endif

} // namespace rk45_api
