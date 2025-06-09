// models/active_model.hpp
// #pragma once

// #if defined(USE_MODEL_DUMMY)
//   #include "models/model_dummy.hpp"
//   using ActiveModel = DummyModel;
// #elif defined(USE_MODEL_204)
//   #include "models/model_204.hpp"
//   using ActiveModel = Model204;
// #else
//   #error "You must define USE_MODEL_DUMMY or USE_MODEL_204"
// #endif

// // ─────────────────────────────────────────────────────────────────────────────
// // Declare the one `devParams` in constant memory, matching whichever
// // ActiveModel you picked above.  Defined in the corresponding *_global.cu.
// // ─────────────────────────────────────────────────────────────────────────────
// extern __constant__ ActiveModel::Parameters devParams;
#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Select the “active” model at compile time
// ─────────────────────────────────────────────────────────────────────────────
//
// Pass either -DUSE_MODEL_DUMMY or -DUSE_MODEL_204 to your compiler (via
// the Makefile) to pick between the two implementations.  This alias
// makes all downstream code use the chosen model.
//
//   • DummyModel  : trivial 3-equation test case (in model_dummy.hpp)  
//   • Model204    : 5-equation snow/runoff model (in model_204.hpp)
#if defined(USE_MODEL_DUMMY)
  #include "models/model_dummy.hpp"
  using ActiveModel = DummyModel;
#elif defined(USE_MODEL_204)
  #include "models/model_204.hpp"
  using ActiveModel = Model204;
#else
  #error "You must compile with either -DUSE_MODEL_DUMMY or -DUSE_MODEL_204"
#endif


// ─────────────────────────────────────────────────────────────────────────────
// Device‐side constant memory for the chosen model’s parameters
// ─────────────────────────────────────────────────────────────────────────────
//
// Each model struct defines its own nested `struct Parameters`.  At startup
// you call `setModelParameters<ActiveModel>(params)` to copy your host‐side
// parameters into this GPU‐resident `__constant__` variable.  All kernels then
// read from `devParams` for tolerances, initial step size, etc.
//
// The actual storage is allocated in the corresponding *_global.cu file:
//
//   models/model_dummy_global.cu   // defines __constant__ DummyModel::Parameters devParams
//   models/model_204_global.cu     // defines __constant__ Model204::Parameters devParams
//
extern __constant__ ActiveModel::Parameters devParams;


// ─────────────────────────────────────────────────────────────────────────────
// Optional device kernel to inspect devParams at runtime
// ─────────────────────────────────────────────────────────────────────────────
//
// In your *_global.cu you also supply:
//
//     __global__ void checkDevParamsKernelXXX();
// 
// which prints out all the fields in `devParams`.  You can invoke it like:
//
//     checkDevParamsKernelXXX<<<1,1>>>();  // on whichever model you chose
//     cudaDeviceSynchronize();
//
// Here we forward‐declare the one that matches your ActiveModel:
#if defined(USE_MODEL_DUMMY)
extern __global__ void checkDevParamsKernelDummy();
#elif defined(USE_MODEL_204)
extern __global__ void checkDevParamsKernel204();
#endif

