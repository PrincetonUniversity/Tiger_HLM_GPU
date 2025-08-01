// debug_kernls.hpp
// Future Improvement: add switch to activate debug kernels when compiling with -DDEBUG_KERNELS
#pragma once

#include "models/model_204.hpp"  // for Runoff5::N_EQ, rhs
#include "I_O/forcing_data.h"    // for SpatialParams

// ────────── Forcing pointer and symbol checking ──────────

// __global__ void checkForcingPtr();
__global__ void checkForcingsOnDevice(const float* forc, int ns, int nf);

// __global__ void checkForcingSymbols();
// __global__ void checkForcingSymbols(const float* ptr, int nf);

// ────────── Forcing array debugging ──────────

__global__ void debugMinuteForcings(const float* forc, int ns);

// __global__ void debugForcings(const float* forc, size_t nF, int ns);
// __global__ void debugForcings2(const float* forc, int ns);
// __global__ void debugForcingsMulti5Steps(const float* forc, int ns);
// __global__ void debugForcingsMultiWithDayBoundary(const float* forc, int ns);


// __global__ void debugHolding(const float* forc, int ns);

// ────────── Forcing + RHS interaction ──────────

__global__ void debugRhsForcings(const SpatialParams* sp_ptr,
                                 const float*         F,
                                 int                  ns,
                                 int                  nForc);

// ────────── SpatialParams debugging ──────────

// __global__ void testKernel();  // minimal test

// __global__ void debugParams(const SpatialParams* sp);
__global__ void debugAllParams(const SpatialParams* sp, int N);

// __global__ void debugRHS(const SpatialParams* sp, int sys_id);

// ────────── y₀ debugging ──────────

__global__ void debugDeviceY0(const double* d_y0_all, int ns);
