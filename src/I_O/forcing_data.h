// I_O/forcing_data.h
#pragma once

// how many forcings we ever support:
constexpr int MAX_FORCINGS = 16;

// constant‐memory metadata for all forcings
extern __constant__ double c_forc_dt[];
extern __constant__ size_t c_forc_nT[];

// device‐global forcing pointer + count
extern __device__ float* d_forc_data;
extern __device__ int   nForc;