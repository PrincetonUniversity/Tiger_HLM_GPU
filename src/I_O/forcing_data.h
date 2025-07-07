// I_O/forcing_data.h
#pragma once

constexpr int MAX_FORCINGS = 16;

// Use __constant__ here
extern __constant__ double c_forc_dt[MAX_FORCINGS];
extern __constant__ size_t c_forc_nT[MAX_FORCINGS];

// Device symbols declared as extern
extern __device__ float* d_forc_data;
extern __device__ int    nForc;





