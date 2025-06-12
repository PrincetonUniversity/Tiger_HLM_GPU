//I_O/forcing_data.cu
#include "forcing_data.h"

// one—and only one—definitions of the constant arrays:
__constant__ double c_forc_dt[MAX_FORCINGS];
__constant__ size_t  c_forc_nT [MAX_FORCINGS];

// old device globals:
__device__ float* d_forc_data = nullptr;
__device__ int    nForc      = 0;