//I_O/forcing_data.cu
#include "forcing_data.h"

// definition of the constant arrays
__constant__ double c_forc_dt[MAX_FORCINGS];
__constant__ size_t  c_forc_nT [MAX_FORCINGS];

// forcing_data.cu
__device__ float* d_forc_data = nullptr;
// __device__ int    nForc;
__constant__ int    nForc;









