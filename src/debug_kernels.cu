// debug_kernels.cu
#include <cstdio>               // for printf
#include <cuda_runtime.h>       // for CUDA runtime types (e.g., dim3, cudaMemcpy, etc.)

// Project-specific headers that define:
#include "models/model_204.hpp"   // for Runoff5::rhs and Runoff5::N_EQ
#include "I_O/forcing_data.h"     // for c_forc_dt, c_forc_nT, d_forc_data, nForc (device-side symbols)


// ────────── Device‐side constant pointers, checking kernel──────────
// __global__ void checkForcingPtr() {
//   // printf("d_forc_data = %p, nForc = %d\n", (void*)d_forc_data, nForc);
//   // Use d_forc_ptr and int nForc already on host side
//   printf("d_forc_ptr = %p, nForc = %d\n", (void*)d_forc_ptr, nForc);


// }

// __global__ void checkForcingSymbols() {
//     printf("[checkForcingSymbols] d_forc_data = %p, nForc = %d\n", (void*)d_forc_data, nForc);

//     // Prevent pruning (even if just this line)
//     if (nForc < 0) printf("nForc is negative?\n");
// }

// __global__ void checkForcingSymbols(const float* ptr, int nf) {
//   printf(
//     "d_forc_data = %p, nForc(arg) = %d\n"
//     "   dt[0]=%g, dt[1]=%g, nT[0]=%zu, nT[1]=%zu\n",
//     (void*)ptr, nf,
//     c_forc_dt[0], c_forc_dt[1],
//     c_forc_nT[0], c_forc_nT[1]
//   );
// }


// ────────── Debugging Forcings ──────────
// __global__ void debugForcings(const float *forc, size_t nF, int ns) {
//   // print the first time‐slice of each of the first 4 systems
//   if (threadIdx.x < 4 && blockIdx.x == 0) {
//     int t = 0;               // time‐step
//     int s = threadIdx.x;     // stream index
//     size_t idx = t*ns + s;
//     printf(" for sys %d, t=%d → forc = %f\n", s, t, forc[idx]);
//   }
// }


// __global__ void debugForcings2(const float *forc, int ns) {
//     int s = threadIdx.x;
//     if (blockIdx.x==0 && s < 4) {       // print for first 4 streams
//         int t = 0;                      // time‐step 0
//         // forcing #0 lives at offset 0
//         size_t idx0 = /* offset of f=0 */ 0 + size_t(t)*ns + s;
//         printf("forcing[0], sys=%d, t=0 → %f\n", s, forc[idx0]);

//         // forcing #1 starts after all timesteps of forcing #0:
//         size_t offset1 = c_forc_nT[0] * size_t(ns);
//         size_t idx1    = offset1 + size_t(t)*ns + s;
//         printf("forcing[1], sys=%d, t=0 → %f\n", s, forc[idx1]);
//     }
// }

// __global__ void debugForcingsMulti5Steps(const float *forc, int ns) {
//     int s = threadIdx.x;           // stream index
//     if (blockIdx.x==0 && s < 4) {  // only do streams 0..3
//         size_t offset1 = c_forc_nT[0] * size_t(ns);
//         for (int t = 0; t < 5; ++t) {
//             // forcing 0 at time t
//             size_t idx0 = size_t(t)*ns + s;
//             printf("f0, sys=%d, t=%d → %f\n", s, t, forc[idx0]);
//             // forcing 1 at time t
//             size_t idx1 = offset1 + size_t(t)*ns + s;
//             printf("f1, sys=%d, t=%d → %f\n", s, t, forc[idx1]);
//         }
//     }
// }

// __global__ void debugForcingsMultiWithDayBoundary(const float *forc, int ns) {
//     int s = threadIdx.x;
//     if (blockIdx.x==0 && s < 4) {
//         // block‐start of second forcing:
//         size_t offset1 = c_forc_nT[0] * size_t(ns);
//         size_t samples1 = c_forc_nT[1];    // e.g. 2 days
//         // print every minute for the *first* daily‐step interval:
//         for (int t = 0; t < min(samples1, 10UL); ++t) {
//             size_t idx0 = /* first forcing */   size_t(t)*ns + s;
//             size_t idx1 = offset1 + size_t(t)*ns + s;
//             printf("t=%3d → pr=%7.3f, t2m=%7.3f\n", t, forc[idx0], forc[idx1]);
//         }
//         // then sample at the day boundary:
//         int day1 = int(c_forc_nT[1] * c_forc_dt[1] * 60.0);  // dt=24h → 1440 min
//         size_t idx1b = offset1 + size_t(day1/1)*ns + s;     // sampleIdx=1
//         printf("at t=%d min → t2m=%7.3f\n", day1, forc[idx1b]);
//     }
// }

// Print pr and t2m for each minute up to first 2 hours and at daily boundary
// __global__ void debugMinuteForcings(const float *forc, int ns) {
//     int s = threadIdx.x;
//     if (blockIdx.x == 0 && s < 1) {             // do first 2 systems, s=0..1
//         size_t offset_pr  = 0;                   // pr is j=0 block
//         size_t offset_t2m = c_forc_nT[0] * ns;   // t2m is j=1 block

//         int max_min = 100;                       // first 100 minutes (2 h)
//         for (int t = 0; t <= max_min; ++t) {
//             // sample index in pr block:
//             size_t idx_pr  = offset_pr  + size_t(t)*ns + s;
//             // sample index in t2m block:
//             // compute minute‐index into daily samples:
//             double dt_t2m_min = c_forc_dt[1] * 60.0; // 24*60 = 1440
//             // sampleIdx = floor(t / dt_t2m_min) → 0 for t<1440, 1 for t≥1440
//             size_t sampleIdx_t2m = (t < (int)dt_t2m_min ? 0 : 1);
//             size_t idx_t2m = offset_t2m + sampleIdx_t2m*ns + s;

//             printf("sys=%d  t=%4d min → pr=%7.3f  t2m=%7.3f\n",
//                    s, t,
//                    forc[idx_pr],
//                    forc[idx_t2m]);
//         }

//         // also show at exactly 1440 min (1 day) and at 2880 min (2 days)
//         int day1 = int(c_forc_dt[1]*60.0);      // =1440
//         int day2 = day1 * 2;                   // =2880
//         for (int t : {day1, day2}) {
//             size_t idx_pr  = offset_pr  + size_t(t)*ns + s; 
//             size_t sampleIdx_t2m = (t < day1 ? 0 : (t < day2 ? 1 : 1));
//             size_t idx_t2m = offset_t2m + sampleIdx_t2m*ns + s;
//             printf("sys=%d  t=%4d min → pr=%7.3f  t2m=%7.3f\n",
//                    s, t,
//                    forc[idx_pr],
//                    forc[idx_t2m]);
//         }
//     }
// }

// ────────── In src/solver/rk45_kernel.cu, above testKernel ──────────

// __global__ void debugHolding(const float *forc, int ns) {
//     int sys = blockIdx.x * blockDim.x + threadIdx.x;
//     if (sys != 0) return;      // only print for system 0

//     // offsets into the big forcing array
//     size_t offset_pr  = 0;
//     size_t offset_t2m = size_t(c_forc_nT[0]) * ns;

//     // sampling intervals in minutes
//     // (c_forc_dt are in hours)
//     double dt_pr_min   = c_forc_dt[0] * 60.0;   // e.g. 1 h→60 min
//     double dt_t2m_min  = c_forc_dt[1] * 60.0;   // e.g. 24 h→1440 min

//     printf(" t   →  pr[idx_pr]   |  t2m[idx_t2m]\n");
//     for (int t = 0; t <= 180; ++t) {  // print first 3 hours = 180 min
//         // integer sample index for each forcing:
//         int idx_pr  = int(t / dt_pr_min);
//         int idx_t2m = int(t / dt_t2m_min);

//         // clamp to valid range
//         if (idx_pr  >= int(c_forc_nT[0])) idx_pr  = int(c_forc_nT[0]) - 1;
//         if (idx_t2m >= int(c_forc_nT[1])) idx_t2m = int(c_forc_nT[1]) - 1;

//         float pr_val  = forc[offset_pr  + size_t(idx_pr)  * ns + sys];
//         float t2m_val = forc[offset_t2m + size_t(idx_t2m) * ns + sys];

//         printf("t=%3d → pr[%2d]=%7.3f   |   t2m[%2d]=%7.3f\n",
//                t, idx_pr, pr_val,
//                idx_t2m, t2m_val);
//     }
// }

// ────────── End debugging forcings ──────────


// ───────── Minimal test kernel ─────────
// __global__ void testKernel() { /* nothing */ }

// ───────── Tiny kernel to verify one SpatialParams via printf ─────────
// __global__ void debugParams(const SpatialParams* sp) {
//     if (blockIdx.x == 0 && threadIdx.x == 0) {
//         printf("GPU sees: stream=%ld, Hu=%g, infil=%g, perco=%g\n",
//                sp[0].stream, sp[0].Hu, sp[0].infil, sp[0].perco);
//     }
// }

// ─────────────────────────────────────────────────────────────

// ───────── Debugging device-side full print of spatial params ─────────
// Print every stream's SpatialParams from the GPU.
// ─────────────────────────────────────────────────────────────────────────
// __global__ void debugAllParams(const SpatialParams* sp, int N) {
//     int sys = blockIdx.x * blockDim.x + threadIdx.x;
//     if (sys < N) {
//         printf("sys=%3d → stream=%10ld, Hu=%6.3f, infil=%6.3f, perco=%6.3f, L=%6.3f, A_h=%6.3f\n",
//                sys,
//                sp[sys].stream,
//                sp[sys].Hu,
//                sp[sys].infil,
//                sp[sys].perco,
//                sp[sys].L,
//                sp[sys].A_h
//         );
//     }
// }
// ───────── ended debugging ──────────────────────────────────────────────────────


// ────────────────────────────────────────────────
// Debug‐kernel: call your RHS once for one stream
// ────────────────────────────────────────────────
// __global__ void debugRHS(const SpatialParams* sp, int sys_id) {
//     // Only one thread actually prints
//     if (blockIdx.x == 0 && threadIdx.x == 0) {
//         // Header so you can spot this block in the log
//         printf("[DebugRHS] ==== BEGIN RHS debug for stream index %d ====\n",
//                sys_id);
//         // Print the stream ID and raw parameters
//         printf("[DebugRHS]   Stream ID = %ld\n", sp[sys_id].stream);
//         printf("[DebugRHS]   Parameters: Hu=%g, infil=%g, perco=%g, L=%g, A_h=%g\n",
//                sp[sys_id].Hu,
//                sp[sys_id].infil,
//                sp[sys_id].perco,
//                sp[sys_id].L,
//                sp[sys_id].A_h);

//         // Prepare a trivial y-vector and compute dydt
//         double y[Runoff5::N_EQ]   = {1.0, 1.0, 1.0, 1.0, 1.0};
//         double dydt[Runoff5::N_EQ];
//         Runoff5::rhs(0.0, y, dydt, Runoff5::N_EQ, sys_id, sp, d_forc_data, nForc);

//         // Print the resulting derivatives cleanly
//         printf("[DebugRHS]   dydt: ");
//         for (int i = 0; i < Runoff5::N_EQ; ++i) {
//             printf("%g ", dydt[i]);
//         }
//         printf("\n[DebugRHS] ====  END RHS debug  ====\n\n");
//     }
// }


__global__ void debugMinuteForcings(const float *forc, int ns) {
    int s = 0;  // only look at system 0
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        size_t offset_pr  = 0;
        size_t offset_t2m = c_forc_nT[0] * size_t(ns);
        int max_min = 180;  // first 3 hours

        printf(" t   →  pr   |  t2m\n");
        for (int t = 0; t <= max_min; ++t) {
            int idx_pr  = min(int(c_forc_nT[0]) - 1, int(t / (c_forc_dt[0] * 60.0)));
            int idx_t2m = min(int(c_forc_nT[1]) - 1, int(t / (c_forc_dt[1] * 60.0)));
            float pr   = forc[offset_pr  + size_t(idx_pr)  * ns + s];
            float t2m  = forc[offset_t2m + size_t(idx_t2m) * ns + s];
            printf("t=%3d → pr[%2d]=%7.3f | t2m[%2d]=%7.3f\n",
                   t, idx_pr, pr, idx_t2m, t2m);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Kernel to debug the way rhs reads F[0] (rain) and F[1] (temp),
// now also printing sys index and stream ID
// ────────────────────────────────────────────────────────────────────────
__global__ void debugRhsForcings(const SpatialParams* sp_ptr,
                                 const float*         F,
                                 int                  ns,
                                 int                  nForc)
{
    if (blockIdx.x==0 && threadIdx.x==0) {
        // pick system 0 and time t=0
        int sys = 0;
        double t = 0.0;

        // grab that stream’s unique ID from your SpatialParams
        long stream_id = sp_ptr[sys].stream;

        printf(">>> debugRhsForcings <<<\n");
        printf("  sys index = %d, stream ID = %ld\n", sys, stream_id);
        printf("  nForc=%d, dt[0]=%g, dt[1]=%g, nT[0]=%llu, nT[1]=%llu\n",
               nForc,
               c_forc_dt[0], c_forc_dt[1],
               (unsigned long long)c_forc_nT[0],
               (unsigned long long)c_forc_nT[1]);

        // Compute the offsets into the big F array
        size_t pr_block   = 0;                    
        size_t t2m_block  = c_forc_nT[0] * size_t(ns);

        float rain0 = F[pr_block + sys];
        float temp0 = (nForc>1 ? F[t2m_block + sys] : 0.0f);

        printf("  GPU sees rainfall= %f, temperature= %f\n", rain0, temp0);

        // Now fire rhs and print dydt
        double y[Runoff5::N_EQ] = {0.01,0.1,0,0,0.01,1,1};   // use your real y0 here
        double dydt[Runoff5::N_EQ];
        Runoff5::rhs(t, y, dydt,
                      Runoff5::N_EQ,
                      sys,
                      sp_ptr,
                      F,
                      nForc);

        printf("  dydt =");
        for (int i = 0; i < Runoff5::N_EQ; ++i)
            printf(" %g", dydt[i]);
        printf("\n-----------------------------\n");
    }
}


// Debugging the initial y[0..6] for sys0
__global__ void debugDeviceY0(const double* d_y0_all, int ns) {
    // only need one thread
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf(">>> GPU initial y[0..6] for sys0 <<<\n");
        // d_y0_all is laid out [ y₀_sys0…y₆_sys0, y₀_sys1… , … ]
        for (int i = 0; i < Runoff5::N_EQ; ++i) {
            printf("  y[%d] = %g\n", i, d_y0_all[i]);
        }
        printf("-----------------------------\n");
    }
}



// above main():
// __global__ void checkForcingsOnDevice(const float* forc,
//                                       int ns,
//                                       int nf)    // <— pass nForc in!
// {
//     if (blockIdx.x==0 && threadIdx.x==0) {
//         printf(">>> DEVICE FORCINGS CHECK <<<\n");
//         printf("  nForc = %d\n", nf);
//         printf("  dt[0]=%g, dt[1]=%g\n", c_forc_dt[0], c_forc_dt[1]);
//         printf("  nT[0]=%llu, nT[1]=%llu\n",
//                (unsigned long long)c_forc_nT[0],
//                (unsigned long long)c_forc_nT[1]);

//         for (int f = 0; f < nf; ++f) {
//             // offset of forcing f at t=0
//             size_t offs = (f==0
//                            ? 0
//                            : c_forc_nT[0] * size_t(ns));
//             float v0 = forc[offs + /*sys=*/0];
//             printf("  forcing[%d], sys0, t=0 → %f\n", f, v0);
//         }
//         printf("-------------------------\n");
//     }
// }

__global__ void checkForcingsOnDevice(const float* forc,
                                      int ns,
                                      int nf)
{
    if (blockIdx.x==0 && threadIdx.x==0) {
        printf(">>> DEVICE FORCINGS MULTI‐STEP CHECK <<<\n");
        for (int f = 0; f < nf; ++f) {
            size_t base = (f==0? 0 : c_forc_nT[0] * size_t(ns));
            printf(" Forcing[%d] (dt=%g, nT=%llu):\n",
                   f, c_forc_dt[f],
                   (unsigned long long)c_forc_nT[f]);
            int maxPrint = min((int)c_forc_nT[f], 48);
            for (int t = 0; t < maxPrint; ++t) {
                size_t offs = base + size_t(t) * ns;
                printf("    t=%2d → %f\n", t, forc[offs + /*sys=*/0]);
            }
        }
        printf("----------------------------------------\n");
    }
}

// ───────── ended debugging ─────────────────────────────────────────────