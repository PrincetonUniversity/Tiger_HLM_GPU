// src/solver/rk45_api.hpp
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <tuple>                    // for std::make_tuple, std::tuple
#include <cstring>                  // for std::memset
#include "rk45.h"                   // For rk45_then_radau_multi<…> and radau_kernel_multi<…>
#include "parameters_loader.hpp"    // for SpatialParams
#include "models/model_204.hpp"     // For Model204::N_EQ, ::SP_TYPE, etc.
#include "I_O/forcing_data.h"

// ========== GLOBAL FORCING SYMBOLS ==========

// device pointer to flattened [forcing][time][stream] data
// extern float*  d_forc_data;
// number of distinct NetCDF variables you’re forcing with
// extern int     nForc;
// per‐forcing time step (dt) in constant memory:
//__constant__ double c_forc_dt[4];
// per‐forcing number of time steps in constant memory:
//__constant__ size_t c_forc_nT[4];

// device pointers for forcing data and its count
// __device__ float* d_forc_data_dev;
// __device__ int    nForc_dev;


// ────────────────────────────────────────────────
// ────────── Forward‐declare your Radau‐only kernel ──────────
// Must exactly match the definition in solver/radau_kernel.cu
template <class Model>
__global__
void radau_kernel_multi(
    double*                              y0_all, // 1
    double*                              y_final_all, // 2
    double*                              query_times, // 3
    double*                              dense_all, // 4
    int                                  num_systems,   // 5 
    int                                  num_queries,   // 6
    double                               t0,    // 7
    double                               tf,        // 8
    const typename Model::SP_TYPE* d_sp,       // 9
    int*                           stiff_system_indices,    // 10
    int                                  n_stiff,     // 11
    const float* d_forc_data,
    int          nForc
);



namespace rk45_api {

using DenseType = std::vector<double>;
using FinalType = std::vector<double>;

// ───────── 1) Allocate GPU buffers & copy inputs ─────────
//   • h_y0 size   = num_systems * Model204::N_EQ
//   • h_query_times size = num_queries
//   • Also: allocate an int flag per system for stiffness detection.
// Returns a tuple of raw device pointers plus sizes for later teardown.
template<class Model204>
auto setup_gpu_buffers(
    const std::vector<double>& h_y0,
    const std::vector<double>& h_query_times
) {
    constexpr int N_EQ = Model204::N_EQ;
    int num_systems = int(h_y0.size() / N_EQ);
    int num_queries = int(h_query_times.size());

    // Compute byte sizes
    size_t bytes_y0      = sizeof(double) * num_systems * N_EQ;
    size_t bytes_final   = sizeof(double) * num_systems * N_EQ;
    size_t bytes_queries = sizeof(double) * num_queries;
    size_t bytes_dense   = sizeof(double) * num_systems * N_EQ * num_queries;
    size_t bytes_stiff   = sizeof(int)    * num_systems;             // new

    // Device pointers
    double *d_y0_all = nullptr, *d_y_final_all = nullptr;
    double *d_query_times = nullptr, *d_dense_all = nullptr;
    int    *d_stiff = nullptr;                                      // new
    cudaError_t err;

    // Allocate
    err = cudaMalloc(&d_y0_all, bytes_y0);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_y0_all failed");
    err = cudaMalloc(&d_y_final_all, bytes_final);
    if (err != cudaSuccess) { cudaFree(d_y0_all); throw std::runtime_error("cudaMalloc d_y_final_all failed"); }
    err = cudaMalloc(&d_query_times, bytes_queries);
    if (err != cudaSuccess) { cudaFree(d_y0_all); cudaFree(d_y_final_all); throw std::runtime_error("cudaMalloc d_query_times failed"); }
    err = cudaMalloc(&d_dense_all, bytes_dense);
    if (err != cudaSuccess) { cudaFree(d_y0_all); cudaFree(d_y_final_all); cudaFree(d_query_times); throw std::runtime_error("cudaMalloc d_dense_all failed"); }

    // Allocate stiffness flags and zero‐initialize
    err = cudaMalloc(&d_stiff, bytes_stiff);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); cudaFree(d_y_final_all); cudaFree(d_query_times); cudaFree(d_dense_all);
        throw std::runtime_error("cudaMalloc d_stiff failed");
    }
    err = cudaMemset(d_stiff, 0, bytes_stiff);                      // new
    if (err != cudaSuccess) throw std::runtime_error("cudaMemset d_stiff failed");

    // Copy host → device
    err = cudaMemcpy(d_y0_all, h_y0.data(), bytes_y0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy to d_y0_all failed");
    err = cudaMemcpy(d_query_times, h_query_times.data(), bytes_queries, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy to d_query_times failed");

    return std::make_tuple(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        d_stiff,                  // new
        num_systems, num_queries
    );
}

// ───────── 2) Launch the GPU RK45 solver ─────────
//   Now passes the stiffness‐flag array into the kernel.
template<class Model204>
void launch_rk45_kernel(
    double* d_y0_all,
    double* d_y_final_all,
    double* d_query_times,
    double* d_dense_all,
    int*    d_stiff,             // new
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf,
    const typename Model204::SP_TYPE* d_sp
) {
    constexpr int THREADS_PER_BLOCK = 128;
    int numBlocks = (num_systems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 blocks(numBlocks);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    // Launch the combined RK45+stiffness‐flagging kernel:
    (rk45_then_radau_multi<Model204>)
        <<< blocks, threadsPerBlock >>>
        ( d_y0_all, d_y_final_all,
          d_query_times, d_dense_all,
          num_systems, num_queries,
          t0, tf,
          d_sp,
          d_stiff                  // ← pass in the flag array
        );

    // Sync + check
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("Kernel execution failed");
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed!!!!!!");
}

// ───────── 3) Copy back results & free GPU memory ─────────
//   We now also copy back the stiffness flags, gather which
//   systems are stiff, and invoke Radau only on those.
template<class Model204>
std::pair<FinalType, DenseType> retrieve_and_free(
    double* d_y0_all,
    double* d_y_final_all,
    double* d_query_times,
    double* d_dense_all,
    int*    d_stiff,            // new
    int     num_systems,
    int     num_queries,
    double  t0,
    double  tf,
    const typename Model204::SP_TYPE* d_sp
) {
    constexpr int N_EQ = Model204::N_EQ;
    size_t bytes_final = sizeof(double) * num_systems * N_EQ;
    size_t bytes_dense = sizeof(double) * num_systems * N_EQ * num_queries;
    size_t bytes_stiff = sizeof(int)    * num_systems;            // new

    // 3a) Copy device → host for solution & stiffness flags
    FinalType h_y_final_all(num_systems * N_EQ);
    DenseType h_dense_raw(num_systems * N_EQ * num_queries);
    std::vector<int> h_stiff(num_systems);                        // new

    cudaMemcpy(h_y_final_all.data(), d_y_final_all, bytes_final,   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dense_raw.data(),    d_dense_all,    bytes_dense, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stiff.data(),        d_stiff,        bytes_stiff, cudaMemcpyDeviceToHost);  // new

    // Free stiffness‐flag array now
    cudaFree(d_stiff);

    // 3b) Determine which systems are stiff
    std::vector<int> stiff_idxs;
    for (int i = 0; i < num_systems; ++i) {
        if (h_stiff[i] != 0) {
            stiff_idxs.push_back(i);
        }
    }


    // 3c) Launch Radau‐only kernel on flagged systems
    if (!stiff_idxs.empty()) {
        int n_stiff = int(stiff_idxs.size());
        int* d_stiff_idxs = nullptr;
        cudaMalloc(&d_stiff_idxs, sizeof(int)*n_stiff);
        cudaMemcpy(d_stiff_idxs, stiff_idxs.data(), sizeof(int)*n_stiff, cudaMemcpyHostToDevice);

        // // Debug print
        // printf("Stiff systems:");
        // for (int id : stiff_idxs) printf(" %d", id);
        // printf("\n");

        constexpr int TPB = 128;
        int blocks2 = (n_stiff + TPB - 1) / TPB;
        radau_kernel_multi<Model204>
          <<< blocks2, TPB >>>(
            d_y0_all,         // 1
            d_y_final_all,    // 2
            d_query_times,    // 3
            d_dense_all,      // 4
            num_systems,      // 5
            num_queries,      // 6
            t0,               // 7
            tf,               // 8
            d_sp,             // 9
            d_stiff_idxs,     // 10
            n_stiff,           // 11
            d_forc_data,
            nForc              // number of forcings
          );
        cudaDeviceSynchronize();
        // launch‐error check
        {
             cudaError_t err = cudaGetLastError();
             if (err != cudaSuccess) {
                 throw std::runtime_error(std::string("radau_kernel_multi launch failed: ")
                                          + cudaGetErrorString(err));
             }
         }
         // execution‐error check
         {
             cudaError_t err = cudaDeviceSynchronize();
             if (err != cudaSuccess) {
                 throw std::runtime_error(std::string("radau_kernel_multi execution failed: ")
                                          + cudaGetErrorString(err));
                                                      
                }
        }
        cudaFree(d_stiff_idxs);
    }

    // 3d) Free remaining device memory
    cudaFree(d_y0_all);
    cudaFree(d_y_final_all);
    cudaFree(d_query_times);
    cudaFree(d_dense_all);

    // 3e) Reorder raw dense output from [sys][comp][q] → [sys][q][comp]
    DenseType h_dense_all(num_systems * N_EQ * num_queries);
    for (int s = 0; s < num_systems; ++s) {
        for (int q = 0; q < num_queries; ++q) {
            for (int c = 0; c < N_EQ; ++c) {
                int src = s * (N_EQ * num_queries)
                        + c * (num_queries)
                        + q;
                int dst = (s * num_queries + q) * N_EQ + c;
                h_dense_all[dst] = h_dense_raw[src];
            }
        }
    }

    return { std::move(h_y_final_all), std::move(h_dense_all) };
}

// ───────── run_rk45<Model204> composes everything ─────────
template <class Model204>
std::pair<FinalType, DenseType>
run_rk45(
    const std::vector<double>& h_y0,
    double t0,
    double tf,
    const std::vector<double>& h_query_times,
    typename Model204::SP_TYPE* d_sp
) {
    // 1) alloc & copy
    auto [ d_y0_all,
           d_y_final_all,
           d_query_times,
           d_dense_all,
           d_stiff,             // new
           ns, nq ]
      = setup_gpu_buffers<Model204>(h_y0, h_query_times);

    // 2) launch RK45 + stiffness‐flagging
    launch_rk45_kernel<Model204>(
      d_y0_all, d_y_final_all,
      d_query_times, d_dense_all,
      d_stiff,              // new
      ns, nq, t0, tf,
      d_sp, // device‐constant pointer to spatial params
      d_forc_data, // forcing data pointer
      nForc         // number of forcings
    );

    // 3) retrieve, launch Radau on stiff systems, reorder, free
    return retrieve_and_free<Model204>(
      d_y0_all, d_y_final_all,
      d_query_times, d_dense_all,
      d_stiff,             // new
      //d_forc_data,   // ← host obtains this before
      //snForc,         // ← and this too
      ns, nq,
      t0, tf,
      d_sp, d_forc_data, nForc
    );
}

} // namespace rk45_api
