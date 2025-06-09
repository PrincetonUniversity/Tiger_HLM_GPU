// src/solver/rk45_api.hpp
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <tuple>                    // for std::make_tuple, std::tuple
#include "rk45.h"                   // For template< class Model > rk45_kernel_multi<…>
#include "models/model_dummy.hpp"   // For DummyModel::Parameters, etc.
#include "models/active_model.hpp"  // Always brings in Model204 as ActiveModel
#include "parameters_loader.hpp"    // for SpatialParams

// #if defined(USE_DUMMY_MODEL)
// #  include "models/model_dummy.hpp"  // DummyModel::N_EQ and extern __constant__ DummyModel::Parameters devParams
// #elif defined(USE_MODEL_204)
// #  include "models/model_204.hpp"    // Model204::N_EQ and extern __constant__ Model204::Parameters devParams
// #else
// #  error "You must compile with either -DUSE_DUMMY_MODEL or -DUSE_MODEL_204"
// #endif

namespace rk45_api {

using DenseType = std::vector<double>;
using FinalType = std::vector<double>;

// ───────── 1) Allocate GPU buffers & copy inputs ─────────
//   • h_y0 size   = num_systems * Model::N_EQ
//   • h_query_times size = num_queries
// Returns a tuple of raw device pointers plus sizes for later teardown.
template<class Model>
auto setup_gpu_buffers(
    const std::vector<double>& h_y0,
    const std::vector<double>& h_query_times
) {
    constexpr int N_EQ = Model::N_EQ;
    int num_systems = int(h_y0.size() / N_EQ);
    int num_queries = int(h_query_times.size());

    // Compute byte sizes
    size_t bytes_y0      = sizeof(double) * num_systems * N_EQ;
    size_t bytes_final   = sizeof(double) * num_systems * N_EQ;
    size_t bytes_queries = sizeof(double) * num_queries;
    size_t bytes_dense   = sizeof(double) * num_systems * N_EQ * num_queries;

    // Device pointers
    double *d_y0_all = nullptr, *d_y_final_all = nullptr;
    double *d_query_times = nullptr, *d_dense_all = nullptr;
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

    // Copy host → device
    err = cudaMemcpy(d_y0_all, h_y0.data(), bytes_y0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy to d_y0_all failed");
    err = cudaMemcpy(d_query_times, h_query_times.data(), bytes_queries, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy to d_query_times failed");

    return std::make_tuple(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        num_systems, num_queries
    );
}

// ───────── 2) Launch the GPU RK45 solver ─────────
//   Invokes the 2D‐grid kernel and synchronizes+error‐checks.
template<class Model>
void launch_rk45_kernel(
    double* d_y0_all,
    double* d_y_final_all,
    double* d_query_times,
    double* d_dense_all,
    int num_systems,
    int num_queries,
    double t0,
    double tf,
    const SpatialParams* d_sp   // pointer to device SpatialParams array
) {
    // constexpr int THREADS_X = 16;
    // constexpr int THREADS_Y = 16;

    // int numBlocksX = (num_systems + THREADS_X - 1) / THREADS_X;
    // int numBlocksY = (num_queries + THREADS_Y - 1) / THREADS_Y;
    // dim3 blocks(numBlocksX, numBlocksY);
    // dim3 threadsPerBlock(THREADS_X, THREADS_Y);
    
    // Only parallelize over systems (one thread per system)
    constexpr int THREADS_PER_BLOCK = 128;
    int numBlocks = (num_systems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 blocks(numBlocks);
    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    // Launch
    // (rk45_kernel_multi<Model>)
    //     <<< blocks, threadsPerBlock >>> (
    //     d_y0_all, d_y_final_all, d_query_times, d_dense_all,
    //     num_systems, num_queries, t0, tf
    // );

    // Launch the new combined RK45+Radau kernel:
    (rk45_then_radau_multi<Model>)
        <<< blocks, threadsPerBlock >>>
        ( d_y0_all, d_y_final_all,
           d_query_times, d_dense_all,
           num_systems, num_queries,
           t0, tf,
           d_sp                // ← forwarded here
        );

    // Sync + check
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) throw std::runtime_error("Kernel execution failed");
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Kernel launch failed");
}

// ───────── 3) Copy back results & free GPU memory ─────────
//   Returns the two host‐side vectors and frees all device pointers.
//
//   **Important**: we take the raw dense‐output layout  
//     [ sys_id ][ comp ][ query ]  
//   and reorder it to  
//     [ sys_id ][ query ][ comp ]  
//   so that the existing CSV‐writer indexing  
//     (s*num_queries + q)*N_EQ + comp  
//   works correctly and produces one continuous curve.
template<class Model>
std::pair<FinalType, DenseType> retrieve_and_free(
    double* d_y0_all,
    double* d_y_final_all,
    double* d_query_times,
    double* d_dense_all,
    int num_systems,
    int num_queries
) {
    constexpr int N_EQ = Model::N_EQ;
    size_t bytes_final = sizeof(double) * num_systems * N_EQ;
    size_t bytes_dense = sizeof(double) * num_systems * N_EQ * num_queries;

    // Allocate host buffers
    FinalType h_y_final_all(num_systems * N_EQ);
    DenseType h_dense_raw(num_systems * N_EQ * num_queries);

    // Copy device → host
    cudaMemcpy(h_y_final_all.data(), d_y_final_all, bytes_final,   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dense_raw.data(),    d_dense_all,    bytes_dense, cudaMemcpyDeviceToHost);

    // Free device
    cudaFree(d_y0_all);
    cudaFree(d_y_final_all);
    cudaFree(d_query_times);
    cudaFree(d_dense_all);

    // Reorder raw dense output from [sys][comp][q] to [sys][q][comp]
    DenseType h_dense_all(num_systems * N_EQ * num_queries);
    for (int s = 0; s < num_systems; ++s) {
        for (int q = 0; q < num_queries; ++q) {
            for (int c = 0; c < N_EQ; ++c) {
                // raw index as written by the kernel:
                int src = s * (N_EQ * num_queries)
                        + c * (num_queries)
                        + q;
                // desired index for CSV‐writer: (s*num_queries + q)*N_EQ + c
                int dst = (s * num_queries + q) * N_EQ + c;
                h_dense_all[dst] = h_dense_raw[src];
            }
        }
    }

    return { std::move(h_y_final_all), std::move(h_dense_all) };
}

// ───────── run_rk45<Model> composes the three steps ─────────
template <class Model>
std::pair<FinalType, DenseType>
run_rk45(
    const std::vector<double>& h_y0,
    double t0,
    double tf,
    const std::vector<double>& h_query_times,
    const SpatialParams* d_sp   // pointer threaded through here too
) {
    // 1) alloc & copy
    auto [d_y0_all, d_y_final_all, d_query_times, d_dense_all, ns, nq]
      = setup_gpu_buffers<Model>(h_y0, h_query_times);

    // 2) launch solver
    launch_rk45_kernel<Model>(
      d_y0_all, d_y_final_all, d_query_times, d_dense_all,
      ns, nq, t0, tf,
      d_sp                       // added d_sp here
    );

    // 3) retrieve & free
    return retrieve_and_free<Model>(
      d_y0_all, d_y_final_all, d_query_times, d_dense_all, ns, nq
    );
}

} // namespace rk45_api
