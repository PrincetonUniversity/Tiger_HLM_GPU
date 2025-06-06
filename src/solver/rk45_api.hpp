// src/solver/rk45_api.hpp
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include "rk45.h"                 // For template< class Model > rk45_kernel_multi<…>
#include "models/model_dummy.hpp" // For DummyModel::Parameters, etc.

namespace rk45_api {

using DenseType = std::vector<double>;
using FinalType = std::vector<double>;

/**
 * run_rk45<Model>:
 *   • h_y0 size   = num_systems * Model::N_EQ
 *   • h_query_times size = num_queries
 *
 * Returns:
 *   pair{ 
 *     FinalType: length = num_systems * Model::N_EQ,
 *     DenseType: length = num_systems * Model::N_EQ * num_queries 
 *   }
 *
 * Internally, it allocates GPU buffers, copies data, launches
 * rk45_kernel_multi<Model><<<…>>>, and copies results back.
 */
template <class Model>
std::pair<FinalType, DenseType>
run_rk45(
    const std::vector<double>& h_y0,          // host initial states
    double t0,
    double tf,
    const std::vector<double>& h_query_times   // host query times
) {
    constexpr int N_EQ = Model::N_EQ;
    int num_systems = static_cast<int>(h_y0.size() / N_EQ);
    int num_queries = static_cast<int>(h_query_times.size());
    if (h_y0.size() != size_t(num_systems * N_EQ)) {
        throw std::runtime_error("run_rk45: h_y0 size mismatch");
    }

    // 1) Allocate device buffers
    double *d_y0_all = nullptr, *d_y_final_all = nullptr;
    double *d_query_times = nullptr, *d_dense_all = nullptr;
    size_t bytes_y0      = sizeof(double) * num_systems * N_EQ;
    size_t bytes_final   = sizeof(double) * num_systems * N_EQ;
    size_t bytes_queries = sizeof(double) * num_queries;
    size_t bytes_dense   = sizeof(double) * num_systems * N_EQ * num_queries;

    cudaError_t err;
    err = cudaMalloc(&d_y0_all, bytes_y0);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_y0_all failed: " + std::string(cudaGetErrorString(err)));
    err = cudaMalloc(&d_y_final_all, bytes_final);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        throw std::runtime_error("cudaMalloc d_y_final_all failed: " + std::string(cudaGetErrorString(err))); 
    }
    err = cudaMalloc(&d_query_times, bytes_queries);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        throw std::runtime_error("cudaMalloc d_query_times failed: " + std::string(cudaGetErrorString(err))); 
    }
    err = cudaMalloc(&d_dense_all, bytes_dense);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        cudaFree(d_query_times);
        throw std::runtime_error("cudaMalloc d_dense_all failed: " + std::string(cudaGetErrorString(err)));
    }

    // 2) Copy data to device
    err = cudaMemcpy(d_y0_all, h_y0.data(), bytes_y0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        cudaFree(d_query_times); 
        cudaFree(d_dense_all);
        throw std::runtime_error("cudaMemcpy to d_y0_all failed: " + std::string(cudaGetErrorString(err))); 
    }
    err = cudaMemcpy(d_query_times, h_query_times.data(), bytes_queries, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        cudaFree(d_query_times); 
        cudaFree(d_dense_all);
        throw std::runtime_error("cudaMemcpy to d_query_times failed: " + std::string(cudaGetErrorString(err))); 
    }

     // 3) Launch kernel
    // Calculate grid and block dimensions
    constexpr int THREADS_X = 16;  // Number of threads per block in X dimension
    constexpr int THREADS_Y = 16;  // Number of threads per block in Y dimension

    int numBlocksX = (num_systems + THREADS_X - 1) / THREADS_X;  // Calculate number of blocks in X
    int numBlocksY = (num_queries + THREADS_Y - 1) / THREADS_Y;  // Calculate number of blocks in Y

    dim3 blocks(numBlocksX, numBlocksY);  // Grid dimensions
    dim3 threadsPerBlock(THREADS_X, THREADS_Y);  // Block dimensions

    // Launch the kernel
    rk45_kernel_multi<Model><<<blocks, threadsPerBlock>>>(
        d_y0_all, d_y_final_all, d_query_times, d_dense_all, num_systems, num_queries, t0, tf
    );

    // 4) Synchronize device and check for errors
    // Synchronize the device to catch runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        cudaFree(d_query_times); 
        cudaFree(d_dense_all);
        throw std::runtime_error("Kernel execution failed: " + std::string(cudaGetErrorString(err))); 
    }

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        cudaFree(d_query_times); 
        cudaFree(d_dense_all);
        throw std::runtime_error("Kernel launch failed: " + std::string(cudaGetErrorString(err))); 
    }

    // 5) Copy results back to host
    std::vector<double> h_y_final_all(num_systems * N_EQ);
    err = cudaMemcpy(h_y_final_all.data(), d_y_final_all, bytes_final, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        cudaFree(d_query_times); 
        cudaFree(d_dense_all);
        throw std::runtime_error("cudaMemcpy to h_y_final_all failed: " + std::string(cudaGetErrorString(err))); 
    }

    std::vector<double> h_dense_all(num_systems * N_EQ * num_queries);
    err = cudaMemcpy(h_dense_all.data(), d_dense_all, bytes_dense, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { 
        cudaFree(d_y0_all); 
        cudaFree(d_y_final_all); 
        cudaFree(d_query_times); 
        cudaFree(d_dense_all);
        throw std::runtime_error("cudaMemcpy to h_dense_all failed: " + std::string(cudaGetErrorString(err))); 
    }

    // 6) Free device memory
    cudaFree(d_y0_all);
    cudaFree(d_y_final_all);
    cudaFree(d_query_times);
    cudaFree(d_dense_all);

    // 7) Return results
    FinalType final_result(h_y_final_all.begin(), h_y_final_all.end());
    DenseType dense_result(h_dense_all.begin(), h_dense_all.end());
    return {std::move(final_result), std::move(dense_result)};
}
} // namespace rk45_api