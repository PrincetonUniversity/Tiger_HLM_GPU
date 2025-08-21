#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <tuple>                    
#include <cstring>     

#include "rk45.h"                           // For rk45_then_radau_multi<…> and radau_kernel_multi<…>
#include "../I_O/parameters_loader.hpp"     // for SpatialParams
#include "../models/model_Runoff5.hpp"      // For Runoff5::N_EQ, ::SP_TYPE, etc.
#include "../I_O/forcing_data.h"            // For ForcingData structure


// ────────── Forward‐declare Radau‐only kernel ──────────
template <class Model>
__global__
void radau_kernel_multi(
    double* y0_all, 
    double* y_final_all, 
    double* query_times, 
    // double* dense_all, 
    float* dense_all, // Use float for dense output to save memory
    int num_systems,   
    int num_queries,   
    double t0,    
    double tf,       
    const typename Model::SP_TYPE* d_sp,       
    int* stiff_system_indices,    
    int n_stiff,     
    const float* d_forc_data,
    int nForc
);



namespace rk45_api {

// using DenseType = std::vector<double>;
using DenseType = std::vector<float>; // Use float for dense output to save memory
using FinalType = std::vector<double>;

// ───────── 1) Allocate GPU buffers & copy inputs ─────────
//   • h_y0 size   = num_systems * Runoff5::N_EQ
//   • h_query_times size = num_queries
//   • Also: allocate an int flag per system for stiffness detection.
// Returns a tuple of raw device pointers plus sizes for later teardown.
template<class Runoff5>
auto setup_gpu_buffers(
    const std::vector<double>& h_y0,
    const std::vector<double>& h_query_times
) {
    constexpr int N_EQ = Runoff5::N_EQ;

    // Logical sizes (kept as int to match the rest of the codebase API)
    const int num_systems = static_cast<int>(h_y0.size() / N_EQ);
    const int num_queries = static_cast<int>(h_query_times.size());

    // ---- Element counts (size_t) ----
    const size_t y_elems     = size_t(num_systems) * size_t(N_EQ);
    const size_t final_elems = y_elems;
    const size_t query_elems = size_t(num_queries);
    const size_t dense_elems = y_elems * size_t(num_queries);  // systems * N_EQ * queries
    const size_t stiff_elems = size_t(num_systems);

    // Guard against overflow in products
    if (num_systems > 0 && num_queries > 0) {
        const size_t back = (dense_elems / size_t(N_EQ)) / size_t(num_queries);
        if (back != size_t(num_systems)) {
            throw std::runtime_error("Overflow computing dense_elems");
        }
    }

    // ---- Byte sizes ----
    const size_t bytes_y0      = y_elems     * sizeof(double);
    const size_t bytes_final   = final_elems * sizeof(double);
    const size_t bytes_queries = query_elems * sizeof(double);
    const size_t bytes_dense   = dense_elems * sizeof(float);   // float dense to save memory
    const size_t bytes_stiff   = stiff_elems * sizeof(int);

    // ---- Device pointers ----
    double *d_y0_all = nullptr, *d_y_final_all = nullptr;
    double *d_query_times = nullptr;
    float  *d_dense_all = nullptr;
    int    *d_stiff = nullptr;

    // ---- Allocate device memory (with cleanup on failure) ----
    cudaError_t err;

    err = cudaMalloc(&d_y0_all, bytes_y0);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc d_y0_all failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_y_final_all, bytes_final);
    if (err != cudaSuccess) {
        cudaFree(d_y0_all);
        throw std::runtime_error(std::string("cudaMalloc d_y_final_all failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_query_times, bytes_queries);
    if (err != cudaSuccess) {
        cudaFree(d_y0_all);
        cudaFree(d_y_final_all);
        throw std::runtime_error(std::string("cudaMalloc d_query_times failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_dense_all, bytes_dense);
    if (err != cudaSuccess) {
        cudaFree(d_y0_all);
        cudaFree(d_y_final_all);
        cudaFree(d_query_times);
        throw std::runtime_error(std::string("cudaMalloc d_dense_all failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_stiff, bytes_stiff);
    if (err != cudaSuccess) {
        cudaFree(d_y0_all);
        cudaFree(d_y_final_all);
        cudaFree(d_query_times);
        cudaFree(d_dense_all);
        throw std::runtime_error(std::string("cudaMalloc d_stiff failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMemset(d_stiff, 0, bytes_stiff);
    if (err != cudaSuccess) {
        cudaFree(d_y0_all);
        cudaFree(d_y_final_all);
        cudaFree(d_query_times);
        cudaFree(d_dense_all);
        cudaFree(d_stiff);
        throw std::runtime_error(std::string("cudaMemset d_stiff failed: ") +
                                 cudaGetErrorString(err));
    }

    // ---- Copy host → device ----
    err = cudaMemcpy(d_y0_all, h_y0.data(), bytes_y0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_y0_all);
        cudaFree(d_y_final_all);
        cudaFree(d_query_times);
        cudaFree(d_dense_all);
        cudaFree(d_stiff);
        throw std::runtime_error(std::string("cudaMemcpy to d_y0_all failed: ") +
                                 cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_query_times, h_query_times.data(), bytes_queries, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_y0_all);
        cudaFree(d_y_final_all);
        cudaFree(d_query_times);
        cudaFree(d_dense_all);
        cudaFree(d_stiff);
        throw std::runtime_error(std::string("cudaMemcpy to d_query_times failed: ") +
                                 cudaGetErrorString(err));
    }

    return std::make_tuple(
        d_y0_all, d_y_final_all,
        d_query_times, d_dense_all,
        d_stiff,
        num_systems, num_queries
    );
}

template<class Runoff5>
std::pair<FinalType, DenseType> retrieve_and_free(
    double* d_y0_all,
    double* d_y_final_all,
    double* d_query_times,
    float*  d_dense_all,   // float dense on device
    int*    d_stiff,
    int     num_systems,
    int     num_queries,
    double  /*t0*/,
    double  /*tf*/,
    const typename Runoff5::SP_TYPE* /*d_sp*/
) {
    constexpr int N_EQ = Runoff5::N_EQ;

    // ——————— Element counts (size_t) ———————————
    const size_t y_elems     = size_t(num_systems) * size_t(N_EQ);
    const size_t dense_elems = y_elems * size_t(num_queries);
    const size_t stiff_elems = size_t(num_systems);

    // Guard against overflow
    if (num_systems > 0 && num_queries > 0) {
        const size_t back = (dense_elems / size_t(N_EQ)) / size_t(num_queries);
        if (back != size_t(num_systems)) {
            throw std::runtime_error("Overflow computing dense_elems");
        }
    }

    // ——————— Byte sizes ————————————————————————
    const size_t bytes_final = y_elems     * sizeof(double);
    const size_t bytes_dense = dense_elems * sizeof(float);
    const size_t bytes_stiff = stiff_elems * sizeof(int);
    const size_t bytes_y0    = y_elems     * sizeof(double);

    // ——————— Host buffers (size_t counts) ———————
    FinalType           h_y_final_all(y_elems);    // vector<double>
    DenseType           h_dense_raw(dense_elems);  // vector<float>
    std::vector<int>    h_stiff(stiff_elems);
    std::vector<double> h_y0_all(y_elems);

    // ——————— Copy device → host —————————————————
    auto ck = [](cudaError_t e, const char* where){
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
        }
    };

    ck(cudaMemcpy(h_y_final_all.data(), d_y_final_all, bytes_final, cudaMemcpyDeviceToHost),
       "cudaMemcpy y_final_all");
    ck(cudaMemcpy(h_dense_raw.data(),    d_dense_all,  bytes_dense, cudaMemcpyDeviceToHost),
       "cudaMemcpy dense_all");
    ck(cudaMemcpy(h_stiff.data(),        d_stiff,      bytes_stiff, cudaMemcpyDeviceToHost),
       "cudaMemcpy stiff");
    ck(cudaMemcpy(h_y0_all.data(),       d_y0_all,     bytes_y0,    cudaMemcpyDeviceToHost),
       "cudaMemcpy y0_all");

    // ---- Free device memory ----
    cudaFree(d_stiff);
    cudaFree(d_y0_all);
    cudaFree(d_y_final_all);
    cudaFree(d_query_times);
    cudaFree(d_dense_all);

    // Kernel already wrote dense as [sys][q][comp]; keep order
    DenseType h_dense_all = std::move(h_dense_raw);

    // Overwrite q=0 with the true y0 values (cast to float)
    for (int s = 0; s < num_systems; ++s) {
        for (int c = 0; c < N_EQ; ++c) {
            const size_t dst = (size_t(s) * size_t(num_queries) + size_t(0)) * size_t(N_EQ) + size_t(c);
            h_dense_all[dst] = static_cast<float>(h_y0_all[size_t(s) * size_t(N_EQ) + size_t(c)]);
        }
    }

    return { std::move(h_y_final_all),
             std::move(h_dense_all) };
}



} // namespace rk45_api
