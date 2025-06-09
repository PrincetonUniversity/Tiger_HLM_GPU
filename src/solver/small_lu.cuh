//src/solver/small_lu.cuh
#pragma once
#include <cuda_runtime.h>

/**
 * Solve A·x = b in place for a small dense matrix A[n×n] and vector b[n].
 * Uses unpivoted Gaussian elimination.  n must be ≤ ~10 or so.
 *
 * @param n    Dimension
 * @param A    Row‐major array of length n*n (will be overwritten)
 * @param b    Right‐hand side of length n (will be overwritten with solution)
 */
__forceinline__ __device__ static void small_matrix_LU_solve(int n, double* A, double* b) {
    // Forward elimination
    for (int k = 0; k < n; ++k) {
        double pivot = A[k*n + k];
        // assume pivot != 0
        for (int j = k+1; j < n; ++j) {
            A[k*n + j] /= pivot;
        }
        b[k] /= pivot;
        A[k*n + k] = 1.0;
        for (int i = k+1; i < n; ++i) {
            double m = A[i*n + k];
            for (int j = k+1; j < n; ++j) {
                A[i*n + j] -= m * A[k*n + j];
            }
            b[i] -= m * b[k];
            A[i*n + k] = 0.0;
        }
    }
    // Back substitution
    for (int i = n-1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i+1; j < n; ++j) {
            sum -= A[i*n + j] * b[j];
        }
        b[i] = sum;  // A[i*n+i] should be 1.0 here
    }
}
