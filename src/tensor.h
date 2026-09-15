#ifndef UMAMBA_TENSOR_H
#define UMAMBA_TENSOR_H

#include "../include/config.h"

namespace umamba {
namespace tensor {

    // Fills array x of size N with value val
    template<int N>
    inline void fill(tensor_t* x, tensor_t val) {
        for (int i = 0; i < N; ++i) {
            x[i] = val;
        }
    }

    // Zeros array x of size N
    template<int N>
    inline void zero(tensor_t* x) {
        fill<N>(x, 0.0f);
    }

    // Copies array src to dest of size N
    template<int N>
    inline void copy(const tensor_t* src, tensor_t* dest) {
        for (int i = 0; i < N; ++i) {
            dest[i] = src[i];
        }
    }

    // Elementwise addition: z = x + y
    template<int N>
    inline void add(const tensor_t* x, const tensor_t* y, tensor_t* z) {
        for (int i = 0; i < N; ++i) {
            z[i] = x[i] + y[i];
        }
    }

    // Elementwise multiplication: z = x * y
    template<int N>
    inline void mul(const tensor_t* x, const tensor_t* y, tensor_t* z) {
        for (int i = 0; i < N; ++i) {
            z[i] = x[i] * y[i];
        }
    }

    // Vector addition (scale + add): z = x + a
    template<int N>
    inline void add_scalar(const tensor_t* x, tensor_t a, tensor_t* z) {
        for (int i = 0; i < N; ++i) {
            z[i] = x[i] + a;
        }
    }

    // Vector multiplication by scalar: z = x * a
    template<int N>
    inline void mul_scalar(const tensor_t* x, tensor_t a, tensor_t* z) {
        for (int i = 0; i < N; ++i) {
            z[i] = x[i] * a;
        }
    }

    // Dot product of vectors x and y of size N
    template<int N>
    inline tensor_t dot(const tensor_t* x, const tensor_t* y) {
        tensor_t sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    // Matrix-vector multiplication: y = A * x
    // A is (M x N), x is (N), y is (M)
    template<int M, int N>
    inline void matvec(const tensor_t* A, const tensor_t* x, tensor_t* y) {
        for (int i = 0; i < M; ++i) {
            tensor_t sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
    }

    // Matrix multiplication: C = A * B
    // A is (M x K), B is (K x N), C is (M x N)
    template<int M, int N, int K>
    inline void matmul(const tensor_t* A, const tensor_t* B, tensor_t* C) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                tensor_t sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    // Matrix transpose: B = A^T
    // A is (M x N), B is (N x M)
    template<int M, int N>
    inline void transpose(const tensor_t* A, tensor_t* B) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                B[j * M + i] = A[i * N + j];
            }
        }
    }

} // namespace tensor
} // namespace umamba

#endif // UMAMBA_TENSOR_H
