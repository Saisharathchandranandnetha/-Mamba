#ifndef UMAMBA_ACTIVATIONS_H
#define UMAMBA_ACTIVATIONS_H

#include "../include/config.h"

namespace umamba {
namespace activation {

    tensor_t sigmoid(tensor_t x);
    tensor_t silu(tensor_t x);
    tensor_t softplus(tensor_t x);
    tensor_t tanh_fast(tensor_t x);

    // Vectorized versions applied in-place
    template<int N>
    inline void apply_silu(tensor_t* x) {
        for (int i = 0; i < N; ++i) {
            x[i] = silu(x[i]);
        }
    }

    template<int N>
    inline void apply_softplus(tensor_t* x) {
        for (int i = 0; i < N; ++i) {
            x[i] = softplus(x[i]);
        }
    }
    
    template<int N>
    inline void apply_sigmoid(tensor_t* x) {
        for (int i = 0; i < N; ++i) {
            x[i] = sigmoid(x[i]);
        }
    }

} // namespace activation
} // namespace umamba

#endif // UMAMBA_ACTIVATIONS_H
