#include "activations.h"
#include <math.h>

namespace umamba {
namespace activation {

    tensor_t sigmoid(tensor_t x) {
        if (x >= 20.0f) return 1.0f;
        if (x <= -20.0f) return 0.0f;
        return 1.0f / (1.0f + expf(-x));
    }

    tensor_t silu(tensor_t x) {
        return x * sigmoid(x);
    }

    tensor_t softplus(tensor_t x) {
        if (x > 20.0f) return x; // Avoid overflow
        return logf(1.0f + expf(x));
    }

    tensor_t tanh_fast(tensor_t x) {
        // We can just use the standard library tanhf
        // which is usually optimized enough for FP32.
        return tanhf(x);
    }

} // namespace activation
} // namespace umamba
