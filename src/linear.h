#ifndef UMAMBA_LINEAR_H
#define UMAMBA_LINEAR_H

#include "../include/config.h"
#include "tensor.h"

namespace umamba {

    // Linear Layer: y = W * x + b
    // W is (OUT_F, IN_F), b is (OUT_F), x is (IN_F), y is (OUT_F)
    template<int IN_F, int OUT_F>
    class Linear {
    private:
        const tensor_t* W; // (OUT_F, IN_F)
        const tensor_t* b; // (OUT_F)
    
    public:
        Linear() : W(nullptr), b(nullptr) {}
        
        void set_weights(const tensor_t* weight, const tensor_t* bias = nullptr) {
            W = weight;
            b = bias;
        }

        void forward(const tensor_t* x, tensor_t* y) const {
            tensor::matvec<OUT_F, IN_F>(W, x, y);
            if (b) {
                tensor::add<OUT_F>(y, b, y);
            }
        }
    };

    void init_linear();

} // namespace umamba

#endif // UMAMBA_LINEAR_H
