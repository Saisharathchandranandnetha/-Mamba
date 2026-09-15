#ifndef UMAMBA_MAMBA_BLOCK_H
#define UMAMBA_MAMBA_BLOCK_H

#include "../include/config.h"
#include "tensor.h"
#include "linear.h"
#include "ssm.h"
#include "activations.h"

namespace umamba {

    class TinyMambaBlock {
    private:
        // Linear layers
        Linear<D_MODEL, 2 * D_INNER> in_proj;
        Linear<D_INNER, D_INNER + 2 * D_STATE> x_proj;
        Linear<D_INNER, D_MODEL> out_proj;
        
        // Conv1D weights
        const tensor_t* conv1d_weight; // (D_INNER, D_CONV)
        const tensor_t* conv1d_bias;   // (D_INNER)

        // dt bias
        const tensor_t* dt_bias; // (D_INNER)

        // Selective SSM
        SelectiveSSM ssm;

        // Conv state (circular buffer or shift register)
        // Since D_CONV is small (e.g., 4), a simple shift register is fine.
        tensor_t conv_state[D_INNER * D_CONV];

    public:
        TinyMambaBlock() : conv1d_weight(nullptr), conv1d_bias(nullptr), dt_bias(nullptr) {
            reset_state();
        }

        void set_weights(
            const tensor_t* in_w, const tensor_t* in_b,
            const tensor_t* x_proj_w, const tensor_t* x_proj_b,
            const tensor_t* out_w, const tensor_t* out_b,
            const tensor_t* conv_w, const tensor_t* conv_b,
            const tensor_t* A, const tensor_t* dt_b
        ) {
            in_proj.set_weights(in_w, in_b);
            x_proj.set_weights(x_proj_w, x_proj_b);
            out_proj.set_weights(out_w, out_b);
            conv1d_weight = conv_w;
            conv1d_bias = conv_b;
            dt_bias = dt_b;
            ssm.set_weights(A);
        }

        void reset_state() {
            tensor::zero<D_INNER * D_CONV>(conv_state);
            ssm.reset_state();
        }

        void step(const tensor_t* x, tensor_t* out) {
            tensor_t xz[2 * D_INNER];
            
            // 1. in_proj
            in_proj.forward(x, xz);
            
            tensor_t* x_inner = &xz[0];
            tensor_t* z_inner = &xz[D_INNER];
            
            // 2. Conv1D over x_inner
            for (int d = 0; d < D_INNER; ++d) {
                int offset = d * D_CONV;
                // Shift register
                for (int c = D_CONV - 1; c > 0; --c) {
                    conv_state[offset + c] = conv_state[offset + c - 1];
                }
                conv_state[offset] = x_inner[d];
                
                // Compute conv
                tensor_t val = conv1d_bias ? conv1d_bias[d] : 0.0f;
                for (int c = 0; c < D_CONV; ++c) {
                    val += conv_state[offset + c] * conv1d_weight[offset + c];
                }
                x_inner[d] = val;
            }
            
            // 3. silu(x)
            activation::apply_silu<D_INNER>(x_inner);
            
            // 4. x_proj to get dt, B, C
            tensor_t dt_B_C[D_INNER + 2 * D_STATE];
            x_proj.forward(x_inner, dt_B_C);
            
            tensor_t* dt = &dt_B_C[0];
            tensor_t* B = &dt_B_C[D_INNER];
            tensor_t* C = &dt_B_C[D_INNER + D_STATE];
            
            // 5. dt = softplus(dt + dt_bias)
            if (dt_bias) {
                tensor::add<D_INNER>(dt, dt_bias, dt);
            }
            activation::apply_softplus<D_INNER>(dt);
            
            // 6. SSM step
            tensor_t y[D_INNER];
            ssm.step(x_inner, dt, B, C, y);
            
            // 7. Gating: y = y * silu(z)
            activation::apply_silu<D_INNER>(z_inner);
            tensor::mul<D_INNER>(y, z_inner, y);
            
            // 8. out_proj
            out_proj.forward(y, out);
        }
    };

    void init_mamba_block();

} // namespace umamba

#endif // UMAMBA_MAMBA_BLOCK_H
