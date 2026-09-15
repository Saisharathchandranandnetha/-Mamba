#ifndef UMAMBA_SSM_H
#define UMAMBA_SSM_H

#include "../include/config.h"
#include "tensor.h"
#include <math.h>

namespace umamba {

    // Standard State Space Model (Fixed parameters)
    // Runs independently over D_INNER channels.
    class TinySSM {
    private:
        // Parameters (pointers to PROGMEM or SRAM)
        const tensor_t* A; // (D_INNER, D_STATE)
        const tensor_t* B; // (D_INNER, D_STATE)
        const tensor_t* C; // (D_INNER, D_STATE)
        const tensor_t* D; // (D_INNER)

        // Runtime State
        tensor_t h[D_INNER * D_STATE];

    public:
        TinySSM() : A(nullptr), B(nullptr), C(nullptr), D(nullptr) {
            reset_state();
        }

        void set_weights(const tensor_t* wA, const tensor_t* wB, const tensor_t* wC, const tensor_t* wD) {
            A = wA; B = wB; C = wC; D = wD;
        }

        void reset_state() {
            tensor::zero<D_INNER * D_STATE>(h);
        }

        // Process a single timestep
        // x: (D_INNER)
        // y: (D_INNER)
        void step(const tensor_t* x, tensor_t* y) {
            for (int d = 0; d < D_INNER; ++d) {
                tensor_t y_d = 0.0f;
                tensor_t x_d = x[d];
                
                int offset = d * D_STATE;
                
                for (int s = 0; s < D_STATE; ++s) {
                    // h_t = A * h_{t-1} + B * x_t
                    tensor_t h_new = A[offset + s] * h[offset + s] + B[offset + s] * x_d;
                    h[offset + s] = h_new;
                    
                    // y_t = C * h_t
                    y_d += C[offset + s] * h_new;
                }
                
                // + D * x_t
                if (D) {
                    y_d += D[d] * x_d;
                }
                y[d] = y_d;
            }
        }
    };

    // Selective State Space Model
    class SelectiveSSM {
    private:
        const tensor_t* A; // (D_INNER, D_STATE)
        
        tensor_t h[D_INNER * D_STATE];

    public:
        SelectiveSSM() : A(nullptr) {
            reset_state();
        }

        void set_weights(const tensor_t* wA) {
            A = wA;
        }

        void reset_state() {
            tensor::zero<D_INNER * D_STATE>(h);
        }

        // Process a single timestep
        // x: (D_INNER)
        // dt: (D_INNER) - input dependent deltas
        // B: (D_STATE) - input dependent B
        // C: (D_STATE) - input dependent C
        // y: (D_INNER)
        void step(const tensor_t* x, const tensor_t* dt, const tensor_t* B, const tensor_t* C, tensor_t* y) {
            for (int d = 0; d < D_INNER; ++d) {
                tensor_t y_d = 0.0f;
                tensor_t x_d = x[d];
                tensor_t dt_d = dt[d]; // already softplus applied outside
                
                int offset = d * D_STATE;
                
                for (int s = 0; s < D_STATE; ++s) {
                    tensor_t A_val = A[offset + s];
                    // Using Euler approximation for simplicity and speed:
                    // A_bar = exp(dt * A) ≈ 1 + dt * A (if dt*A is small enough)
                    // Or exact: expf(dt_d * A_val)
                    // Let's use exact for correctness, the FPU can handle it
                    // Actually, to match Mamba closely, ZOH is often:
                    // A_bar = exp(dt * A)
                    // B_bar = (exp(dt * A) - 1) / A * dt * B 
                    // However, simplified Euler is common in tiny implementations:
                    tensor_t A_bar = expf(dt_d * A_val);
                    tensor_t B_bar = dt_d * B[s];
                    
                    tensor_t h_new = A_bar * h[offset + s] + B_bar * x_d;
                    h[offset + s] = h_new;
                    
                    y_d += C[s] * h_new;
                }
                
                y[d] = y_d;
            }
        }
    };

    void init_ssm();

} // namespace umamba

#endif // UMAMBA_SSM_H
