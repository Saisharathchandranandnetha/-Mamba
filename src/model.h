#ifndef UMAMBA_MODEL_H
#define UMAMBA_MODEL_H

#include "../include/config.h"
#include "embedding.h"
#include "mamba_block.h"
#include "linear.h"

namespace umamba {

    class MicroMambaModel {
    private:
        Embedding embedding;
        TinyMambaBlock block;
        Linear<D_MODEL, VOCAB_SIZE> lm_head;
        
    public:
        MicroMambaModel() {}

        void set_weights(
            const tensor_t* emb_w,
            const tensor_t* in_w, const tensor_t* in_b,
            const tensor_t* x_proj_w, const tensor_t* x_proj_b,
            const tensor_t* out_w, const tensor_t* out_b,
            const tensor_t* conv_w, const tensor_t* conv_b,
            const tensor_t* A, const tensor_t* dt_b,
            const tensor_t* lm_head_w, const tensor_t* lm_head_b
        ) {
            embedding.set_weights(emb_w);
            block.set_weights(in_w, in_b, x_proj_w, x_proj_b, out_w, out_b, conv_w, conv_b, A, dt_b);
            lm_head.set_weights(lm_head_w, lm_head_b);
        }

        void reset_state() {
            block.reset_state();
        }

        // Process a single token ID, return logits
        void forward(int token_id, tensor_t* logits) {
            tensor_t x[D_MODEL];
            
            // 1. Embedding
            embedding.forward(token_id, x);
            
            // 2. Mamba Block
            tensor_t out[D_MODEL];
            block.step(x, out);
            
            // 3. LM Head
            lm_head.forward(out, logits);
        }
    };

    void init_model();

} // namespace umamba

#endif // UMAMBA_MODEL_H
