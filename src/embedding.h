#ifndef UMAMBA_EMBEDDING_H
#define UMAMBA_EMBEDDING_H

#include "../include/config.h"
#include "tensor.h"

namespace umamba {

    // Simple Embedding Layer
    // Stores pointer to weight matrix: shape (VOCAB_SIZE, D_MODEL)
    class Embedding {
    private:
        const tensor_t* weight; // size: VOCAB_SIZE * D_MODEL
    
    public:
        Embedding() : weight(nullptr) {}

        // Bind weights
        void set_weights(const tensor_t* w) {
            weight = w;
        }

        // Lookup single token embedding
        void forward(int token_id, tensor_t* out) const {
            if (!weight) return;
            if (token_id < 0 || token_id >= VOCAB_SIZE) {
                token_id = 0; // fallback to space/pad
            }
            // copy row token_id from weight matrix to out
            tensor::copy<D_MODEL>(&weight[token_id * D_MODEL], out);
        }
    };

    void init_embedding();

} // namespace umamba

#endif // UMAMBA_EMBEDDING_H
