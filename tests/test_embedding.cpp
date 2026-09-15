#include <iostream>
#include <cassert>
#include <cmath>
#include "../src/embedding.h"

using namespace umamba;

bool close(float a, float b, float tol=1e-5) {
    return std::fabs(a - b) < tol;
}

int main() {
    tensor_t dummy_weights[VOCAB_SIZE * D_MODEL];
    
    // Fill with some deterministic values
    for (int i = 0; i < VOCAB_SIZE * D_MODEL; ++i) {
        dummy_weights[i] = (float)i * 0.01f;
    }

    Embedding emb;
    emb.set_weights(dummy_weights);

    tensor_t out[D_MODEL];
    
    // Test token 2 (space)
    emb.forward(2, out);
    
    assert(close(out[0], (2 * D_MODEL + 0) * 0.01f));
    assert(close(out[D_MODEL-1], (2 * D_MODEL + D_MODEL - 1) * 0.01f));
    
    std::cout << "Embedding tests passed!\n";
    return 0;
}
