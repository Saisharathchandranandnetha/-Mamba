#include <iostream>
#include <cassert>
#include <cmath>
#include "../src/model.h"
#include "../src/tokenizer.h"
#include "../src/weights.h"

using namespace umamba;
using namespace umamba::weights;

bool close(float a, float b, float tol=1e-5) {
    return std::fabs(a - b) < tol;
}

int main() {
    tokenizer::init();
    
    MicroMambaModel model;
    model.set_weights(
        emb_w,
        in_w, in_b,
        x_proj_w, x_proj_b,
        out_w, out_b,
        conv_w, conv_b,
        A, dt_b,
        lm_head_w, lm_head_b
    );
    
    model.reset_state();
    
    int tokens[16];
    size_t n = tokenizer::encode("hello", tokens, 16);
    assert(n == 5);
    
    tensor_t logits[VOCAB_SIZE];
    
    // Step 0
    model.forward(tokens[0], logits);
    
    // Python output LOGITS_0[0:5]: [-0.00790675  0.03642783 -0.02780319  0.02889601 -0.04025494]
    assert(close(logits[0], -0.00790675f, 1e-4));
    assert(close(logits[1],  0.03642783f, 1e-4));
    assert(close(logits[2], -0.02780319f, 1e-4));
    assert(close(logits[3],  0.02889601f, 1e-4));
    assert(close(logits[4], -0.04025494f, 1e-4));
    
    // Run remaining tokens
    for(size_t i=1; i<n; ++i) {
        model.forward(tokens[i], logits);
    }
    
    // Generate 5 tokens
    int curr_token = tokens[n-1];
    std::cout << "C++ GENERATED:\n";
    for(int i=0; i<5; ++i) {
        model.forward(curr_token, logits);
        
        int next_token = 0;
        tensor_t best = logits[0];
        for(int v=1; v<VOCAB_SIZE; ++v) {
            if(logits[v] > best) {
                best = logits[v];
                next_token = v;
            }
        }
        std::cout << "TOKEN ID: " << next_token << " CHAR: '" << tokenizer::decode(next_token) << "'\n";
        curr_token = next_token;
        
        // Assert matches Python reference: 54 ('1')
        assert(next_token == 54);
    }
    
    std::cout << "Full Model Python reference match tests passed!\n";
    return 0;
}

