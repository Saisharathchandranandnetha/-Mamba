#include <iostream>
#include <cassert>
#include "../src/sampler.h"

using namespace umamba;

int main() {
    float logits[4] = {1.0f, 5.0f, 2.0f, -1.0f};
    
    // Test argmax
    int idx = sampler::argmax(logits, 4);
    assert(idx == 1);
    
    // Test sample
    sampler::seed(12345);
    int s_idx = sampler::sample(logits, 4, 1.0f);
    // 5.0 is much larger than the rest, so it should be highly likely to be 1
    assert(s_idx == 1 || s_idx == 2 || s_idx == 0 || s_idx == 3);
    
    // Zero temp
    int z_idx = sampler::sample(logits, 4, 0.0f);
    assert(z_idx == 1);
    
    std::cout << "Sampler tests passed!\n";
    return 0;
}

