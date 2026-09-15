#include <iostream>
#include <cassert>
#include <cmath>
#include "../src/mamba_block.h"

using namespace umamba;

bool close(float a, float b, float tol=1e-5) {
    return std::fabs(a - b) < tol;
}

int main() {
    TinyMambaBlock block;
    
    // Create dummy weights
    tensor_t in_w[2 * D_INNER * D_MODEL];
    tensor_t in_b[2 * D_INNER];
    
    tensor_t x_proj_w[(D_INNER + 2 * D_STATE) * D_INNER];
    tensor_t x_proj_b[D_INNER + 2 * D_STATE];
    
    tensor_t out_w[D_MODEL * D_INNER];
    tensor_t out_b[D_MODEL];
    
    tensor_t conv_w[D_INNER * D_CONV];
    tensor_t conv_b[D_INNER];
    
    tensor_t A[D_INNER * D_STATE];
    tensor_t dt_b[D_INNER];
    
    // Init to small values
    for(int i=0; i<2*D_INNER*D_MODEL; ++i) in_w[i] = 0.01f;
    for(int i=0; i<2*D_INNER; ++i) in_b[i] = 0.0f;
    for(int i=0; i<(D_INNER+2*D_STATE)*D_INNER; ++i) x_proj_w[i] = 0.01f;
    for(int i=0; i<D_INNER+2*D_STATE; ++i) x_proj_b[i] = 0.0f;
    for(int i=0; i<D_MODEL*D_INNER; ++i) out_w[i] = 0.01f;
    for(int i=0; i<D_MODEL; ++i) out_b[i] = 0.0f;
    for(int i=0; i<D_INNER*D_CONV; ++i) conv_w[i] = 0.1f;
    for(int i=0; i<D_INNER; ++i) conv_b[i] = 0.0f;
    for(int i=0; i<D_INNER*D_STATE; ++i) A[i] = -0.1f;
    for(int i=0; i<D_INNER; ++i) dt_b[i] = 0.1f;
    
    block.set_weights(in_w, in_b, x_proj_w, x_proj_b, out_w, out_b, conv_w, conv_b, A, dt_b);
    
    tensor_t x[D_MODEL];
    for(int i=0; i<D_MODEL; ++i) x[i] = 0.5f;
    
    tensor_t out[D_MODEL];
    
    block.step(x, out);
    
    assert(!std::isnan(out[0]));
    std::cout << "Mamba block basic forward pass passed!\n";
    
    return 0;
}

