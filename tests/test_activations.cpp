#include <iostream>
#include <cmath>
#include <cassert>
#include "../src/activations.h"

using namespace umamba::activation;

bool close(float a, float b, float tol=1e-5) {
    return std::fabs(a - b) < tol;
}

void test_sigmoid() {
    assert(close(sigmoid(0.0f), 0.5f));
    assert(close(sigmoid(25.0f), 1.0f));
    assert(close(sigmoid(-25.0f), 0.0f));
    std::cout << "test_sigmoid passed\n";
}

void test_silu() {
    assert(close(silu(0.0f), 0.0f));
    assert(close(silu(20.0f), 20.0f));
    assert(close(silu(-25.0f), 0.0f));
    std::cout << "test_silu passed\n";
}

void test_softplus() {
    assert(close(softplus(0.0f), 0.693147f)); // log(2)
    assert(close(softplus(25.0f), 25.0f));
    std::cout << "test_softplus passed\n";
}

void test_tanh_fast() {
    assert(close(tanh_fast(0.0f), 0.0f));
    assert(close(tanh_fast(20.0f), 1.0f));
    assert(close(tanh_fast(-20.0f), -1.0f));
    std::cout << "test_tanh_fast passed\n";
}

int main() {
    test_sigmoid();
    test_silu();
    test_softplus();
    test_tanh_fast();
    std::cout << "All activation tests passed!\n";
    return 0;
}

