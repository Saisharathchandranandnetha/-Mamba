#include <iostream>
#include <cmath>
#include <cassert>
#include "../src/tensor.h"

using namespace umamba::tensor;

bool close(float a, float b, float tol=1e-5) {
    return std::fabs(a - b) < tol;
}

void test_fill_zero_copy() {
    float x[4];
    float y[4];
    
    fill<4>(x, 3.14f);
    for (int i=0; i<4; ++i) assert(close(x[i], 3.14f));
    
    zero<4>(x);
    for (int i=0; i<4; ++i) assert(close(x[i], 0.0f));
    
    x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f; x[3] = 4.0f;
    copy<4>(x, y);
    for (int i=0; i<4; ++i) assert(close(x[i], y[i]));
    std::cout << "test_fill_zero_copy passed\n";
}

void test_elementwise() {
    float x[3] = {1.0f, 2.0f, 3.0f};
    float y[3] = {4.0f, 5.0f, 6.0f};
    float z[3];
    
    add<3>(x, y, z);
    assert(close(z[0], 5.0f)); assert(close(z[1], 7.0f)); assert(close(z[2], 9.0f));
    
    mul<3>(x, y, z);
    assert(close(z[0], 4.0f)); assert(close(z[1], 10.0f)); assert(close(z[2], 18.0f));
    
    add_scalar<3>(x, 10.0f, z);
    assert(close(z[0], 11.0f)); assert(close(z[1], 12.0f)); assert(close(z[2], 13.0f));
    
    mul_scalar<3>(x, 2.0f, z);
    assert(close(z[0], 2.0f)); assert(close(z[1], 4.0f)); assert(close(z[2], 6.0f));
    
    std::cout << "test_elementwise passed\n";
}

void test_dot() {
    float x[3] = {1.0f, 2.0f, 3.0f};
    float y[3] = {4.0f, 5.0f, 6.0f};
    float d = dot<3>(x, y);
    assert(close(d, 32.0f)); // 4 + 10 + 18 = 32
    std::cout << "test_dot passed\n";
}

void test_matvec() {
    float A[2*3] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float x[3] = {1.0f, 2.0f, 3.0f};
    float y[2];
    
    matvec<2, 3>(A, x, y);
    assert(close(y[0], 14.0f)); // 1*1 + 2*2 + 3*3 = 14
    assert(close(y[1], 32.0f)); // 4*1 + 5*2 + 6*3 = 32
    std::cout << "test_matvec passed\n";
}

void test_matmul() {
    float A[2*3] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float B[3*2] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    float C[2*2];
    
    matmul<2, 2, 3>(A, B, C);
    assert(close(C[0], 58.0f));  // 1*7 + 2*9 + 3*11 = 58
    assert(close(C[1], 64.0f));  // 1*8 + 2*10 + 3*12 = 64
    assert(close(C[2], 139.0f)); // 4*7 + 5*9 + 6*11 = 139
    assert(close(C[3], 154.0f)); // 4*8 + 5*10 + 6*12 = 154
    std::cout << "test_matmul passed\n";
}

void test_transpose() {
    float A[2*3] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float B[3*2];
    
    transpose<2, 3>(A, B);
    assert(close(B[0], 1.0f));
    assert(close(B[1], 4.0f));
    assert(close(B[2], 2.0f));
    assert(close(B[3], 5.0f));
    assert(close(B[4], 3.0f));
    assert(close(B[5], 6.0f));
    std::cout << "test_transpose passed\n";
}

int main() {
    test_fill_zero_copy();
    test_elementwise();
    test_dot();
    test_matvec();
    test_matmul();
    test_transpose();
    std::cout << "All tensor tests passed!\n";
    return 0;
}

