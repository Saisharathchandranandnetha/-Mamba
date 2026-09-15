#include <iostream>
#include <cstring>
#include <cassert>
#include "../src/tokenizer.h"

using namespace umamba;

int main() {
    tokenizer::init();
    
    const char* text = "Hello World 123\n";
    int tokens[32];
    
    size_t n = tokenizer::encode(text, tokens, 32);
    assert(n == 16);
    
    char decoded[32];
    size_t m = tokenizer::decode(tokens, n, decoded, 32);
    
    assert(m == n);
    assert(std::strcmp(text, decoded) == 0);
    
    std::cout << "Tokenizer tests passed!\n";
    return 0;
}
