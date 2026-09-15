#ifndef UMAMBA_TOKENIZER_H
#define UMAMBA_TOKENIZER_H

#include "../include/config.h"
#include <stddef.h>

namespace umamba {
namespace tokenizer {

    // Initialize the tokenizer mapping (called once)
    void init();

    // Encode a string into token IDs. Returns number of tokens written.
    size_t encode(const char* text, int* tokens, size_t max_tokens);

    // Decode a token ID back to a character
    char decode(int token_id);

    // Decode a sequence of token IDs to a string buffer. Returns number of chars written.
    size_t decode(const int* tokens, size_t num_tokens, char* text_out, size_t max_chars);

} // namespace tokenizer
    
    // Provide the stub called by umamba.ino
    void init_tokenizer();

} // namespace umamba

#endif // UMAMBA_TOKENIZER_H
