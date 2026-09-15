#include "tokenizer.h"

namespace umamba {
namespace tokenizer {

    // Decoding map: Token ID -> Char
    // Vocab size is 64.
    static char id_to_char[VOCAB_SIZE];
    
    // Encoding map: Char (ASCII) -> Token ID
    static int char_to_id[256];

    void init() {
        for (int i = 0; i < 256; ++i) char_to_id[i] = 0; 
        for (int i = 0; i < VOCAB_SIZE; ++i) id_to_char[i] = ' ';

        id_to_char[0] = ' ';
        char_to_id[(unsigned char)' '] = 0;

        for (int i = 0; i < 26; ++i) {
            char c = 'a' + i;
            id_to_char[1 + i] = c;
            char_to_id[(unsigned char)c] = 1 + i;
        }

        for (int i = 0; i < 26; ++i) {
            char c = 'A' + i;
            id_to_char[27 + i] = c;
            char_to_id[(unsigned char)c] = 27 + i;
        }

        for (int i = 0; i < 10; ++i) {
            char c = '0' + i;
            id_to_char[53 + i] = c;
            char_to_id[(unsigned char)c] = 53 + i;
        }

        id_to_char[63] = '\n';
        char_to_id[(unsigned char)'\n'] = 63;
    }

    size_t encode(const char* text, int* tokens, size_t max_tokens) {
        size_t count = 0;
        while (text[count] != '\0' && count < max_tokens) {
            tokens[count] = char_to_id[(unsigned char)text[count]];
            count++;
        }
        return count;
    }

    char decode(int token_id) {
        if (token_id >= 0 && token_id < VOCAB_SIZE) {
            return id_to_char[token_id];
        }
        return ' ';
    }

    size_t decode(const int* tokens, size_t num_tokens, char* text_out, size_t max_chars) {
        size_t count = 0;
        while (count < num_tokens && count < max_chars - 1) {
            text_out[count] = decode(tokens[count]);
            count++;
        }
        text_out[count] = '\0';
        return count;
    }

} // namespace tokenizer

    // Provide the stub called by umamba.ino
    void init_tokenizer() {
        tokenizer::init();
    }
} // namespace umamba
