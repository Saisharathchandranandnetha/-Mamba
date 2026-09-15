#ifndef UMAMBA_CONFIG_H
#define UMAMBA_CONFIG_H

#ifdef ARDUINO
#include <Arduino.h>
#endif

namespace umamba {

    // Model Dimensions
    constexpr int VOCAB_SIZE = 64;
    constexpr int D_MODEL = 16;
    constexpr int D_INNER = 16;
    constexpr int D_STATE = 8;
    constexpr int D_CONV = 4;
    constexpr int N_LAYERS = 1;
    constexpr int SEQ_LEN = 64;

    // Type definitions
    typedef float tensor_t;

} // namespace umamba

#endif // UMAMBA_CONFIG_H
