#include "memory.h"
#ifdef ARDUINO
#include <Arduino.h>

extern "C" char *sbrk(int incr);

namespace umamba {
namespace memory {

    int free_ram() {
        char top;
        return &top - reinterpret_cast<char*>(sbrk(0));
    }

    void print_report() {
        Serial.println(F("===== \xC2\xB5Mamba Memory ====="));
        
        // Compute memory stats based on compile-time config
        int model_weights = (
            VOCAB_SIZE * D_MODEL +              // emb
            2 * D_INNER * D_MODEL + 2 * D_INNER + // in_proj
            (D_INNER + 2 * D_STATE) * D_INNER + (D_INNER + 2 * D_STATE) + // x_proj
            D_MODEL * D_INNER + D_MODEL +       // out_proj
            D_INNER * D_CONV + D_INNER +        // conv
            D_INNER * D_STATE + D_INNER +       // A, dt_bias
            VOCAB_SIZE * D_MODEL + VOCAB_SIZE   // lm_head
        ) * 4;
        
        int ssm_state = (D_INNER * D_STATE + D_INNER * D_CONV) * 4;
        int activations = D_MODEL * 4 * 4; // approx 4x D_MODEL buffers during forward
        
        Serial.print(F("Model weights:       ")); Serial.print(model_weights); Serial.println(F(" bytes"));
        Serial.print(F("SSM state:           ")); Serial.print(ssm_state); Serial.println(F(" bytes"));
        Serial.print(F("Activations:         ")); Serial.print(activations); Serial.println(F(" bytes"));
        Serial.print(F("Tokenizer:           ")); Serial.print(256 + VOCAB_SIZE); Serial.println(F(" bytes"));
        
        Serial.println();
        Serial.print(F("SRAM available:      ")); Serial.print(free_ram()); Serial.println(F(" bytes"));
        Serial.println(F("========================"));
    }

} // namespace memory
} // namespace umamba

#else
namespace umamba {
namespace memory {
    int free_ram() { return 1024*1024; } // Dummy for PC
    void print_report() {}
}
}
#endif

namespace umamba {
    void init_memory() {}
}
