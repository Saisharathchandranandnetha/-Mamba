#include "benchmark.h"
#include "model.h"
#include "weights.h"
#ifdef ARDUINO
#include <Arduino.h>
#endif

namespace umamba {
namespace benchmark {

    void run_full_benchmark() {
#ifdef ARDUINO
        MicroMambaModel model;
        using namespace umamba::weights;
        model.set_weights(
            emb_w, in_w, in_b, x_proj_w, x_proj_b,
            out_w, out_b, conv_w, conv_b,
            A, dt_b, lm_head_w, lm_head_b
        );
        model.reset_state();

        tensor_t logits[VOCAB_SIZE];
        
        Serial.println(F("--- Benchmarking 10 tokens ---"));
        unsigned long t0 = micros();
        for (int i = 0; i < 10; ++i) {
            model.forward(1, logits); // use 'a'
        }
        unsigned long t1 = micros();
        
        float ms_per_token = (float)(t1 - t0) / 10.0f / 1000.0f;
        Serial.print(F("Total/token:      "));
        Serial.print(ms_per_token);
        Serial.println(F(" ms"));
        
        Serial.print(F("Tokens/sec:       "));
        Serial.println(1000.0f / ms_per_token);
#endif
    }

} // namespace benchmark
    
    void init_benchmark() {}
} // namespace umamba
