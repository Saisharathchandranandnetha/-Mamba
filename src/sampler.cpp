#include "sampler.h"
#include <math.h>

namespace umamba {
namespace sampler {

    static unsigned int next_rand = 1;

    void seed(unsigned int s) {
        next_rand = s;
    }

    // Linear congruential generator (simple, deterministic)
    float random_float() {
        next_rand = next_rand * 1103515245 + 12345;
        unsigned int val = (next_rand / 65536) % 32768;
        return (float)val / 32768.0f;
    }

    int argmax(const tensor_t* logits, int size) {
        int best_idx = 0;
        tensor_t best_val = logits[0];
        for (int i = 1; i < size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    int sample(const tensor_t* logits, int size, float temperature) {
        if (temperature <= 0.0f) {
            return argmax(logits, size);
        }

        // Apply temperature and compute max for numerical stability
        tensor_t max_val = logits[0] / temperature;
        for (int i = 1; i < size; ++i) {
            tensor_t v = logits[i] / temperature;
            if (v > max_val) max_val = v;
        }

        // Compute exp and sum
        tensor_t sum = 0.0f;
        tensor_t exp_probs[VOCAB_SIZE]; // Assuming size <= VOCAB_SIZE
        
        for (int i = 0; i < size; ++i) {
            exp_probs[i] = expf(logits[i] / temperature - max_val);
            sum += exp_probs[i];
        }

        // Sample
        float r = random_float() * sum;
        float cum = 0.0f;
        for (int i = 0; i < size; ++i) {
            cum += exp_probs[i];
            if (r <= cum) {
                return i;
            }
        }
        return size - 1; // Fallback
    }

} // namespace sampler

    void init_sampler() {
        sampler::seed(42);
    }
} // namespace umamba
