#ifndef UMAMBA_SAMPLER_H
#define UMAMBA_SAMPLER_H

#include "../include/config.h"

namespace umamba {
namespace sampler {

    // Simple pseudo-random number generator for deterministic behavior
    void seed(unsigned int s);
    float random_float();

    // Argmax sampling
    int argmax(const tensor_t* logits, int size);

    // Temperature sampling
    int sample(const tensor_t* logits, int size, float temperature);

} // namespace sampler
} // namespace umamba

#endif // UMAMBA_SAMPLER_H
