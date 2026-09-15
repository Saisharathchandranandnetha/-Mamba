#ifndef UMAMBA_DISTRIBUTED_H
#define UMAMBA_DISTRIBUTED_H

#include "../include/config.h"

namespace umamba {
namespace distributed {

    // Abstract interfaces for future ESP32 distributed cluster architecture
    
    class TensorTransport {
    public:
        virtual ~TensorTransport() = default;
        virtual void send_tensor(const tensor_t* data, int size) = 0;
        virtual void receive_tensor(tensor_t* data, int size) = 0;
    };

    class EmbeddingEngine {
    public:
        virtual ~EmbeddingEngine() = default;
        virtual void forward(int token_id, tensor_t* out) = 0;
    };

    class MambaEngine {
    public:
        virtual ~MambaEngine() = default;
        virtual void step(const tensor_t* x, tensor_t* out) = 0;
        virtual void reset_state() = 0;
    };

    class OutputEngine {
    public:
        virtual ~OutputEngine() = default;
        virtual void forward(const tensor_t* x, tensor_t* logits) = 0;
    };

    class Node {
    public:
        virtual ~Node() = default;
        virtual void init() = 0;
        virtual void loop() = 0;
    };

} // namespace distributed
} // namespace umamba

#endif // UMAMBA_DISTRIBUTED_H

