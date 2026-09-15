#ifndef UMAMBA_MEMORY_H
#define UMAMBA_MEMORY_H

#include "../include/config.h"
#include <stddef.h>

namespace umamba {
namespace memory {

    // Returns estimated free SRAM
    int free_ram();

    // Prints a formatted memory report over Serial
    void print_report();

} // namespace memory
} // namespace umamba

#endif // UMAMBA_MEMORY_H
