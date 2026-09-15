#!/bin/bash
PROJECT_DIR="/home/netha/Projects/µMamba"
SRC_DIR="$PROJECT_DIR/src"
INC_DIR="$PROJECT_DIR/include"

headers=(memory tensor activations linear tokenizer embedding ssm mamba_block sampler model benchmark)

for h in "${headers[@]}"; do
    NAME=$(echo $h | tr '[:lower:]' '[:upper:]')
    
    cat << H_EOF > "$SRC_DIR/$h.h"
#ifndef UMAMBA_${NAME}_H
#define UMAMBA_${NAME}_H

#include "../include/config.h"

namespace umamba {
    void init_${h}();
}

#endif
H_EOF

    cat << C_EOF > "$SRC_DIR/$h.cpp"
#include "${h}.h"

namespace umamba {
    void init_${h}() {
        // stub
    }
}
C_EOF
done
