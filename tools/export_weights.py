import numpy as np

def generate_weights():
    # Model dimensions
    VOCAB_SIZE = 64
    D_MODEL = 16
    D_INNER = 16
    D_STATE = 8
    D_CONV = 4
    
    np.random.seed(42)
    
    def rand_tensor(shape):
        return (np.random.randn(*shape) * 0.1).astype(np.float32)
        
    weights = {
        "emb_w": rand_tensor((VOCAB_SIZE, D_MODEL)),
        "in_w": rand_tensor((2 * D_INNER, D_MODEL)),
        "in_b": rand_tensor((2 * D_INNER,)),
        "x_proj_w": rand_tensor((D_INNER + 2 * D_STATE, D_INNER)),
        "x_proj_b": rand_tensor((D_INNER + 2 * D_STATE,)),
        "out_w": rand_tensor((D_MODEL, D_INNER)),
        "out_b": rand_tensor((D_MODEL,)),
        "conv_w": rand_tensor((D_INNER, D_CONV)),
        "conv_b": rand_tensor((D_INNER,)),
        "A": -np.abs(rand_tensor((D_INNER, D_STATE))), # A must be negative for stability
        "dt_b": rand_tensor((D_INNER,)),
        "lm_head_w": rand_tensor((VOCAB_SIZE, D_MODEL)),
        "lm_head_b": rand_tensor((VOCAB_SIZE,))
    }
    
    return weights

def export_to_header(weights, filename):
    with open(filename, 'w') as f:
        f.write("#ifndef UMAMBA_WEIGHTS_H\n")
        f.write("#define UMAMBA_WEIGHTS_H\n\n")
        f.write("#include \"../include/config.h\"\n")
        f.write("#ifdef ARDUINO\n")
        f.write("#include <avr/pgmspace.h>\n")
        f.write("#else\n")
        f.write("#define PROGMEM\n")
        f.write("#endif\n\n")
        f.write("namespace umamba {\n")
        f.write("namespace weights {\n\n")
        
        for name, tensor in weights.items():
            flat = tensor.flatten()
            f.write(f"    const tensor_t {name}[{len(flat)}] PROGMEM = {{\n        ")
            f.write(", ".join([f"{x:.6f}f" for x in flat]))
            f.write("\n    };\n\n")
            
        f.write("} // namespace weights\n")
        f.write("} // namespace umamba\n\n")
        f.write("#endif // UMAMBA_WEIGHTS_H\n")

if __name__ == "__main__":
    w = generate_weights()
    export_to_header(w, "../src/weights.h")
    print("Weights exported to src/weights.h")

