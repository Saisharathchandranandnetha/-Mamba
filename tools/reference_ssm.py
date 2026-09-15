import numpy as np
import sys

def main():
    np.random.seed(42)
    
    D_INNER = 16
    D_STATE = 8
    SEQ_LEN = 3
    
    # Init random weights
    A = np.random.randn(D_INNER, D_STATE).astype(np.float32) * 0.1
    B = np.random.randn(D_INNER, D_STATE).astype(np.float32) * 0.1
    C = np.random.randn(D_INNER, D_STATE).astype(np.float32) * 0.1
    D = np.random.randn(D_INNER).astype(np.float32) * 0.1
    
    # Init random input
    X = np.random.randn(SEQ_LEN, D_INNER).astype(np.float32)
    
    # State
    h = np.zeros((D_INNER, D_STATE), dtype=np.float32)
    
    print("WEIGHTS")
    print("A:", A[0, :3])
    print("B:", B[0, :3])
    print("C:", C[0, :3])
    print("D:", D[0])
    
    print("\nRESULTS")
    for t in range(SEQ_LEN):
        x_t = X[t]
        y_t = np.zeros(D_INNER, dtype=np.float32)
        
        for d in range(D_INNER):
            h[d] = A[d] * h[d] + B[d] * x_t[d]
            y_t[d] = np.dot(C[d], h[d]) + D[d] * x_t[d]
            
        print(f"Step {t} y[0]: {y_t[0]:.6f} h[0,0]: {h[0,0]:.6f}")

    # Output arrays for C++ test
    print("\nC++ TEST DATA")
    
    def print_c_array(name, arr):
        flat = arr.flatten()
        s = f"float {name}[] = {{"
        s += ", ".join([f"{x:.6f}f" for x in flat])
        s += "};"
        print(s)
        
    print_c_array("A", A)
    print_c_array("B", B)
    print_c_array("C", C)
    print_c_array("D", D)
    print_c_array("X", X)

if __name__ == "__main__":
    main()

