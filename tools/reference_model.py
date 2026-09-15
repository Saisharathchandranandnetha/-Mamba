import numpy as np
from export_weights import generate_weights

# Activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def silu(x):
    return x * sigmoid(x)

def softplus(x):
    return np.log(1.0 + np.exp(np.clip(x, -20, 20)))

def init_tokenizer():
    id_to_char = [' ']*64
    char_to_id = {chr(i): 0 for i in range(256)}
    
    id_to_char[0] = ' '
    char_to_id[' '] = 0
    
    for i in range(26):
        c = chr(ord('a') + i)
        id_to_char[1+i] = c
        char_to_id[c] = 1+i
        
        c2 = chr(ord('A') + i)
        id_to_char[27+i] = c2
        char_to_id[c2] = 27+i
        
    for i in range(10):
        c = chr(ord('0') + i)
        id_to_char[53+i] = c
        char_to_id[c] = 53+i
        
    id_to_char[63] = '\n'
    char_to_id['\n'] = 63
    
    return char_to_id, id_to_char

class PythonMicroMamba:
    def __init__(self, weights):
        self.w = weights
        self.D_INNER = 16
        self.D_STATE = 8
        self.D_CONV = 4
        self.conv_state = np.zeros((self.D_INNER, self.D_CONV), dtype=np.float32)
        self.ssm_state = np.zeros((self.D_INNER, self.D_STATE), dtype=np.float32)
        self.char_to_id, self.id_to_char = init_tokenizer()

    def reset(self):
        self.conv_state.fill(0)
        self.ssm_state.fill(0)

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, token_id):
        if 0 <= token_id < 64:
            return self.id_to_char[token_id]
        return ' '

    def step(self, token_id):
        # 1. Embedding
        x = self.w['emb_w'][token_id]
        
        # 2. Mamba Block
        # in_proj
        xz = x @ self.w['in_w'].T + self.w['in_b']
        x_inner = xz[:self.D_INNER]
        z_inner = xz[self.D_INNER:]
        
        # Conv1D
        self.conv_state = np.roll(self.conv_state, 1, axis=1)
        self.conv_state[:, 0] = x_inner
        
        # Apply conv
        conv_out = np.zeros(self.D_INNER, dtype=np.float32)
        for d in range(self.D_INNER):
            conv_out[d] = np.sum(self.conv_state[d] * self.w['conv_w'][d]) + self.w['conv_b'][d]
        
        x_inner = silu(conv_out)
        
        # x_proj
        dt_B_C = x_inner @ self.w['x_proj_w'].T + self.w['x_proj_b']
        dt = dt_B_C[:self.D_INNER]
        B = dt_B_C[self.D_INNER : self.D_INNER + self.D_STATE]
        C = dt_B_C[self.D_INNER + self.D_STATE:]
        
        # softplus
        dt = softplus(dt + self.w['dt_b'])
        
        # SSM
        y = np.zeros(self.D_INNER, dtype=np.float32)
        for d in range(self.D_INNER):
            A_bar = np.exp(dt[d] * self.w['A'][d])
            B_bar = dt[d] * B
            self.ssm_state[d] = A_bar * self.ssm_state[d] + B_bar * x_inner[d]
            y[d] = np.dot(C, self.ssm_state[d])
            
        # Gating
        y = y * silu(z_inner)
        
        # out_proj
        out = y @ self.w['out_w'].T + self.w['out_b']
        
        # 3. LM Head
        logits = out @ self.w['lm_head_w'].T + self.w['lm_head_b']
        
        return logits

if __name__ == "__main__":
    w = generate_weights()
    model = PythonMicroMamba(w)
    
    prompt = "hello"
    print(f"PROMPT: {prompt}")
    
    tokens = model.encode(prompt)
    print(f"TOKENS: {tokens}")
    
    model.reset()
    
    # Process prompt
    for t in tokens[:-1]:
        model.step(t)
        
    # Generate
    curr_token = tokens[-1]
    out_text = prompt
    print("GENERATING...")
    
    for _ in range(5):
        logits = model.step(curr_token)
        # Argmax
        next_token = np.argmax(logits)
        char = model.decode(next_token)
        out_text += char
        print(f"TOKEN ID: {next_token} CHAR: '{char}'")
        curr_token = next_token
        
    print(f"FINAL: {out_text}")
    
    # Output specifically for C++ reference test
    model.reset()
    logits_0 = model.step(tokens[0])
    print(f"LOGITS_0[0:5]: {logits_0[:5]}")

