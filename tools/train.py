import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Exact dimensions from config.h
VOCAB_SIZE = 64
D_MODEL = 16
D_INNER = 16
D_STATE = 8
D_CONV = 4

# Initialize tokenizer (exact same mapping as C++)
char_to_id = {chr(i): 0 for i in range(256)}
id_to_char = [' ']*64

char_to_id[' '] = 0; id_to_char[0] = ' '
for i in range(26):
    c = chr(ord('a') + i); id_to_char[1+i] = c; char_to_id[c] = 1+i
    c2 = chr(ord('A') + i); id_to_char[27+i] = c2; char_to_id[c2] = 27+i
for i in range(10):
    c = chr(ord('0') + i); id_to_char[53+i] = c; char_to_id[c] = 53+i
char_to_id['\n'] = 63; id_to_char[63] = '\n'

def encode(text):
    return [char_to_id.get(c, 0) for c in text]

# PyTorch implementation of our exact C++ Architecture
class TinyMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.in_proj = nn.Linear(D_MODEL, 2 * D_INNER)
        self.conv1d = nn.Conv1d(D_INNER, D_INNER, kernel_size=D_CONV, padding=D_CONV-1, groups=D_INNER)
        self.x_proj = nn.Linear(D_INNER, D_INNER + 2 * D_STATE)
        self.dt_b = nn.Parameter(torch.zeros(D_INNER))
        self.A = nn.Parameter(torch.randn(D_INNER, D_STATE) * 0.1 - 0.5) 
        self.out_proj = nn.Linear(D_INNER, D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)
        
    def forward(self, x_ids):
        x = self.emb(x_ids)
        xz = self.in_proj(x)
        x_inner, z_inner = xz[..., :D_INNER], xz[..., D_INNER:]
        
        # Conv1d
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., :x_inner.size(1)]
        x_inner = F.silu(x_conv.transpose(1, 2))
        
        # Projections
        dt_B_C = self.x_proj(x_inner)
        dt = dt_B_C[..., :D_INNER]
        B = dt_B_C[..., D_INNER : D_INNER + D_STATE]
        C = dt_B_C[..., D_INNER + D_STATE:]
        
        dt = F.softplus(dt + self.dt_b)
        
        # SSM Core
        batch, seq_len, _ = x_inner.shape
        y = torch.zeros(batch, seq_len, D_INNER, device=x.device)
        
        for b in range(batch):
            h = torch.zeros(D_INNER, D_STATE, device=x.device)
            for t in range(seq_len):
                A_bar = torch.exp(dt[b, t].unsqueeze(1) * self.A)
                B_bar = dt[b, t].unsqueeze(1) * B[b, t].unsqueeze(0)
                h = A_bar * h + B_bar * x_inner[b, t].unsqueeze(1)
                y[b, t] = torch.sum(C[b, t].unsqueeze(0) * h, dim=1)
                
        # Gating & Output
        y = y * F.silu(z_inner)
        return self.lm_head(self.out_proj(y))

def export_weights(model, filename):
    w = {}
    sd = model.state_dict()
    w['emb_w'] = sd['emb.weight'].numpy()
    w['in_w'] = sd['in_proj.weight'].numpy()
    w['in_b'] = sd['in_proj.bias'].numpy()
    w['x_proj_w'] = sd['x_proj.weight'].numpy()
    w['x_proj_b'] = sd['x_proj.bias'].numpy()
    w['out_w'] = sd['out_proj.weight'].numpy()
    w['out_b'] = sd['out_proj.bias'].numpy()
    # Conv weights shape is (D_INNER, 1, D_CONV). Squeeze to (D_INNER, D_CONV). Reverse kernel to match C++ shift register!
    w['conv_w'] = np.flip(sd['conv1d.weight'].squeeze(1).numpy(), axis=1)
    w['conv_b'] = sd['conv1d.bias'].numpy()
    w['A'] = sd['A'].numpy()
    w['dt_b'] = sd['dt_b'].numpy()
    w['lm_head_w'] = sd['lm_head.weight'].numpy()
    w['lm_head_b'] = sd['lm_head.bias'].numpy()
    
    with open(filename, 'w') as f:
        f.write("#ifndef UMAMBA_WEIGHTS_H\n#define UMAMBA_WEIGHTS_H\n\n")
        f.write("#include \"../include/config.h\"\n#ifdef ARDUINO\n#include <avr/pgmspace.h>\n#else\n#define PROGMEM\n#endif\n\n")
        f.write("namespace umamba {\nnamespace weights {\n\n")
        for name, tensor in w.items():
            flat = tensor.flatten()
            f.write(f"    const tensor_t {name}[{len(flat)}] PROGMEM = {{\n        ")
            f.write(", ".join([f"{x:.6f}f" for x in flat]))
            f.write("\n    };\n\n")
        f.write("} // namespace weights\n} // namespace umamba\n\n#endif // UMAMBA_WEIGHTS_H\n")

print("Initializing TinyMamba in PyTorch...")
model = TinyMamba()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# Training Dataset: Overfit on a specific response
question = "what can you do for me?"
answer = " i am umamba a tiny AI on arduino\n"
text = question + answer
input_ids = torch.tensor([encode(text[:-1])])
target_ids = torch.tensor([encode(text[1:])])

print(f"Training on string: '{text}'")
print("Training for 200 epochs (this will take 5 seconds)...")

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("Exporting trained weights to C++...")
export_weights(model, "src/weights.h")
print("Done! The Arduino now has a brain.")
