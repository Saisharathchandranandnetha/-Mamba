#include "include/config.h"
#include "src/memory.h"
#include "src/tokenizer.h"
#include "src/model.h"
#include "src/sampler.h"
#include "src/benchmark.h"
#include "src/weights.h"

umamba::MicroMambaModel model;

void setup() {
    Serial.begin(115200);
    while (!Serial) { ; }
    
    Serial.println();
    Serial.println(F("============================="));
    Serial.println(F("  \xC2\xB5Mamba System Ready"));
    Serial.println(F("============================="));
    
    umamba::tokenizer::init();
    umamba::sampler::seed(42);
    
    using namespace umamba::weights;
    model.set_weights(
        emb_w, in_w, in_b, x_proj_w, x_proj_b,
        out_w, out_b, conv_w, conv_b,
        A, dt_b, lm_head_w, lm_head_b
    );
    model.reset_state();
    
    umamba::memory::print_report();
    Serial.println(F("Commands: prompt <text>, memory, benchmark, reset, help"));
    Serial.print(F("> "));
}

void parse_command(String cmd) {
    cmd.trim();
    if (cmd.length() == 0) return;
    
    if (cmd.startsWith("prompt ")) {
        String prompt_str = cmd.substring(7);
        Serial.print(F("PROMPT: ")); Serial.println(prompt_str);
        
        int tokens[64];
        size_t n = umamba::tokenizer::encode(prompt_str.c_str(), tokens, 64);
        
        Serial.print(F("TOKENS: "));
        for (size_t i = 0; i < n; ++i) {
            Serial.print(tokens[i]); Serial.print(F(" "));
        }
        Serial.println();
        
        Serial.println(F("GENERATING..."));
        
        umamba::tensor_t logits[umamba::VOCAB_SIZE];
        
        // Process prompt
        for (size_t i = 0; i < n - 1; ++i) {
            model.forward(tokens[i], logits);
        }
        
        int curr_token = tokens[n - 1];
        String output_text = prompt_str;
        
        // Generate tokens
        for (int i = 0; i < 20; ++i) {
            model.forward(curr_token, logits);
            int next_token = umamba::sampler::argmax(logits, umamba::VOCAB_SIZE);
            char c = umamba::tokenizer::decode(next_token);
            
            output_text += c;
            Serial.print(F("TOKEN ID: ")); Serial.print(next_token);
            Serial.print(F(" CHAR: '")); Serial.print(c); Serial.println(F("'"));
            
            curr_token = next_token;
            if (c == '\n') break;
        }
        
        Serial.print(F("\nOUTPUT: "));
        Serial.println(output_text);
        Serial.println();
    }
    else if (cmd == "memory") {
        umamba::memory::print_report();
    }
    else if (cmd == "benchmark") {
        umamba::benchmark::run_full_benchmark();
    }
    else if (cmd == "reset") {
        model.reset_state();
        Serial.println(F("Model state reset."));
    }
    else if (cmd == "help") {
        Serial.println(F("Commands:"));
        Serial.println(F("  prompt <text> - generate text"));
        Serial.println(F("  memory        - print memory report"));
        Serial.println(F("  benchmark     - run speed benchmark"));
        Serial.println(F("  reset         - clear SSM states"));
    }
    else {
        Serial.println(F("Unknown command. Type 'help'."));
    }
    Serial.print(F("> "));
}

void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        parse_command(cmd);
    }
}

