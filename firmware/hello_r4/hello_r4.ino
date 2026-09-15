/*
 * µMamba — Hello R4
 * 
 * Minimal smoke-test sketch for Arduino UNO R4 Minima.
 * Prints boot message and reports SRAM/flash availability.
 * 
 * Board: Arduino UNO R4 Minima (Renesas RA4M1, Cortex-M4)
 * FQBN:  arduino:renesas_uno:minima
 */

// ── Heap / stack free-RAM estimator for ARM Cortex-M ──
extern "C" char *sbrk(int incr);

static int freeRAM() {
    char top;
    return &top - reinterpret_cast<char*>(sbrk(0));
}

void setup() {
    Serial.begin(115200);
    while (!Serial) { ; } // wait for native USB CDC

    Serial.println();
    Serial.println(F("============================="));
    Serial.println(F("  \xC2\xB5Mamba booted"));            // µ = UTF-8 C2 B5
    Serial.println(F("============================="));
    Serial.println();

    // ── Hardware info ──
    Serial.println(F("Board   : Arduino UNO R4 Minima"));
    Serial.println(F("MCU     : Renesas RA4M1 (Cortex-M4)"));
    Serial.println(F("Flash   : 256 KB"));
    Serial.println(F("SRAM    : 32 KB"));
    Serial.println();

    // ── Memory report ──
    Serial.print(F("Free SRAM (estimated): "));
    Serial.print(freeRAM());
    Serial.println(F(" bytes"));
    Serial.println();

    // ── Timing baseline ──
    unsigned long t0 = micros();
    volatile float acc = 0.0f;
    for (int i = 0; i < 1000; i++) {
        acc += (float)i * 0.001f;
    }
    unsigned long t1 = micros();

    Serial.print(F("1000 float MADs     : "));
    Serial.print(t1 - t0);
    Serial.println(F(" us"));

    // Quick matmul benchmark: 16x16 * 16x16
    float A[16][16], B[16][16], C[16][16];
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            A[i][j] = (float)(i + j) * 0.01f;
            B[i][j] = (float)(i - j) * 0.01f;
            C[i][j] = 0.0f;
        }

    t0 = micros();
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            for (int k = 0; k < 16; k++)
                C[i][j] += A[i][k] * B[k][j];
    t1 = micros();

    Serial.print(F("16x16 matmul        : "));
    Serial.print(t1 - t0);
    Serial.println(F(" us"));

    Serial.print(F("Free SRAM after test: "));
    Serial.print(freeRAM());
    Serial.println(F(" bytes"));

    Serial.println();
    Serial.println(F("Phase 0 complete — ready for \xC2\xB5Mamba."));
}

void loop() {
    // nothing — one-shot diagnostic
}

