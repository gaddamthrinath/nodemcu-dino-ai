// ================================================================
// Dino AI - High-Speed NodeMCU (ESP8266) Firmware
// Zero-latency non-blocking serial communication + Neural Network
// ================================================================

#include "model.h"

#define LED_PIN 2

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5); // Ultra-fast timeout
  while (!Serial) {
    ;
  }

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH); // Turn LED OFF initially

  // Startup blink indicator
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, LOW);
    delay(60);
    digitalWrite(LED_PIN, HIGH);
    delay(60);
  }
}

float raw[N_RAW];
char rxBuf[128];
int rxPos = 0;

void processPacket(char *line) {
  // Handle ping
  if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
    return;
  }

  // Fast parsing without String allocations
  int parsed = 0;
  char *token = strtok(line, ",");
  while (token != NULL && parsed < N_RAW) {
    raw[parsed++] = atof(token);
    token = strtok(NULL, ",");
  }

  if (parsed == N_RAW) {
    // Run the neural network in model.h (takes < 0.1ms on 80MHz ESP8266)
    int action = dino_predict(raw); // 0 = RUN, 1 = JUMP, 2 = DUCK

    // Turn LED ON when jumping/ducking
    digitalWrite(LED_PIN, (action != 0) ? LOW : HIGH);

    // Send action back immediately
    Serial.println(action);
  }
}

void loop() {
  // Non-blocking serial reader for ultra-low latency
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxPos > 0) {
        rxBuf[rxPos] = '\0';
        processPacket(rxBuf);
        rxPos = 0;
      }
    } else if (rxPos < (int)sizeof(rxBuf) - 1) {
      rxBuf[rxPos++] = c;
    }
  }
}
