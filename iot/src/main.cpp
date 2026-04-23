/*
  smart_belt_fixed.ino
  =====================
  Smart Belt Fall Detection — ESP32 Firmware
  Alliance University Capstone Project

  Output format (CSV at 200Hz, 115200 baud):
    acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,pressure_left,pressure_right

  Pin mapping:
    MPU6050 → SDA: GPIO 21 | SCL: GPIO 22
    Pressure Left  → GPIO 34
    Pressure Right → GPIO 32  (FIXED from teammate's GPIO 35)
    Buzzer         → GPIO 26

  Units:
    Acceleration : raw int16 (divide by 16384.0 for g)
    Gyroscope    : raw int16 (divide by 131.0 for deg/s)
    Pressure     : ADC raw 0–4095
*/

#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// ── Pin Definitions ──────────────────────────────────────────────
const int PRESSURE_LEFT_PIN  = 34;   // matches config.py
const int PRESSURE_RIGHT_PIN = 32;   // FIXED: was 35 in teammate's version
const int BUZZER_PIN         = 26;

// ── IMU variables ─────────────────────────────────────────────────
int16_t ax, ay, az;
int16_t gx, gy, gz;

// ── Timing for 200Hz output ───────────────────────────────────────
// 200Hz = 1 sample every 5ms
const unsigned long SAMPLE_INTERVAL_US = 5000;  // 5000 microseconds = 5ms
unsigned long lastSampleTime = 0;

// ── Buzzer state ──────────────────────────────────────────────────
bool buzzerActive      = false;
unsigned long buzzerStartTime = 0;
const unsigned long BUZZER_DURATION = 3000;  // 3 seconds

// ── Simple fall detection for onboard buzzer alert ────────────────
// (Python pipeline does the real detection — this is just a local backup)
float prevAccelMag = 0.0;
const float FALL_JERK_THRESHOLD = 8.0;  // g/s

void setup() {
  Serial.begin(115200);

  pinMode(PRESSURE_LEFT_PIN,  INPUT);
  pinMode(PRESSURE_RIGHT_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(21, 22);          // SDA=GPIO21, SCL=GPIO22
  Wire.setClock(400000);       // 400kHz I2C

  mpu.initialize();
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);   // ±2g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);   // ±250 deg/s
  mpu.setDLPFMode(MPU6050_DLPF_BW_42);              // 42Hz LPF

  if (!mpu.testConnection()) {
    // Flash buzzer twice to indicate MPU6050 failure
    for (int i = 0; i < 2; i++) {
      digitalWrite(BUZZER_PIN, HIGH); delay(300);
      digitalWrite(BUZZER_PIN, LOW);  delay(300);
    }
  }

  // Short ready beep
  digitalWrite(BUZZER_PIN, HIGH); delay(100);
  digitalWrite(BUZZER_PIN, LOW);

  lastSampleTime = micros();
}

void loop() {
  unsigned long now = micros();

  // ── Maintain 200Hz sample rate ────────────────────────────────
  if (now - lastSampleTime < SAMPLE_INTERVAL_US) return;
  lastSampleTime = now;

  // ── Read MPU6050 ─────────────────────────────────────────────
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // ── Read pressure sensors (ADC 0–4095) ───────────────────────
  int pressureLeft  = analogRead(PRESSURE_LEFT_PIN);
  int pressureRight = analogRead(PRESSURE_RIGHT_PIN);

  // ── Output CSV line (parsed by iot_receiver.py) ──────────────
  // Format: acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z,pressure_left,pressure_right
  Serial.print(ax);           Serial.print(',');
  Serial.print(ay);           Serial.print(',');
  Serial.print(az);           Serial.print(',');
  Serial.print(gx);           Serial.print(',');
  Serial.print(gy);           Serial.print(',');
  Serial.print(gz);           Serial.print(',');
  Serial.print(pressureLeft); Serial.print(',');
  Serial.println(pressureRight);

  // ── Onboard fall detection for local buzzer ───────────────────
  // Python pipeline does the real ML detection.
  // This is just a simple jerk-threshold backup alert.
  float accelMag = sqrt((float)ax*ax + (float)ay*ay + (float)az*az) / 16384.0;
  float jerk     = abs(accelMag - prevAccelMag) / (SAMPLE_INTERVAL_US / 1e6);
  prevAccelMag   = accelMag;

  if (jerk > FALL_JERK_THRESHOLD && !buzzerActive) {
    buzzerActive    = true;
    buzzerStartTime = millis();
    digitalWrite(BUZZER_PIN, HIGH);
  }

  // ── Buzzer timeout ────────────────────────────────────────────
  if (buzzerActive && (millis() - buzzerStartTime >= BUZZER_DURATION)) {
    digitalWrite(BUZZER_PIN, LOW);
    buzzerActive = false;
  }
}
