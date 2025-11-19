// --- DM420Y Stepper Motor Control (Pulse/Direction) ---
// Corrected for simplified DM420Y terminal block (PU, DR, 5V+, MF)
// This code manages the READY/BUSY state via Serial communication.

// --- HARDWARE CONFIGURATION ---
const int STEP_PIN = 9;   // Connect to DM420Y's PU (Pulse)
const int DIR_PIN = 8;    // Connect to DM420Y's DR (Direction)
const int MF_PIN = 10;    // Connect to DM420Y's MF (Motor Free/Enable)

// Motor parameters (NEMA 17, 200 steps/rev)
// *** UPDATED: Now targeting 90 degrees. If driver is set to 1/10 microstepping (2000 steps/rev), 
// 90 degrees requires 500 steps (2000 steps / 360 deg * 90 deg = 500 steps). ***
// Adjust this number if your microstepping DIP switches are different!
const int STEPS_FOR_MOVEMENT = 3500; 

// *** UPDATED: Increased speed 10x by reducing the delay. ***
// Speed adjustment: Reduce this value to go faster. If motor stalls, increase it slightly.
const int PULSE_DELAY_US = 250; 

// --- STATE AND CONTROL VARIABLES ---
enum SystemState { READY, BUSY };
SystemState currentState = READY;

// Serial Communication Signals
const char READY_SIGNAL = 'A'; 
const char BUSY_SIGNAL  = 'B'; 
const char RECYCLE_COMMAND = 'R'; // Clockwise
const char TRASH_COMMAND   = 'T'; // Counter-Clockwise
const char RESET_COMMAND   = 'X'; 

// --- SETUP ---
void setup() {
  // Set the control pins as outputs
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(MF_PIN, OUTPUT);
  
  // Initialize Serial communication
  Serial.begin(9600);
  Serial.println("--- DM420Y Stepper Control Initialized (PUL/DIR) ---");
  Serial.println("CURRENT ANGLE: 90 degrees (500 steps). CURRENT SPEED: 500 us delay.");
  Serial.println("CHECK DIP SWITCHES: Ensure current (SW1-SW3) is set for 2.0A!");
  Serial.println("CHECK DIP SWITCHES: Ensure microstepping (SW5-SW8) matches the STEPS_FOR_MOVEMENT value!");
  
  // 1. Ensure control pins start low
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  // 2. ENABLE THE DRIVER: MF pin must be LOW to keep the motor coils powered/enabled.
  digitalWrite(MF_PIN, LOW);
  Serial.println("DM420Y Driver Enabled (MF pin LOW).");

  sendReadySignal();
}

// --- MAIN LOOP ---
void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == RECYCLE_COMMAND && currentState == READY) {
      Serial.print("Command received: R (Recycle). ");
      runSortSequence(HIGH); // HIGH for Clockwise (Recycle)
    } else if (command == TRASH_COMMAND && currentState == READY) {
      Serial.print("Command received: T (Trash). ");
      runSortSequence(LOW); // LOW for Counter-Clockwise (Trash)
    } else if (command == RESET_COMMAND) {
      currentState = READY;
      Serial.println("System reset manually (X command).");
      sendReadySignal();
    } else if (currentState == BUSY) {
        Serial.print("System is BUSY, rejecting command: ");
        Serial.println(command);
    } 
  } 
}

// --- HELPER FUNCTIONS ---

void sendReadySignal() {
  currentState = READY;
  Serial.write(READY_SIGNAL);
  Serial.println("System is READY (A).");
}

void sendBusySignal() {
  currentState = BUSY;
  Serial.write(BUSY_SIGNAL);
  Serial.println("Starting sequence... System is BUSY (B).");
}

/**
 * Generates the necessary step pulses for the DM420Y driver.
 */
void makeSteps(int steps) {
  for (int i = 0; i < steps; i++) {
    // 1. Send HIGH pulse
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(PULSE_DELAY_US);
    
    // 2. Send LOW pulse (The falling edge (HIGH to LOW) triggers the step)
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(PULSE_DELAY_US);
  }
}

/**
 * Executes the full sorting sequence (turn, wait, return).
 */
void runSortSequence(int dirState) {
  // 1. Signal BUSY
  sendBusySignal();

  // 2. Set the direction for the first movement
  digitalWrite(DIR_PIN, dirState);
  
  Serial.print("Moving 90 degrees (");
  Serial.print(STEPS_FOR_MOVEMENT);
  Serial.print(" steps) in direction: ");
  Serial.println((dirState == HIGH ? "Clockwise" : "Counter-Clockwise"));
  
  // Execute the movement
  makeSteps(STEPS_FOR_MOVEMENT);

  // 3. Wait for 5 seconds (The holding/dump time)
  Serial.println("Holding position for 2 seconds...");
  delay(2000);

  // 4. Turn back to the original position
  // Reverse the direction state for the return trip
  int returnDirState = (dirState == HIGH) ? LOW : HIGH;
  digitalWrite(DIR_PIN, returnDirState);
  
  Serial.println("Returning to home position...");
  
  // Execute the return movement (same number of steps)
  makeSteps(STEPS_FOR_MOVEMENT);
  
  Serial.println("Sequence complete.");

  // 5. Signal READY
  sendReadySignal();
}