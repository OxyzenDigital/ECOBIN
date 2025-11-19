// --- DM420Y Stepper Motor Control (Pulse/Direction) ---
// Includes LIVE vs TEST mode toggle for Arduino IDE Serial testing.

// ------------------------------------------------------
//                   MODE SELECTOR
// ------------------------------------------------------
// true  = TEST MODE (You type R/T/X in Arduino Serial Monitor)
// false = LIVE MODE (Arduino ONLY accepts commands from external program)
const bool TEST_MODE = true;
// ------------------------------------------------------


// --- HARDWARE CONFIGURATION ---
const int STEP_PIN = 9;   // PU+
const int DIR_PIN  = 8;   // DR+
const int MF_PIN   = 10;  // MF (Enable)

// Step parameters (adjust for your microstepping)
const int STEPS_FOR_MOVEMENT = 3500;
const int PULSE_DELAY_US = 250;

// State machine
enum SystemState { READY, BUSY };
SystemState currentState = READY;

// Serial Commands
const char READY_SIGNAL = 'A';
const char BUSY_SIGNAL  = 'B';
const char RECYCLE_COMMAND = 'R';
const char TRASH_COMMAND   = 'T';
const char RESET_COMMAND   = 'X';


// ------------------------------------------------------
// SETUP
// ------------------------------------------------------
void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(MF_PIN, OUTPUT);

  Serial.begin(9600);
  Serial.println();
  Serial.println("-------------------------------------------");
  Serial.println(" DM420Y Stepper Control Initialized");
  Serial.println("-------------------------------------------");

  if (TEST_MODE) {
    Serial.println(" MODE: TEST MODE (type R, T, X in Serial Monitor)");
  } else {
    Serial.println(" MODE: LIVE MODE (Serial Monitor input ignored)");
  }
  Serial.println("-------------------------------------------");

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  // Enable driver
  digitalWrite(MF_PIN, LOW);
  Serial.println("DM420Y Driver Enabled (MF LOW).");

  sendReadySignal();
}


// ------------------------------------------------------
// LOOP
// ------------------------------------------------------
void loop() {

  // ------------------------------------------------------
  // LIVE MODE
  // ------------------------------------------------------
  if (!TEST_MODE) {
    if (Serial.available() > 0) {
      // Human typing is NOT allowed in LIVE mode
      Serial.println("LIVE MODE: Manual typing ignored.");
      Serial.read();  // clear buffer
    }
    return;  // external system still sends Serial commands normally
  }


  // ------------------------------------------------------
  // TEST MODE (Arduino IDE manual control)
  // ------------------------------------------------------
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == RECYCLE_COMMAND && currentState == READY) {
      Serial.print("Command received: R (Recycle). ");
      runSortSequence(HIGH);
    }
    else if (command == TRASH_COMMAND && currentState == READY) {
      Serial.print("Command received: T (Trash). ");
      runSortSequence(LOW);
    }
    else if (command == RESET_COMMAND) {
      currentState = READY;
      Serial.println("System reset manually (X).");
      sendReadySignal();
    }
    else if (currentState == BUSY) {
      Serial.print("System BUSY, rejecting: ");
      Serial.println(command);
    }
  }
}


// ------------------------------------------------------
// HELPER FUNCTIONS
// ------------------------------------------------------
void sendReadySignal() {
  currentState = READY;
  Serial.write(READY_SIGNAL);
  Serial.println("System READY (A).");
}

void sendBusySignal() {
  currentState = BUSY;
  Serial.write(BUSY_SIGNAL);
  Serial.println("System BUSY (B).");
}

void makeSteps(int steps) {
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(PULSE_DELAY_US);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(PULSE_DELAY_US);
  }
}

void runSortSequence(int dirState) {
  sendBusySignal();

  digitalWrite(DIR_PIN, dirState);
  makeSteps(STEPS_FOR_MOVEMENT);

  Serial.println("Holding for 2 seconds...");
  delay(2000);

  digitalWrite(DIR_PIN, (dirState == HIGH ? LOW : HIGH));
  makeSteps(STEPS_FOR_MOVEMENT);

  Serial.println("Sequence complete.");
  sendReadySignal();
}
