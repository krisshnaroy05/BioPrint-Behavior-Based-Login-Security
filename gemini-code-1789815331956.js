class BotDetector {
  constructor() {
    this.mousePositions = []; // [{x, y, t}]
    this.keyEvents = [];      // [{key, type, t, isTrusted}]
    this.botScore = 0;
    this.reasons = [];

    this.initListeners();
  }

  initListeners() {
    // 1. Capture Mouse Dynamics
    window.addEventListener('mousemove', (e) => {
      this.mousePositions.push({
        x: e.clientX,
        y: e.clientY,
        t: performance.now(),
        isTrusted: e.isTrusted
      });
    });

    // 2. Capture Keystroke Dynamics
    const formInputs = document.querySelectorAll('input');
    formInputs.forEach(input => {
      input.addEventListener('keydown', (e) => this.recordKey(e));
      input.addEventListener('keyup', (e) => this.recordKey(e));
    });
  }

  recordKey(e) {
    this.keyEvents.push({
      key: e.key,
      type: e.type,
      t: performance.now(),
      isTrusted: e.isTrusted
    });
  }

  // Evaluate raw signals before submitting form
  evaluateBotSignals() {
    this.reasons = [];
    let isBot = false;

    // --- CHECK 1: Browser Environment Fingerprinting ---
    if (navigator.webdriver) {
      isBot = true;
      this.reasons.push("Automated browser flag detected (navigator.webdriver)");
    }

    // --- CHECK 2: Event Integrity (Synthetic Event Injection) ---
    const untrustedKey = this.keyEvents.some(k => k.isTrusted === false);
    const untrustedMouse = this.mousePositions.some(m => m.isTrusted === false);
    if (untrustedKey || untrustedMouse) {
      isBot = true;
      this.reasons.push("Synthetic events detected (isTrusted === false)");
    }

    // --- CHECK 3: Keystroke Velocity & Zero-Variance Check ---
    const dwellTimes = [];
    const keyDowns = {};

    this.keyEvents.forEach(evt => {
      if (evt.type === 'keydown') {
        keyDowns[evt.key] = evt.t;
      } else if (evt.type === 'keyup' && keyDowns[evt.key]) {
        dwellTimes.push(evt.t - keyDowns[evt.key]);
        delete keyDowns[evt.key];
      }
    });

    if (dwellTimes.length > 2) {
      // Calculate standard deviation of dwell time
      const mean = dwellTimes.reduce((a, b) => a + b) / dwellTimes.length;
      const variance = dwellTimes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / dwellTimes.length;
      const stdDev = Math.sqrt(variance);

      // Bot signal: Typing speed is unrealistically fast (<10ms hold) or robotic variance (<2ms)
      if (mean < 10) {
        isBot = true;
        this.reasons.push(`Superhuman typing speed (Mean Dwell: ${mean.toFixed(1)}ms)`);
      }
      if (stdDev < 2.0) {
        isBot = true;
        this.reasons.push(`Robotic typing timing (StdDev: ${stdDev.toFixed(2)}ms)`);
      }
    }

    // --- CHECK 4: Mouse Path Straightness (Curvature Check) ---
    if (this.mousePositions.length > 5) {
      let straightLineSteps = 0;
      for (let i = 2; i < this.mousePositions.length; i++) {
        const p1 = this.mousePositions[i - 2];
        const p2 = this.mousePositions[i - 1];
        const p3 = this.mousePositions[i];

        // Cross product to find collinearity (straight line)
        const crossProduct = Math.abs((p2.y - p1.y) * (p3.x - p2.x) - (p2.x - p1.x) * (p3.y - p2.y));
        if (crossProduct === 0) straightLineSteps++;
      }

      const linearRatio = straightLineSteps / (this.mousePositions.length - 2);
      if (linearRatio > 0.9) { // 90%+ points on an exact mathematical line
        isBot = true;
        this.reasons.push(`Unnatural mouse straightness (${(linearRatio * 100).toFixed(0)}% linear trajectory)`);
      }
    } else {
      // Zero mouse movement recorded on input form submit
      isBot = true;
      this.reasons.push("No mouse telemetry detected prior to form submission");
    }

    return {
      isBot,
      reasons: this.reasons,
      telemetryPayload: {
        dwellTimes,
        mousePositions: this.mousePositions
      }
    };
  }
}

// Global instance
const botDetector = new BotDetector();