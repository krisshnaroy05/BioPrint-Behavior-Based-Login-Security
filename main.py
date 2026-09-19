import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    raise ImportError("scikit-learn is required. Install via `pip install scikit-learn`")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="BioPrint Unified Telemetry Engine & Authentication Server")

# Enable CORS for local/extension development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory user database for imposter detection & biometric baselines
USER_DATABASE: Dict[str, Dict[str, Any]] = {}


class BioPrintBotDetector:
    """
    Evaluates telemetry payloads sent by web GUIs, scripts, or automated bots.
    Combines rule-based heuristics with an unsupervised anomaly detection model.
    """
    def __init__(self):
        # Heuristic Thresholds
        self.MIN_MOUSE_SAMPLES = 5          # Minimum recorded mouse points for non-touch inputs
        self.MAX_STRAIGHTNESS = 0.992       # Straightness ratio >= 0.992 indicates robotic straight lines
        self.MIN_FLIGHT_STD = 12.0          # Standard deviation of inter-key interval (ms)
        self.MIN_DWELL_MEAN = 10.0          # Average key hold duration (ms)
        self.MAX_CPS = 22.0                 # Max reasonable human typing speed (characters per second)

        # Unsupervised ML Model trained to detect non-human behavioral anomalies
        self.anomaly_model = IsolationForest(contamination=0.08, random_state=42)
        self._bootstrap_model()

    def _bootstrap_model(self):
        """Pre-trains model on synthetic human behavioral baselines."""
        np.random.seed(42)
        human_baseline = np.column_stack([
            np.random.normal(90, 20, 300),     # dwell_mean (~90ms)
            np.random.normal(18, 5, 300),      # dwell_sd (~18ms)
            np.random.normal(160, 45, 300),    # flight_mean (~160ms)
            np.random.normal(40, 12, 300),     # flight_sd (~40ms)
            np.random.normal(0.4, 0.15, 300),  # mouse_speed_mean (~0.4 px/ms)
            np.random.normal(0.8, 0.25, 300),  # mouse_turn_sd (~0.8 rad)
            np.random.normal(0.72, 0.1, 300)   # straightness_mean (~0.72)
        ])
        self.anomaly_model.fit(human_baseline)

    def extract_feature_vector(self, payload: dict) -> np.ndarray:
        """Extracts numerical features from full web payloads or fallback script structures."""
        pw_derived = payload.get('keystrokes', {})
        if isinstance(pw_derived, dict):
            pw_derived = pw_derived.get('password', {}).get('derived', {})
        else:
            pw_derived = {}

        dwell_stats = pw_derived.get('dwell') or {}
        ud_stats = pw_derived.get('ud') or {}

        dwell_mean = dwell_stats.get('mean', 0)
        dwell_sd = dwell_stats.get('sd', 0)
        flight_mean = ud_stats.get('mean', 0)
        flight_sd = ud_stats.get('sd', 0)

        mouse_derived = payload.get('pointer', {}).get('derived', {}) if isinstance(payload.get('pointer'), dict) else {}
        speed_stats = mouse_derived.get('speedPxPerMs') or {}
        turn_stats = mouse_derived.get('turnAngleRad') or {}
        straight_stats = mouse_derived.get('straightness') or {}

        mouse_speed_mean = speed_stats.get('mean', 0)
        mouse_turn_sd = turn_stats.get('sd', 0)
        straightness_mean = straight_stats.get('mean', 0)

        return np.array([[
            dwell_mean,
            dwell_sd,
            flight_mean,
            flight_sd,
            mouse_speed_mean,
            mouse_turn_sd,
            straightness_mean
        ]])

    def evaluate_payload(self, payload: dict):
        """
        Evaluates full payload structure.
        Returns tuple: (is_bot: bool, confidence_score: float, violations: list[str])
        """
        violations = []
        
        env = payload.get('environment', {})
        signals = payload.get('signals', {})
        pointer = payload.get('pointer', {})
        derived_mouse = pointer.get('derived', {}) if isinstance(pointer, dict) else {}

        ks_obj = payload.get('keystrokes', {})
        ks_pass = ks_obj.get('password', {}) if isinstance(ks_obj, dict) else {}
        pass_derived = ks_pass.get('derived', {}) if isinstance(ks_pass, dict) else {}

        # --- Rule 1: Browser Environment Automation Flags ---
        if env.get('webdriver') or 'webdriver' in signals.get('flags', []):
            violations.append("Browser automation flag active (`navigator.webdriver = true`)")
        if env.get('headlessUserAgent'):
            violations.append("Headless Chrome user agent detected")

        # --- Rule 2: Event Integrity & Synthetic Injection ---
        if signals.get('untrustedEvents', 0) > 0:
            violations.append(f"Untrusted DOM events detected (count: {signals['untrustedEvents']})")
        if signals.get('syntheticClicks', 0) > 0:
            violations.append("Scripted click event detected (`detail === 0` without keyboard interaction)")

        # --- Rule 3: Keystroke Anomalies ---
        if ks_pass.get('paste', 0) > 0 or ks_pass.get('nonKeyInputEvents', 0) > 0:
            violations.append("Password form field populated via paste or script injection")

        cps = pass_derived.get('typingSpeedCps')
        if cps and cps > self.MAX_CPS:
            violations.append(f"Superhuman typing speed detected ({cps} CPS)")

        ud_stats = pass_derived.get('ud') or {}
        if ud_stats.get('n', 0) >= 3 and ud_stats.get('sd', 99) < self.MIN_FLIGHT_STD:
            violations.append(f"Robotic flight time distribution (std: {ud_stats.get('sd')} ms)")

        dwell_stats = pass_derived.get('dwell') or {}
        if dwell_stats.get('n', 0) >= 3 and dwell_stats.get('mean', 99) < self.MIN_DWELL_MEAN:
            violations.append(f"Inhuman key dwell duration (mean: {dwell_stats.get('mean')} ms)")

        # --- Rule 4: Pointer / Mouse Movement Trajectory ---
        pointer_type = pointer.get('type') if isinstance(pointer, dict) else None
        mouse_samples = derived_mouse.get('samples', 0)
        
        if pointer_type and pointer_type != 'touch' and mouse_samples < self.MIN_MOUSE_SAMPLES:
            violations.append(f"Missing pointer interaction prior to submit (samples: {mouse_samples})")

        if derived_mouse.get('teleports', 0) > 0:
            violations.append("Instant cursor coordinate teleportation detected (>10 px/ms)")

        straightness = derived_mouse.get('straightness') or {}
        if straightness.get('n', 0) >= 2 and straightness.get('mean', 0) >= self.MAX_STRAIGHTNESS:
            violations.append(f"Unnatural linear cursor movement (straightness ratio: {straightness.get('mean')})")

        # --- Layer 2: Machine Learning Model Evaluation ---
        feat_vector = self.extract_feature_vector(payload)
        ml_score = self.anomaly_model.score_samples(feat_vector)[0]
        is_ml_anomaly = self.anomaly_model.predict(feat_vector)[0] == -1

        if is_ml_anomaly and len(violations) > 0:
            violations.append(f"Behavioral pattern flagged by Anomaly Engine (score: {ml_score:.3f})")

        # Final Decision
        is_bot = len(violations) > 0
        confidence = max(0.0, min(100.0, (ml_score + 0.5) * 100)) if not violations else max(0.0, 30.0 - (len(violations) * 10))

        return is_bot, round(confidence, 1), violations


# Initialize Detector Instance
detector = BioPrintBotDetector()


def calculate_avg_dwell(payload: dict) -> float:
    """Helper to calculate average dwell time from either web GUI derived stats or direct keystrokes array."""
    # 1. Try web GUI schema format
    try:
        pw_dwell = payload.get('keystrokes', {}).get('password', {}).get('derived', {}).get('dwell', {})
        if 'mean' in pw_dwell and pw_dwell['mean'] > 0:
            return float(pw_dwell['mean'])
    except AttributeError:
        pass

    # 2. Try raw array of keystroke events
    keystrokes = payload.get('keystrokes', [])
    if isinstance(keystrokes, list) and len(keystrokes) > 0:
        dwells = []
        for k in keystrokes:
            if isinstance(k, dict) and 'pressTime' in k and 'releaseTime' in k:
                if k['releaseTime'] > k['pressTime']:
                    dwells.append(k['releaseTime'] - k['pressTime'])
        if dwells:
            return sum(dwells) / len(dwells)

    return 0.0


@app.post("/api/bioprint/telemetry")
async def process_telemetry(request: Request):
    """
    Unified Endpoint processing Web GUI submissions, API scripts, and imposter login attempts.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    # --- 1. Extract Identity & Session Mode ---
    username = payload.get('subject', {}).get('username') or payload.get('username')
    password = payload.get('subject', {}).get('password') or payload.get('password')
    session = payload.get('session', {})
    mode = session.get('mode') or payload.get('mode', 'verify')

    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")

    logging.info(f"Received telemetry payload for user '{username}' in '{mode}' mode.")

    # --- 2. LAYER 1: Bot & Automation Script Filter ---
    is_bot, confidence, violations = detector.evaluate_payload(payload)

    if is_bot:
        logging.warning(f"[BLOCKED - BOT/SCRIPT] User: '{username}'. Violations: {violations}")
        return JSONResponse(
            status_code=403,
            content={
                "status": "REJECTED",
                "reason": "BOT_OR_SCRIPT_DETECTED",
                "is_bot": True,
                "confidence_score": confidence,
                "explainability": violations
            }
        )

    # --- 3. LAYER 2: Imposter & Biometric Rhythm Engine ---
    avg_dwell = calculate_avg_dwell(payload)

    if mode == "enroll":
        if username not in USER_DATABASE:
            USER_DATABASE[username] = {
                "password": password,
                "samples": [],
                "baseline_dwell": 0.0
            }
        
        if password:
            USER_DATABASE[username]["password"] = password

        USER_DATABASE[username]["samples"].append(avg_dwell)
        samples = USER_DATABASE[username]["samples"]
        USER_DATABASE[username]["baseline_dwell"] = sum(samples) / len(samples)

        return {
            "status": "enrolled",
            "is_bot": False,
            "confidence_score": confidence,
            "message": f"Sample {len(samples)} recorded. Baseline dwell: {round(USER_DATABASE[username]['baseline_dwell'], 2)}ms",
            "sampleCount": len(samples)
        }

    elif mode in ["verify", "login"]:
        if username not in USER_DATABASE:
            logging.warning(f"[BLOCKED - UNKNOWN USER] Attempted login for '{username}'")
            return {
                "status": "blocked",
                "is_bot": False,
                "message": "ACCESS BLOCKED",
                "reason": "User not found in system."
            }

        user_record = USER_DATABASE[username]
        
        # Check standard credentials (imposter detection)
        if password and user_record["password"] and user_record["password"] != password:
            logging.warning(f"[BLOCKED - INVALID CREDENTIALS] Imposter login attempt for '{username}'")
            return {
                "status": "blocked",
                "is_bot": False,
                "message": "ACCESS BLOCKED",
                "reason": "Invalid credentials."
            }

        baseline = user_record["baseline_dwell"]
        diff = abs(avg_dwell - baseline)

        # Behavioral rhythm anomaly tolerance (60ms maximum variation)
        if diff <= 60.0 or baseline == 0.0:
            score = round(max(0.0, 100.0 - diff), 2)
            logging.info(f"[ACCEPTED] Legitimate user '{username}' verified (Score: {score}).")
            return {
                "status": "granted",
                "is_bot": False,
                "message": "ACCESS GRANTED",
                "score": score,
                "dwellMs": round(avg_dwell, 2),
                "confidence_score": confidence,
                "explainability": ["Behavioral signatures match legitimate human parameters"]
            }
        else:
            logging.warning(f"[BLOCKED - RHYTHM ANOMALY] Imposter or unnatural cadence for '{username}' (Diff: {diff}ms)")
            return {
                "status": "blocked",
                "is_bot": False,
                "message": "ACCESS BLOCKED",
                "reason": f"Biometric rhythm anomaly detected (Variance: {round(diff, 2)}ms)",
                "dwellMs": round(avg_dwell, 2)
            }

    raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}' specified.")


if __name__ == "__main__":
    import uvicorn
    print("Starting Unified BioPrint Identification Engine on http://0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)