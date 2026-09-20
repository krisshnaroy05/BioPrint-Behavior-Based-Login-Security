import logging
import math
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="BioPrint Behavioral Biometric Engine v3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_DATABASE: Dict[str, Dict[str, Any]] = {}

# Verification Threshold Configuration
VERIFY_THRESHOLD = 80.0  # Require >= 80% match for ACCESS GRANTED


class BehavioralFeatureExtractor:
    """Extracts 8-D normalized feature vectors from telemetry payloads."""

    @staticmethod
    def extract(payload: dict) -> np.ndarray:
        ks = payload.get('keystrokes', {}).get('password', {}).get('derived', {})
        ptr = payload.get('pointer', {}).get('derived', {})

        dwell_mean = float(ks.get('dwell', {}).get('mean') or 95.0)
        dwell_sd = float(ks.get('dwell', {}).get('sd') or 22.0)
        flight_mean = float(ks.get('ud', {}).get('mean') or 140.0)
        flight_sd = float(ks.get('ud', {}).get('sd') or 35.0)
        cps = float(ks.get('typingSpeedCps') or 3.8)

        # Mouse pointer metrics with fallback for idle/missing mouse
        speed_mean = float(ptr.get('speedPxPerMs', {}).get('mean') or 0.45)
        straightness = float(ptr.get('straightness', {}).get('mean') or 0.70)
        turn_sd = float(ptr.get('turnAngleRad', {}).get('sd') or 0.85)

        return np.array([
            dwell_mean, dwell_sd, flight_mean, flight_sd,
            cps, speed_mean, straightness, turn_sd
        ], dtype=np.float64)


class Tier1BotDetector:
    """Flags programmatic automation, key injectors, and extreme speed anomalies."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=0.02, random_state=42)
        self._train_baseline()

    def _train_baseline(self):
        np.random.seed(42)
        # Broader distribution of typical human typing and mouse patterns
        humans = np.column_stack([
            np.random.normal(95, 30, 600),   # dwell_mean
            np.random.normal(22, 10, 600),   # dwell_sd
            np.random.normal(140, 50, 600),  # flight_mean
            np.random.normal(38, 20, 600),   # flight_sd
            np.random.normal(4.0, 1.8, 600),  # cps
            np.random.normal(0.45, 0.25, 600),# speed
            np.random.normal(0.70, 0.18, 600),# straightness
            np.random.normal(0.85, 0.35, 600) # turn_sd
        ])
        scaled_humans = self.scaler.fit_transform(humans)
        self.model.fit(scaled_humans)

    def evaluate(self, feature_vector: np.ndarray, flags: List[str]) -> Tuple[bool, float, List[str]]:
        violations = []

        # Explicit Bot Flags from Frontend Sensors
        if 'robotic_key_interval' in flags or 'fixed_dwell_10ms' in flags:
            violations.append("Robotic fixed key interval detected")
        if 'paste_event_detected' in flags:
            violations.append("Programmatic paste detected in input field")
        if 'zero_pointer_curvature' in flags:
            violations.append("Synthetic linear pointer path detected")

        # Physical Human Limits
        if feature_vector[4] > 25.0:  # Typing speed > 25 chars/sec
            violations.append(f"Inhuman typing speed ({feature_vector[4]:.1f} CPS)")
        if feature_vector[0] < 10.0:  # Key dwell time < 10ms
            violations.append(f"Inhuman key dwell time ({feature_vector[0]:.1f}ms)")

        is_bot = len(violations) > 0
        confidence = 10.0 if is_bot else 98.0
        return is_bot, confidence, violations


class Tier2UserBiometricMatcher:
    """Verifies identity against enrolled baseline using Z-Score distances."""

    FEATURE_NAMES = [
        "Dwell Mean", "Dwell SD", "Flight Mean",
        "Flight SD", "Typing Speed", "Mouse Speed",
        "Straightness", "Turn Angle SD"
    ]
    WEIGHTS = np.array([1.2, 1.0, 1.1, 1.5, 0.8, 0.7, 0.9, 0.6])

    @classmethod
    def fit_user_baseline(cls, samples: List[np.ndarray]) -> Dict[str, np.ndarray]:
        arr = np.array(samples)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        stds = np.maximum(stds, np.array([5.0, 2.0, 10.0, 3.0, 0.3, 0.05, 0.02, 0.05]))
        return {"mean": means, "std": stds}

    @classmethod
    def verify(cls, current_feat: np.ndarray, baseline: Dict[str, np.ndarray]) -> Tuple[bool, float, List[str]]:
        means = baseline["mean"]
        stds = baseline["std"]

        # Calculate Z-score distance per biomarker
        z_scores = np.abs(current_feat - means) / stds
        weighted_z = z_scores * cls.WEIGHTS
        distance = float(np.mean(weighted_z))

        # Map distance to match score percentage (0 - 100%)
        match_score = max(5.0, min(99.0, 98.0 * math.exp(-0.35 * distance)))

        # Identify deviating biomarkers (> 2.0 standard deviations)
        mismatch_reasons = []
        for idx, z in enumerate(z_scores):
            if z > 2.0:
                mismatch_reasons.append(
                    f"{cls.FEATURE_NAMES[idx]} deviated ({current_feat[idx]:.1f} vs baseline {means[idx]:.1f})"
                )

        # STRICT THRESHOLD CHECK: Match score must be >= VERIFY_THRESHOLD (80%)
        is_match = match_score >= VERIFY_THRESHOLD
        return is_match, round(match_score, 1), mismatch_reasons

    @classmethod
    def adapt_profile(cls, baseline: Dict[str, np.ndarray], new_sample: np.ndarray, alpha: float = 0.15) -> Dict[str, np.ndarray]:
        updated_mean = (1 - alpha) * baseline["mean"] + alpha * new_sample
        new_diff = np.abs(new_sample - updated_mean)
        updated_std = (1 - alpha) * baseline["std"] + alpha * new_diff
        return {"mean": updated_mean, "std": np.maximum(updated_std, 1e-2)}


bot_detector = Tier1BotDetector()


async def process_telemetry_pipeline(request: Request, mode_override: str = None):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    username = payload.get('subject', {}).get('username') or payload.get('username')
    if not username:
        raise HTTPException(status_code=400, detail="Username parameter missing.")

    mode = mode_override or payload.get('session', {}).get('mode') or 'verify'
    flags = payload.get('signals', {}).get('flags', []) or payload.get('biometrics', {}).get('syntheticFlags', []) or []

    features = BehavioralFeatureExtractor.extract(payload)

    # Step 1: Tier-1 Bot Check
    is_bot, bot_conf, bot_violations = bot_detector.evaluate(features, flags)
    if is_bot:
        logging.warning(f"[BOT BLOCKED] User: '{username}'. Violations: {bot_violations}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked",
                "reason": "BOT_DETECTED",
                "is_bot": True,
                "score": bot_conf,
                "confidence_score": bot_conf,
                "explainability": bot_violations,
                "message": f"ACCESS DENIED: Bot behavior detected ({' | '.join(bot_violations)})"
            }
        )

    # Step 2: Enrollment Mode
    if mode == "enroll":
        if username not in USER_DATABASE:
            USER_DATABASE[username] = {"samples": []}

        USER_DATABASE[username]["samples"].append(features)
        count = len(USER_DATABASE[username]["samples"])

        if count >= 3:
            USER_DATABASE[username]["baseline"] = Tier2UserBiometricMatcher.fit_user_baseline(
                USER_DATABASE[username]["samples"]
            )
            msg = f"Biometric enrollment complete ({count}/3 samples stored)."
        else:
            msg = f"Sample {count}/3 recorded. Complete remaining attempts."

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "mode": "enroll",
                "username": username,
                "sample_count": count,
                "is_bot": False,
                "score": 98.0,
                "confidence_score": 98.0,
                "message": msg
            }
        )

    # Step 3: Check Enrollment Status for Verify Mode
    if username not in USER_DATABASE or "baseline" not in USER_DATABASE[username]:
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked",
                "reason": "USER_NOT_ENROLLED",
                "is_bot": False,
                "score": 0.0,
                "message": f"User '{username}' is not enrolled. Enroll 3 times first."
            }
        )

    # Step 4: Tier-2 Identity Verification
    baseline = USER_DATABASE[username]["baseline"]
    is_match, score, explainability = Tier2UserBiometricMatcher.verify(features, baseline)

    # Handle Flight Rhythm Mismatch Flag
    if 'flight_rhythm_mismatch' in flags:
        score = min(score, 45.0)
        is_match = False
        explainability.append("Simulated rhythm mismatch forced by client trigger")

    # Access Decision (Score < 80% returns ACCESS DENIED / blocked)
    if not is_match or score < VERIFY_THRESHOLD:
        reasons = explainability if explainability else [f"Match score ({score}%) below threshold ({VERIFY_THRESHOLD}%)"]
        logging.warning(f"[ACCESS DENIED] User: '{username}'. Score: {score}%")
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked",
                "reason": "BIOMETRIC_MISMATCH",
                "score": score,
                "confidence_score": score,
                "is_bot": False,
                "explainability": reasons,
                "message": f"ACCESS DENIED: Behavioral match {score}% is below required {VERIFY_THRESHOLD}%"
            }
        )

    # Adaptive Baseline Update on Strong Match (>= 90%)
    if score >= 90.0:
        USER_DATABASE[username]["baseline"] = Tier2UserBiometricMatcher.adapt_profile(baseline, features)

    logging.info(f"[ACCESS GRANTED] User: '{username}'. Score: {score}%")
    return JSONResponse(
        status_code=200,
        content={
            "status": "granted",
            "username": username,
            "score": score,
            "confidence_score": score,
            "is_bot": False,
            "message": f"ACCESS GRANTED: Biometric signature match ({score}%)"
        }
    )


@app.get("/")
async def root():
    return {"status": "online", "engine": "BioPrint Engine v3.1"}

@app.post("/api/enroll")
async def enroll_endpoint(request: Request):
    return await process_telemetry_pipeline(request, mode_override="enroll")

@app.post("/api/verify")
async def verify_endpoint(request: Request):
    return await process_telemetry_pipeline(request, mode_override="verify")

@app.post("/api/bioprint/telemetry")
async def telemetry_endpoint(request: Request):
    return await process_telemetry_pipeline(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
