import logging
import math
import os
import sys
import threading
import webbrowser
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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
VERIFY_THRESHOLD = 80.0  # Require >= 80% match for ACCESS GRANTED


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


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
        humans = np.column_stack([
            np.random.normal(95, 30, 600),
            np.random.normal(22, 10, 600),
            np.random.normal(140, 50, 600),
            np.random.normal(38, 20, 600),
            np.random.normal(4.0, 1.8, 600),
            np.random.normal(0.45, 0.25, 600),
            np.random.normal(0.70, 0.18, 600),
            np.random.normal(0.85, 0.35, 600)
        ])
        scaled_humans = self.scaler.fit_transform(humans)
        self.model.fit(scaled_humans)

    def evaluate(self, feature_vector: np.ndarray, flags: List[str]) -> Tuple[bool, float, List[str]]:
        violations = []

        if 'robotic_key_interval' in flags or 'fixed_dwell_10ms' in flags:
            violations.append("Robotic fixed key interval detected")
        if 'paste_event_detected' in flags:
            violations.append("Programmatic paste detected in input field")
        if 'zero_pointer_curvature' in flags:
            violations.append("Synthetic linear pointer path detected")

        if feature_vector[4] > 25.0:
            violations.append(f"Inhuman typing speed ({feature_vector[4]:.1f} CPS)")
        if feature_vector[0] < 10.0:
            violations.append(f"Inhuman key dwell time ({feature_vector[0]:.1f}ms)")

        is_bot = len(violations) > 0
        confidence = 10.0 if is_bot else 98.0
        return is_bot, confidence, violations


class SupervisedUserBiometricMatcher:
    """
    Supervised ML Classifier using Random Forest.
    Trains a binary model per user using 7 enrolled samples (Class 1)
    augmented with mild physical noise, and synthetic background impostors (Class 0).
    """

    FEATURE_NAMES = [
        "Dwell Mean", "Dwell SD", "Flight Mean",
        "Flight SD", "Typing Speed", "Mouse Speed",
        "Straightness", "Turn Angle SD"
    ]

    @classmethod
    def _generate_background_impostors(cls, count: int = 200) -> np.ndarray:
        """Generates a background dataset of human typing/mouse metrics."""
        np.random.seed(42)
        return np.column_stack([
            np.random.normal(95, 35, count),   # dwell_mean
            np.random.normal(22, 12, count),   # dwell_sd
            np.random.normal(140, 60, count),  # flight_mean
            np.random.normal(38, 25, count),   # flight_sd
            np.random.normal(4.0, 2.0, count), # cps
            np.random.normal(0.45, 0.3, count),# speed
            np.random.normal(0.70, 0.2, count),# straightness
            np.random.normal(0.85, 0.4, count) # turn_sd
        ])

    @classmethod
    def train_user_model(cls, samples: List[np.ndarray]) -> Dict[str, Any]:
        """
        Trains a Supervised RandomForestClassifier for the specific user.
        Uses Gaussian noise augmentation on enrolled samples to form Class 1.
        """
        positive_samples = np.array(samples)

        # Augment 7 enrolled samples into 35 positive training instances
        augmented_positives = []
        for sample in positive_samples:
            augmented_positives.append(sample)
            for _ in range(4):
                noise = np.random.normal(0, 0.03 * (np.abs(sample) + 1e-5), size=sample.shape)
                augmented_positives.append(sample + noise)

        X_pos = np.array(augmented_positives)
        y_pos = np.ones(len(X_pos))

        # Generate negative background samples (impostors)
        X_neg = cls._generate_background_impostors(count=200)
        y_neg = np.zeros(len(X_neg))

        X = np.vstack([X_pos, X_neg])
        y = np.hstack([y_pos, y_neg])

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X, y)

        return {
            "model": clf,
            "baseline_mean": np.mean(positive_samples, axis=0)
        }

    @classmethod
    def verify(cls, current_feat: np.ndarray, profile: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        clf: RandomForestClassifier = profile["model"]
        baseline_mean = profile["baseline_mean"]

        # Supervised probability for Class 1 (Legitimate User)
        probabilities = clf.predict_proba(current_feat.reshape(1, -1))[0]
        class_idx = list(clf.classes_).index(1.0) if 1.0 in clf.classes_ else 1
        match_score = round(float(probabilities[class_idx]) * 100.0, 1)

        mismatch_reasons = []
        for idx in range(len(current_feat)):
            baseline_val = baseline_mean[idx]
            curr_val = current_feat[idx]
            denom = abs(baseline_val) if abs(baseline_val) > 1e-5 else 1.0
            diff_pct = abs(curr_val - baseline_val) / denom * 100.0

            if diff_pct > 30.0:
                mismatch_reasons.append(
                    f"{cls.FEATURE_NAMES[idx]} deviated by {diff_pct:.1f}% from baseline ({curr_val:.1f} vs {baseline_val:.1f})"
                )

        is_match = match_score >= VERIFY_THRESHOLD
        return is_match, match_score, mismatch_reasons


bot_detector = Tier1BotDetector()


async def process_telemetry_pipeline(request: Request, mode_override: str = None):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    username = payload.get('subject', {}).get('username') or payload.get('username')
    password = payload.get('subject', {}).get('password') or payload.get('password') or ""

    if not username:
        raise HTTPException(status_code=400, detail="Username parameter missing.")

    mode = mode_override or payload.get('session', {}).get('mode') or 'verify'
    flags = payload.get('signals', {}).get('flags', []) or payload.get('biometrics', {}).get('syntheticFlags', []) or []

    features = BehavioralFeatureExtractor.extract(payload)

    # Tier 1 Bot Evaluation
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

    # ENROLLMENT MODE
    if mode == "enroll":
        if username not in USER_DATABASE:
            USER_DATABASE[username] = {"samples": [], "password": password}

        # Validate that the password matches across all enrollment trials
        enrolled_password = USER_DATABASE[username].get("password")
        if enrolled_password and password != enrolled_password:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "error",
                    "reason": "PASSWORD_MISMATCH",
                    "is_bot": False,
                    "score": 0.0,
                    "message": "Password mismatch! You must enter the EXACT SAME password across all 7 enrollment trials."
                }
            )

        USER_DATABASE[username]["samples"].append(features)
        count = len(USER_DATABASE[username]["samples"])

        if count >= 7:
            USER_DATABASE[username]["profile"] = SupervisedUserBiometricMatcher.train_user_model(
                USER_DATABASE[username]["samples"]
            )
            msg = f"Supervised ML enrollment complete ({count}/7 trials trained)."
        else:
            msg = f"Sample {count}/7 recorded. Complete remaining attempts with the same password."

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

    # VERIFICATION MODE
    if username not in USER_DATABASE or "profile" not in USER_DATABASE[username]:
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked",
                "reason": "USER_NOT_ENROLLED",
                "is_bot": False,
                "score": 0.0,
                "message": f"User '{username}' is not enrolled. Complete all 7 enrollment trials first."
            }
        )

    # Check Password Validity
    enrolled_password = USER_DATABASE[username].get("password")
    if password != enrolled_password:
        logging.warning(f"[ACCESS DENIED] User: '{username}'. Reason: Incorrect password.")
        return JSONResponse(
            status_code=200,
            content={
                "status": "blocked",
                "reason": "INCORRECT_PASSWORD",
                "score": 0.0,
                "confidence_score": 0.0,
                "is_bot": False,
                "explainability": ["Entered password does not match enrolled baseline credential."],
                "message": "ACCESS DENIED: Incorrect password entered."
            }
        )

    # Tier 2 Supervised ML Verification
    profile = USER_DATABASE[username]["profile"]
    is_match, score, explainability = SupervisedUserBiometricMatcher.verify(features, profile)

    if 'flight_rhythm_mismatch' in flags:
        score = min(score, 45.0)
        is_match = False
        explainability.append("Simulated rhythm mismatch forced by client trigger")

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

    logging.info(f"[ACCESS GRANTED] User: '{username}'. Supervised Score: {score}%")
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


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    file_path = get_resource_path("bioprint_merged.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>Error: bioprint_merged.html not found</h2>", status_code=404)


@app.post("/api/enroll")
async def enroll_endpoint(request: Request):
    return await process_telemetry_pipeline(request, mode_override="enroll")

@app.post("/api/verify")
async def verify_endpoint(request: Request):
    return await process_telemetry_pipeline(request, mode_override="verify")

@app.post("/api/bioprint/telemetry")
async def telemetry_endpoint(request: Request):
    return await process_telemetry_pipeline(request)


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    import uvicorn
    threading.Timer(1.2, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
