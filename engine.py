import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Tuple

class BehavioralBiometricEngine:
    def __init__(self, ema_alpha: float = 0.15):
        self.ema_alpha = ema_alpha  # Drift update rate
        self.models: Dict[str, IsolationForest] = {}
        self.baselines: Dict[str, Dict[str, Any]] = {}

    def validate_anti_bot(self, payload: Dict[str, Any], raw_stats: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Pre-ML verification to detect automated and scripted inputs."""
        key_events = payload.get("key_events", [])
        
        if len(key_events) < 4:
            return False, "Input stream too short; probable programmatic injection."

        # Bot check: Perfect uniformity in flight time (zero human variance)
        if raw_stats["flight_variance"] < 1.0:
            return False, "Timing variance is near-zero; programmatic playback detected."

        # Bot check: Missing intentional typo ritual when expected
        if not raw_stats["has_typo"]:
            return False, "Required intentional typo sequence absent."

        return True, None

    def train_user_profile(self, user_id: str, enrollment_vectors: List[List[float]]):
        X = np.array(enrollment_vectors)
        
        # Isolation Forest modeling baseline distributions[cite: 1]
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X)
        self.models[user_id] = model
        
        # Store mean and variance for explainability calculations
        self.baselines[user_id] = {
            "mean": np.mean(X, axis=0).tolist(),
            "std": (np.std(X, axis=0) + 1e-4).tolist()  # Avoid div/0
        }

    def authenticate(self, user_id: str, vector: List[float], raw_stats: Dict[str, Any]) -> Dict[str, Any]:
        if user_id not in self.models:
            return {"authenticated": False, "reason": "User not enrolled."}

        model = self.models[user_id]
        baseline = self.baselines[user_id]
        
        x = np.array([vector])
        anomaly_score = float(-model.score_samples(x)[0]) # Lower is more normal
        is_inlier = model.predict(x)[0] == 1

        # Calibrate confidence score [0% - 100%]
        # Isolation Forest scores typically range around 0.35 (inlier) to 0.70+ (outlier)
        confidence = float(np.clip((0.65 - anomaly_score) / 0.30 * 100, 0.0, 100.0))

        # Explainability: Compute Z-Scores for each biometric signal[cite: 1]
        feature_names = [
            "Pace-Normalized Typo Ratio",
            "Dwell-to-Flight Pace Ratio",
            "Keystroke Flight Variance",
            "Mouse Movement Dynamics"
        ]
        
        means = np.array(baseline["mean"])
        stds = np.array(baseline["std"])
        z_scores = np.abs((np.array(vector) - means) / stds)
        
        flagged_signals = []
        for name, z in zip(feature_names, z_scores):
            if z > 2.5:
                flagged_signals.append(f"{name} deviated significantly (Z={z:.2f})")

        passed = is_inlier and confidence >= 60.0

        # Adaptive Profile Update: Apply EMA if authenticated with high confidence[cite: 1]
        if passed and confidence >= 80.0:
            updated_mean = (1 - self.ema_alpha) * np.array(baseline["mean"]) + self.ema_alpha * np.array(vector)
            self.baselines[user_id]["mean"] = updated_mean.tolist()

        return {
            "authenticated": passed,
            "confidence_score": round(confidence, 2),
            "anomaly_score": round(anomaly_score, 4),
            "adaptive_drift_applied": passed and confidence >= 80.0,
            "flagged_signals": flagged_signals,
            "summary": "Access Granted: Behavioral profile matched." if passed else "Access Denied: Impostor dynamics detected."
        }
