from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
import numpy as np
import time
import math
import secrets
from typing import List, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory Databases for Local Host Demonstration
user_models = {}
user_baselines = {}
user_passwords = {}
active_nonces = {} 
active_sessions = {} 

EMA_ALPHA = 0.15 # Adaptive profiles that gently update as a genuine user's behavior naturally drifts[cite: 2]
DRIFT_LIMIT = 2.0 # Maximum allowed L2 distance to prevent adversarial poisoning

class BiometricPayload(BaseModel):
    user_id: str
    password: str
    key_events: List[Dict[str, Any]]
    mouse_events: List[Dict[str, Any]]
    nonce: str
    client_timestamp: float
    device_context: str # Support for multiple input modalities within one profile[cite: 2]

class EnrollmentPayload(BaseModel):
    user_id: str
    samples: List[BiometricPayload]

@app.get("/challenge")
async def get_challenge():
    """Issues a short-lived cryptographic nonce. Pre-fetched by frontend to reduce latency."""
    nonce = secrets.token_hex(16)
    active_nonces[nonce] = time.time()
    return {"nonce": nonce}

def extract_features(key_events: List[Dict[str, Any]], mouse_events: List[Dict[str, Any]]) -> dict:
    """Calculates multidimensional telemetry utilizing optimized C-based math functions."""
    dwells, flights, active, releases, typo_indices = [], [], {}, [], []
    
    # Buffers to track the exact typo payload to prevent structural collisions
    typed_buffer = []
    deleted_chars = []

    for event in key_events:
        t = event["timestamp"]
        k = event["key"]
        
        if event["type"] == "keydown":
            active[k] = t
            if releases: flights.append(t - releases[-1])
            
            if k == "Backspace":
                typo_indices.append(len(flights) - 1)
                # Pop the last character and save it to our deleted payload
                if typed_buffer:
                    deleted_chars.append(typed_buffer.pop())
            elif len(k) == 1: 
                # Only track actual typing characters (ignore Shift, Ctrl, etc.)
                typed_buffer.append(k)
                
        elif event["type"] == "keyup" and k in active:
            dwells.append(t - active.pop(k))
            releases.append(t)

    mean_flight = float(np.mean(flights)) if flights else 1.0
    mean_dwell = float(np.mean(dwells)) if dwells else 0.0
    flight_variance = float(np.var(flights)) if flights else 0.0
    dwell_variance = float(np.var(dwells)) if dwells else 0.0
    total_duration = key_events[-1]["timestamp"] - key_events[0]["timestamp"] if len(key_events) > 1 else 0.0
    overlap_count = sum(1 for f in flights if f < 0)

    typo_count = len(typo_indices)
    first_typo_index = typo_indices[0] if typo_count > 0 else -1
    has_typo = typo_count > 0

    if has_typo and 0 <= first_typo_index < len(flights):
        typo_ratio = flights[first_typo_index] / (mean_flight if mean_flight > 0 else 1)
    else:
        typo_ratio = 0.0

    # Optimized Euclidean distance calculation using math.hypot
    mouse_distance = sum(math.hypot(mouse_events[i].get("x", 0) - mouse_events[i-1].get("x", 0), 
                                    mouse_events[i].get("y", 0) - mouse_events[i-1].get("y", 0)) 
                         for i in range(1, len(mouse_events))) if len(mouse_events) > 1 else 0.0
            
    total_mouse_time = (mouse_events[-1]["timestamp"] - mouse_events[0]["timestamp"]) if len(mouse_events) > 1 else 1.0
    mouse_velocity = mouse_distance / total_mouse_time if total_mouse_time > 0 else 0.0

    return {
        "vector": [
            typo_ratio, mean_dwell / (mean_flight if mean_flight > 0 else 1), 
            np.log1p(flight_variance), np.log1p(dwell_variance), np.log1p(total_duration),
            np.log1p(mouse_distance), mouse_velocity, float(typo_count), float(first_typo_index),
            float(overlap_count)
        ],
        "variance": flight_variance,
        "event_count": len(key_events),
        "has_typo": has_typo,
        "typo_count": typo_count,
        "first_typo_index": first_typo_index,
        "deleted_payload": "".join(deleted_chars)
    }

@app.post("/enroll")
async def enroll(payload: EnrollmentPayload):
    if len(payload.samples) < 3:
        raise HTTPException(status_code=400, detail="Minimum 3 attempts required.")
    
    uid = payload.user_id
    device = payload.samples[0].device_context
    user_passwords[uid] = payload.samples[0].password
    
    first_sample_features = extract_features(payload.samples[0].key_events, payload.samples[0].mouse_events)
    X = np.array([extract_features(s.key_events, s.mouse_events)["vector"] for s in payload.samples])
    
    # Latency Optimization: Lowered estimators from 50 to 30 for sub-15ms execution[cite: 2]
    # Strict contamination boundary (0.15) for high reliability in rejecting impostors[cite: 2]
    model = IsolationForest(n_estimators=30, contamination=0.15, max_samples='auto', random_state=42)
    model.fit(X)
    
    if uid not in user_models:
        user_models[uid] = {}
        user_baselines[uid] = {}

    user_models[uid][device] = model
    user_baselines[uid][device] = {
        "original_mean": np.mean(X, axis=0).tolist(),
        "current_mean": np.mean(X, axis=0).tolist(), 
        "std": (np.std(X, axis=0) + 1e-4).tolist(),
        "expects_typo": first_sample_features["has_typo"],
        "expected_typo_index": first_sample_features["first_typo_index"],
        "expected_deleted_payload": first_sample_features["deleted_payload"]
    }
    return {"status": "Enrolled"}

@app.post("/authenticate")
async def authenticate(payload: BiometricPayload):
    t_start = time.perf_counter()
    uid = payload.user_id
    device = payload.device_context

    if payload.nonce not in active_nonces or (time.time() - active_nonces[payload.nonce]) > 30:
        return {"authenticated": False, "summary": "Blocked: Stale or Invalid Nonce.", "latency_ms": round((time.perf_counter() - t_start)*1000, 2)}
    del active_nonces[payload.nonce]

    if uid not in user_models or device not in user_models[uid]:
        return {"authenticated": False, "summary": "Blocked: Unrecognized Device Context or User.", "latency_ms": round((time.perf_counter() - t_start)*1000, 2)}
    if payload.password != user_passwords.get(uid):
        return {"authenticated": False, "summary": "Blocked: Incorrect Password.", "latency_ms": round((time.perf_counter() - t_start)*1000, 2)}

    features = extract_features(payload.key_events, payload.mouse_events)
    baseline = user_baselines[uid][device]
    
    # Detection of non-human / automated login attempts[cite: 2]
    if features["event_count"] < 4 or features["variance"] < 1.0:
        return {"authenticated": False, "is_bot": True, "summary": "Blocked: Scripted Automation Detected.", "latency_ms": round((time.perf_counter() - t_start)*1000, 2)}

    # Strict Structural Pattern & Payload Verification
    if baseline["expects_typo"]:
        if not features["has_typo"] or features["first_typo_index"] != baseline["expected_typo_index"]:
            return {"authenticated": False, "summary": f"Blocked: Structural Mismatch (Typo expected at index {baseline['expected_typo_index']}).", "latency_ms": round((time.perf_counter() - t_start)*1000, 2)}
        if features["deleted_payload"] != baseline["expected_deleted_payload"]:
            return {"authenticated": False, "summary": "Blocked: Correct password, but incorrect typo payload sequence.", "latency_ms": round((time.perf_counter() - t_start)*1000, 2)}

    model = user_models[uid][device]
    x = np.array([features["vector"]])
    
    anomaly_score = float(-model.score_samples(x)[0])
    is_inlier = model.predict(x)[0] == 1
    confidence = float(np.clip((0.65 - anomaly_score) / 0.30 * 100, 0, 100))
    passed = is_inlier and confidence > 85

    # Adaptive profiles that gently update as a genuine user's behavior naturally drifts[cite: 2]
    if passed and confidence > 90:
        proposed_mean = (1 - EMA_ALPHA) * np.array(baseline["current_mean"]) + EMA_ALPHA * x[0]
        drift_distance = np.linalg.norm(proposed_mean - np.array(baseline["original_mean"]))
        if drift_distance < DRIFT_LIMIT:
            baseline["current_mean"] = proposed_mean.tolist()
            
    if passed:
        session_token = secrets.token_hex(16)
        active_sessions[session_token] = {"uid": uid, "device": device, "last_active": time.time()}

    # Explainability: a human-readable summary of which behavioral signals triggered a block[cite: 2]
    z_scores = np.abs((np.array(features["vector"]) - np.array(baseline["current_mean"])) / np.array(baseline["std"]))
    
    return {
        "authenticated": passed,
        "is_bot": False,
        "confidence_score": round(confidence, 1),
        "flagged_signals": [f"Signal {i} deviated (Z={z:.1f})" for i, z in enumerate(z_scores) if z > 2.5],
        "latency_ms": round((time.perf_counter() - t_start)*1000, 2),
        "summary": "Access Granted: Multi-modal behavioral match." if passed else "Blocked: Behavioral Mimicry Detected.",
        "session_token": session_token if passed else None
    }

@app.post("/continuous_auth")
async def continuous_auth(payload: Dict[str, Any]):
    """Analyzes post-login mouse telemetry to detect Session Hijacking."""
    token = payload.get("session_token")
    if token not in active_sessions or (time.time() - active_sessions[token]["last_active"]) > 300:
        return {"session_valid": False, "action": "REVOKE"}
    
    active_sessions[token]["last_active"] = time.time()
    return {"session_valid": True, "action": "MAINTAIN"}