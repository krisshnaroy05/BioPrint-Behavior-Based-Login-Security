from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import time

from ml.feature_extractor import BiometricFeatureExtractor
from ml.engine import BehavioralBiometricEngine

app = FastAPI(title="BioPrint Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = BiometricFeatureExtractor()
engine = BehavioralBiometricEngine()

class BiometricPayload(BaseModel):
    user_id: str
    key_events: List[Dict[str, Any]]
    mouse_events: List[Dict[str, Any]]

class EnrollmentPayload(BaseModel):
    user_id: str
    samples: List[BiometricPayload]

@app.post("/enroll")
async def enroll(payload: EnrollmentPayload):
    if len(payload.samples) < 5:
        raise HTTPException(status_code=400, detail="Minimum 5 enrollment attempts required.")[cite: 1]

    vectors = []
    for sample in payload.samples:
        extracted = extractor.extract_features(sample.dict())
        vectors.append(extracted["vector"])

    engine.train_user_profile(payload.user_id, vectors)
    return {"status": "Enrolled", "samples_processed": len(vectors)}

@app.post("/authenticate")
async def authenticate(payload: BiometricPayload):
    t_start = time.perf_counter()
    data = payload.dict()
    
    extracted = extractor.extract_features(data)
    vector = extracted["vector"]
    raw_stats = extracted["raw_stats"]

    # 1. Anti-Bot and Replay Verification[cite: 1]
    is_human, bot_reason = engine.validate_anti_bot(data, raw_stats)
    if not is_human:
        return {
            "authenticated": False,
            "is_bot": True,
            "confidence_score": 0.0,
            "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            "summary": f"Blocked automated traffic: {bot_reason}"
        }

    # 2. Machine Learning Anomaly Inference[cite: 1]
    result = engine.authenticate(payload.user_id, vector, raw_stats)
    result["is_bot"] = False
    result["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    
    return result
