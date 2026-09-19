from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="BioPrint Telemetry Engine")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory user database
USER_DATABASE: Dict[str, Dict[str, Any]] = {}

class Keystroke(BaseModel):
    key: str
    pressTime: float
    releaseTime: float

class TelemetryPayload(BaseModel):
    username: str
    password: str
    mode: str  # "enroll" or "verify"
    keystrokes: List[Keystroke]

@app.post("/api/bioprint/telemetry")
async def process_telemetry(payload: TelemetryPayload):
    if not payload.username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    # Calculate average key dwell time (releaseTime - pressTime)
    dwell_times = [k.releaseTime - k.pressTime for k in payload.keystrokes if k.releaseTime > k.pressTime]
    avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0.0

    if payload.mode == "enroll":
        if payload.username not in USER_DATABASE:
            USER_DATABASE[payload.username] = {
                "password": payload.password,
                "samples": [],
                "baseline_dwell": 0.0
            }
        
        USER_DATABASE[payload.username]["samples"].append(avg_dwell)
        samples = USER_DATABASE[payload.username]["samples"]
        USER_DATABASE[payload.username]["baseline_dwell"] = sum(samples) / len(samples)

        return {
            "status": "enrolled",
            "message": f"Sample {len(samples)}/3 recorded. Avg dwell: {round(avg_dwell, 2)}ms",
            "sampleCount": len(samples)
        }

    elif payload.mode == "verify":
        if payload.username not in USER_DATABASE:
            return {
                "status": "blocked",
                "message": "ACCESS BLOCKED",
                "reason": "User not found in system."
            }

        user_record = USER_DATABASE[payload.username]
        if user_record["password"] != payload.password:
            return {
                "status": "blocked",
                "message": "ACCESS BLOCKED",
                "reason": "Invalid credentials."
            }

        baseline = user_record["baseline_dwell"]
        diff = abs(avg_dwell - baseline)

        # Verification threshold (variance tolerance: 60ms)
        if diff <= 60.0:
            return {
                "status": "granted",
                "message": "ACCESS GRANTED",
                "score": round(max(0, 100 - diff), 2),
                "dwellMs": round(avg_dwell, 2)
            }
        else:
            return {
                "status": "blocked",
                "message": "ACCESS BLOCKED",
                "reason": f"Rhythm anomaly detected (Variance: {round(diff, 2)}ms)",
                "dwellMs": round(avg_dwell, 2)
            }

    raise HTTPException(status_code=400, detail="Invalid mode specified.")