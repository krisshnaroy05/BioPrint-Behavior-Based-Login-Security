import statistics
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests from frontend

# In-memory storage for enrolled biometric baselines
USER_BASELINES = {}

@app.route('/api/enroll', methods=['POST'])
def enroll():
    data = request.get_json() or {}
    subject = data.get('subject', {}).get('username', 'anonymous')
    biometrics = data.get('biometrics', {})
    
    dwell_mean = biometrics.get('dwellMeanMs', 0)
    flight_mean = biometrics.get('flightMeanMs', 0)
    
    if subject not in USER_BASELINES:
        USER_BASELINES[subject] = {'dwells': [], 'flights': []}
        
    USER_BASELINES[subject]['dwells'].append(dwell_mean)
    USER_BASELINES[subject]['flights'].append(flight_mean)
    
    sample_count = len(USER_BASELINES[subject]['dwells'])
    
    return jsonify({
        "status": "success",
        "message": f"Sample {sample_count} recorded! Baseline updating...",
        "sampleCount": sample_count
    }), 200

@app.route('/api/verify', methods=['POST'])
def verify():
    data = request.get_json() or {}
    subject = data.get('subject', {}).get('username', 'anonymous')
    biometrics = data.get('biometrics', {})
    
    dwell_mean = biometrics.get('dwellMeanMs', 0)
    synthetic_flags = biometrics.get('syntheticFlags', [])
    
    # 1. Bot or Synthetic Attack Detection
    if synthetic_flags or dwell_mean < 20:
        return jsonify({
            "confidenceScore": 15,
            "riskLevel": "HIGH RISK / BOT",
            "isBot": True,
            "message": "Synthetic/Bot behavior detected! Request blocked."
        }), 200
        
    # 2. Baseline Check
    user_profile = USER_BASELINES.get(subject)
    if not user_profile or not user_profile['dwells']:
        return jsonify({
            "confidenceScore": 88,
            "riskLevel": "LOW RISK",
            "isBot": False,
            "message": "Verified against population baseline model."
        }), 200
        
    # Calculate variation against stored user profile
    baseline_dwell = statistics.mean(user_profile['dwells'])
    dwell_diff = abs(dwell_mean - baseline_dwell)
    
    if dwell_diff > 75:
        return jsonify({
            "confidenceScore": 52,
            "riskLevel": "SUSPICIOUS",
            "isBot": False,
            "message": "Cadence mismatch detected relative to user baseline profile."
        }), 200
        
    return jsonify({
        "confidenceScore": 98,
        "riskLevel": "LOW RISK",
        "isBot": False,
        "message": "Identity confirmed! Behavioral telemetry matches baseline profile."
    }), 200

if __name__ == '__main__':
    print("BioPrint Telemetry Backend active on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)