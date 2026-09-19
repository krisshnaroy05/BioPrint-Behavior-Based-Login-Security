import numpy as np
from typing import Dict, List, Any, Optional

class BiometricFeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        key_events = payload.get("key_events", [])
        mouse_events = payload.get("mouse_events", [])
        
        # 1. Keystroke Dwell and Flight Times
        dwell_times = []
        flight_times = []
        
        # Track active key presses: { key: press_time }
        active_presses = {}
        release_times = []
        
        typo_sequence_indices = []
        
        for i, event in enumerate(key_events):
            k_type = event["type"]
            k_val = event["key"]
            t = event["timestamp"]
            
            if k_type == "keydown":
                active_presses[k_val] = t
                if release_times:
                    # Flight time: press of current key minus release of previous key
                    flight_times.append(t - release_times[-1])
                if k_val == "Backspace":
                    typo_sequence_indices.append(len(flight_times) - 1)
            elif k_type == "keyup":
                if k_val in active_presses:
                    dwell = t - active_presses.pop(k_val)
                    dwell_times.append(dwell)
                    release_times.append(t)

        dwell_arr = np.array(dwell_times) if dwell_times else np.array([0.0])
        flight_arr = np.array(flight_times) if flight_times else np.array([0.0])

        mean_dwell = float(np.mean(dwell_arr))
        mean_flight = float(np.mean(flight_arr)) if len(flight_arr) > 0 else 1.0
        flight_variance = float(np.var(flight_arr)) if len(flight_arr) > 0 else 0.0

        # Safe denominator for pace normalization
        norm_factor = mean_flight if mean_flight > 1e-3 else 1.0

        # 2. Intentional Typo Extraction & Pace Normalization
        # R_typo = Flight time to backspace / Mean session flight time
        has_typo = len(typo_sequence_indices) > 0
        if has_typo:
            target_idx = typo_sequence_indices[0]
            typo_flight = flight_times[target_idx] if target_idx < len(flight_times) else mean_flight
            typo_digram_ratio = typo_flight / norm_factor
        else:
            typo_digram_ratio = 0.0

        # 3. Mouse Trajectory Features (Low-Weight Baseline)
        mouse_dist = 0.0
        mouse_speeds = []
        for i in range(1, len(mouse_events)):
            p1 = mouse_events[i - 1]
            p2 = mouse_events[i]
            dt = p2["timestamp"] - p1["timestamp"]
            if dt > 0:
                dist = np.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])
                mouse_dist += dist
                mouse_speeds.append(dist / dt)

        mean_mouse_speed = float(np.mean(mouse_speeds)) if mouse_speeds else 0.0
        mouse_speed_var = float(np.var(mouse_speeds)) if mouse_speeds else 0.0

        # Feature Vector:
        # [0] Pace-Normalized Typo Ratio (High Weight)
        # [1] Normalized Dwell-to-Flight Ratio
        # [2] Flight Time Variance (Micro-dynamics consistency)
        # [3] Mouse Velocity Variance (Minimal Weight)
        vector = [
            round(typo_digram_ratio, 4),
            round(mean_dwell / norm_factor, 4),
            round(np.log1p(flight_variance), 4),
            round(np.log1p(mouse_speed_var), 4)
        ]

        return {
            "vector": vector,
            "raw_stats": {
                "mean_dwell": mean_dwell,
                "mean_flight": mean_flight,
                "has_typo": has_typo,
                "flight_variance": flight_variance,
                "mouse_points_count": len(mouse_events)
            }
        }
