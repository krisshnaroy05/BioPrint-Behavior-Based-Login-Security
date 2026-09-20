# 🔐 BioPrint AI — Behavioral Biometric Authentication Engine

> **Passwordless-proof identity through behavioral biometrics.**
> A login system that verifies *who you are* by *how you type and move* — not just what password you enter.

Built for the **BioPrint: Behavior-Based Login Security** hackathon track
(Theme: *Passwordless-proof identity through behavioral biometrics*, 36-hour build).

Status: Hackathon Demo  |  Language: Python 3.9+  |  Backend: FastAPI  |  ML: scikit-learn
---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Features](#-key-features)
4. [Architecture](#-architecture)
5. [How It Works — Deep Dive](#-how-it-works--deep-dive)
   - [Behavioral Signals Captured](#behavioral-signals-captured)
   - [Feature Vector](#feature-vector)
   - [Tier 1 — Bot & Automation Detection](#tier-1--bot--automation-detection)
   - [Tier 2 — Supervised Per-User Matching](#tier-2--supervised-per-user-matching)
   - [Enrollment Pipeline](#enrollment-pipeline)
   - [Verification Pipeline](#verification-pipeline)
6. [Tech Stack](#-tech-stack)
7. [Project Structure](#-project-structure)
8. [Getting Started](#-getting-started)
9. [Frontend Walkthrough](#-frontend-walkthrough)
10. [API Reference](#-api-reference)
11. [Telemetry Payload Schema](#-telemetry-payload-schema)
12. [Demo Flow / Script](#-demo-flow--script)
13. [Judging Criteria Alignment](#-judging-criteria-alignment)
14. [Security Considerations](#-security-considerations)
15. [Known Limitations](#-known-limitations--notes)
16. [Roadmap / Stretch Goals](#-roadmap--stretch-goals)
17. [Troubleshooting / FAQ](#-troubleshooting--faq)
18. [Contributing](#-contributing)
19. [Team](#-team)

---

## 📖 Overview

Passwords can be guessed, phished, or leaked — but the way a person types, pauses, hesitates, and moves their mouse
is a signal that's extraordinarily difficult for an attacker (human or bot) to replicate, even if they know the
exact password.

**BioPrint AI** is a full-stack behavioral authentication engine that:

- Captures **keystroke dynamics** (dwell time, flight/up-down time, typing speed) and **pointer dynamics**
  (movement speed, path straightness, turn-angle variance) while a user fills in the login form.
- Builds a **unique behavioral fingerprint** per user during a guided enrollment flow — completely independent of
  their password.
- Verifies live login attempts against that fingerprint using a **supervised machine learning model**, and can
  **block access on a purely behavioral mismatch**, even when the password entered is 100% correct.
- Does all of this **without any OTP, email link, or secondary-verification fallback** — the behavioral signal
  alone is the second factor.
- Separately detects **non-human / bot-driven login attempts** (scripted input, paste injection, robotic key
  timing, perfectly straight pointer paths) as a distinct fraud signal, flagged *before* identity matching even runs.

The system ships as a single, self-contained full-stack app: a FastAPI backend that trains and serves the ML
models, plus a single-file animated frontend that captures telemetry, drives enrollment/login, and visualizes the
decision pipeline live.

---

## 🎯 Problem Statement

> Build a Chrome extension or a website login page that authenticates users using behavioral biometrics... even if
> an attacker has the correct password, the system should still be able to recognize that the person behind the
> keyboard is not the genuine account owner and block the login purely by comparing live behavior against the
> user's established pattern... without falling back on OTPs or any other secondary-verification step... and
> equally flag non-human traffic, such as scripted or bot-driven login attempts.

**BioPrint AI implements this as a website login page** (not a browser extension), satisfying every core
requirement:

| Requirement | Implementation |
|---|---|
| Team-designed enrollment flow | 7-trial guided password entry with live progress HUD |
| Behavioral fingerprinting engine, password-independent | 8-D feature vector built purely from typing/mouse dynamics |
| Live authentication check, no OTP fallback | Random Forest match score gates access directly |
| Bot / automation detection as a distinct signal | Isolation Forest + rule-based Tier 1 screen, separate `is_bot` flag |
| Working demo showing genuine vs. blocked logins | Built-in "Script Bot", "Paste Attack", and "Impostor" simulation buttons |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🖊️ **Keystroke Dynamics** | Dwell time (mean & SD), flight/up-down time (mean & SD), and typing speed (characters/sec), captured per password entry |
| 🖱️ **Pointer Dynamics** | Mouse movement speed, path straightness ratio, and turn-angle variance, captured across the session |
| 🧠 **Two-Tier Detection Pipeline** | Tier 1: unsupervised Isolation Forest anomaly/bot screen · Tier 2: supervised per-user Random Forest identity match |
| 🤖 **Bot & Automation Detection** | Flags robotic fixed key intervals, programmatic paste events, zero-curvature (perfectly straight) pointer paths, and inhuman CPS/dwell values |
| 📊 **Explainable Scoring** | Every decision returns a numeric confidence score *and* a human-readable list of the exact reasons behind it |
| 🔁 **Guided Enrollment Flow** | 7-trial enrollment with same-password consistency validation and a live sample-count progress bar |
| ⚡ **Real-Time Verification** | Single in-memory forward pass per request — decision latency is displayed live in the UI (milliseconds) |
| 🎛️ **Live Demo / Attack Simulation Studio** | One-click buttons to simulate a script bot, a clipboard-paste attack, and an impostor login, directly from the browser |
| 📈 **Live Intelligence Dashboard** | Canvas-rendered keystroke rhythm chart, confidence gauge, entropy score, payload size, and risk badge, all updating in real time |
| 🧾 **Downloadable Telemetry JSON** | Inspect or export the exact payload sent to the backend for any given attempt (`Copy JSON` / `Download JSON`) |
| 💻 **SDK / Integration Modal** | In-app modal documenting how to wire the telemetry capture script into another site |
| 📦 **Portable Executable** | Packaged via PyInstaller into a single `.exe` — no Python install needed to run the demo |

---

## 🏗️ Architecture

```
┌──────────────────────────────┐        JSON telemetry (POST)        ┌───────────────────────────────────┐
│    bioprint_merged.html      │ ───────────────────────────────────▶│              main.py                │
│  ───────────────────────────  │                                      │   FastAPI · BioPrint Engine v3.1    │
│  • Live Studio (capture UI)   │◀─────────────────────────────────── │                                      │
│  • Intelligence dashboard     │        JSON decision (200 OK)        │                                      │
│  • SDK Setup modal            │                                      │                                      │
└──────────────────────────────┘                                      └───────────────────┬─────────────────┘
                                                                                            │
                                                                       process_telemetry_pipeline()
                                                                                            │
                                                             ┌──────────────────────────────┼──────────────────────────────┐
                                                             │                                                              │
                                                  Tier 1 · Tier1BotDetector                                    (only if not a bot)
                                                  • IsolationForest (pretrained,                                            │
                                                    synthetic human baseline,                                               │
                                                    contamination=0.02)                                                     │
                                                  • Rule checks: robotic key                                                │
                                                    interval, paste event,                                                  │
                                                    zero pointer curvature,                                                 │
                                                    CPS > 25, dwell < 10ms                                                  │
                                                             │                                                              │
                                                             ▼                                                              ▼
                                                    is_bot? ──Yes──▶ 200 OK, status="blocked",              Enrollment mode  │  Verification mode
                                                       │              reason="BOT_DETECTED"                       │                    │
                                                       No                                                          ▼                    ▼
                                                       │                                              Store sample → train        Password check
                                                       ▼                                              SupervisedUserBiometric-     → SupervisedUserBiometric-
                                              (continues to mode branch)                               Matcher after 7/7           Matcher.verify()
                                                                                                                                     → threshold ≥ 80%
                                                                                                                                     → grant / deny + explainability

                                                                    In-memory USER_DATABASE: { username: { samples[], password, profile: { model, baseline_mean } } }
```

### Request lifecycle (`process_telemetry_pipeline`)

1. Parse & validate JSON body (`username`, `password`, telemetry, `session.mode`, `signals.flags`).
2. Extract the 8-D feature vector via `BehavioralFeatureExtractor`.
3. Run **Tier 1 bot evaluation** — if flagged, short-circuit and return `BOT_DETECTED` immediately (identity is
   never checked for bot traffic).
4. Branch on `mode`:
   - `enroll` → validate password consistency across trials → store sample → train model on the 7th sample.
   - `verify` → check user is enrolled → check password → run **Tier 2 supervised match** → threshold decision.
5. Return a structured JSON decision with `status`, `score`, `is_bot`, and `explainability`.

---

## 🔬 How It Works — Deep Dive

### Behavioral Signals Captured

The frontend listens to raw DOM events on the password field and across the page and derives statistics from them
in real time:

**Keystroke dynamics** (captured specifically on the password input):
- **Dwell time** — how long each key is physically held down (key-down → key-up), in milliseconds. Every person
  has a characteristic average press duration and variance.
- **Flight / up-down (UD) time** — the interval between releasing one key and pressing the next. This captures the
  *rhythm* between keystrokes, not just individual presses.
- **Typing speed (CPS)** — characters typed per second, derived from total characters and total elapsed time.

**Pointer dynamics** (captured across mouse movement on the page):
- **Speed (px/ms)** — how fast the cursor moves.
- **Straightness** — ratio of straight-line distance to actual path length between two points; humans move in
  gentle curves (straightness noticeably < 1.0), while scripted/synthetic movement is often perfectly linear
  (straightness ≈ 1.0).
- **Turn-angle variance (SD)** — how much the direction of movement changes moment to moment; human hand tremor
  and correction produces meaningfully more variance than programmatic cursor movement.

### Feature Vector

`BehavioralFeatureExtractor.extract()` reduces every telemetry payload down to a fixed 8-dimensional NumPy vector,
with sane fallback defaults if a signal is missing (e.g. a very short password):

```python
[
  dwell_mean,     # default 95.0 ms
  dwell_sd,       # default 22.0 ms
  flight_mean,    # default 140.0 ms
  flight_sd,      # default 35.0 ms
  typing_cps,     # default 3.8 chars/sec
  pointer_speed,  # default 0.45 px/ms
  straightness,   # default 0.70
  turn_angle_sd,  # default 0.85 rad
]
```

This fixed-width vector is what every downstream model — bot detector and per-user matcher — actually consumes.
Using a compact, well-defined feature space (rather than raw event streams) keeps inference latency low and makes
the per-user model easy to train on just 7 samples.

### Tier 1 — Bot & Automation Detection

`Tier1BotDetector` runs on **every single request**, before any identity logic, and combines two layers:

**a) Unsupervised anomaly model**
- An `IsolationForest` (`contamination=0.02`, `random_state=42`) is pre-trained at server startup on a large
  synthetic population of "plausible human" behavior (`np.random.normal` around realistic dwell/flight/speed
  values), scaled with a `StandardScaler`.
- This gives the engine a general sense of what human behavioral telemetry looks like, independent of any specific
  enrolled user.

**b) Deterministic rule checks** (evaluated against the flags the frontend attaches, plus hard feature thresholds)

| Signal | Trigger | Why it matters |
|---|---|---|
| `robotic_key_interval` / `fixed_dwell_10ms` | Frontend detects near-zero variance between consecutive key events | Real humans never press keys with machine-precision, fixed timing |
| `paste_event_detected` | A `paste` DOM event fired into the password field | Legitimate users type; credential-stuffing scripts and clipboard attacks paste |
| `zero_pointer_curvature` | Mouse path straightness ≈ 1.0 | Programmatic cursor movement (e.g. Selenium `ActionChains`) draws perfectly straight lines; humans don't |
| CPS > 25 | `feature_vector[4] > 25.0` | No human sustains 25+ characters/sec |
| Dwell < 10ms | `feature_vector[0] < 10.0` | No human holds a key for under 10ms consistently |

If **any** of these trip, the request is immediately short-circuited with `status: "blocked"`,
`reason: "BOT_DETECTED"`, a low confidence score (`10.0`), and the exact violation(s) in `explainability` — the
identity/password logic never even runs for confirmed bot traffic.

### Tier 2 — Supervised Per-User Matching

`SupervisedUserBiometricMatcher` is where the "passwordless-proof identity" actually lives. Instead of a single
global model, **every enrolled user gets their own personal `RandomForestClassifier`** (`n_estimators=100`,
`max_depth=6`, `random_state=42`), trained specifically to distinguish *that user's* behavior from everyone else's.

**Why per-user models, not a shared one?** Behavioral biometrics is inherently personal — dwell time that's
perfectly normal for one person can be a red flag for another. A single shared threshold can't capture that;
independent binary classifiers, one per identity, can.

**Why Random Forest?** It handles the small-sample, tabular, mixed-scale feature space well, is fast to train
(<100ms for 7 samples), doesn't need feature scaling, and — critically for the "explainability" stretch goal — is
easy to complement with interpretable per-feature deviation reporting.

### Enrollment Pipeline

1. Frontend guides the user through **7 password-entry trials**, showing live progress (`sample_count/7`).
2. Every trial must use the **exact same password** — a mismatch anywhere in the 7 trials is rejected immediately
   with `PASSWORD_MISMATCH` so the behavioral fingerprint isn't built against inconsistent credentials.
3. Each trial's 8-D feature vector is appended to `USER_DATABASE[username]["samples"]`.
4. On the **7th sample**, `SupervisedUserBiometricMatcher.train_user_model()` runs:
   - **Positive class (label = 1):** the 7 real samples, each **Gaussian-noise-augmented 4×** (`noise = N(0, 0.03 *
     |sample|)`) to produce **35 positive training instances** — enough variation for the forest to generalize to
     natural day-to-day drift without losing the user's core signature.
   - **Negative class (label = 0):** a **200-sample synthetic background population** of generic human behavior
     (`_generate_background_impostors`), acting as a stand-in for "anyone who isn't this user."
   - The classifier is trained on the combined 235-row dataset.
   - The **mean of the 7 raw (un-augmented) samples** is stored as `baseline_mean`, used later purely for
     human-readable deviation explanations.

### Verification Pipeline

1. Confirm the user exists and has a **trained** profile (`USER_NOT_ENROLLED` otherwise).
2. Check the submitted **password** against the one recorded at enrollment (`INCORRECT_PASSWORD` if wrong — note
   this still runs *before* the behavioral check, since a wrong password should never even reach the biometric
   decision).
3. Run `SupervisedUserBiometricMatcher.verify()`:
   - `clf.predict_proba()` on the live 8-D feature vector returns the probability the forest assigns to the
     "legitimate user" class → this becomes the **match score** (0–100%).
   - In parallel, a **per-feature percentage deviation** from `baseline_mean` is computed; any feature that
     deviates **> 30%** is added to the `explainability` list with a human-readable description (feature name,
     % deviation, current vs. baseline value).
4. **Decision threshold:** `VERIFY_THRESHOLD = 80.0`. A match score **≥ 80%** → `status: "granted"`. Below that →
   `status: "blocked"`, `reason: "BIOMETRIC_MISMATCH"`, with the itemized deviations as the explanation — **even
   though the password was correct.**
5. A special `flight_rhythm_mismatch` flag (used by the frontend's "Impostor" simulation button) forcibly caps the
   score at 45% for demo purposes, guaranteeing a visible, repeatable block during live judging.

> **Key point for judges:** steps 2 and 3 are fully independent checks. A correct password with a failed
> behavioral match is **denied** — there is no OTP, email, or fallback step that could override that decision.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Backend framework** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) (ASGI) |
| **ML / numerical** | [scikit-learn](https://scikit-learn.org/) (`IsolationForest`, `RandomForestClassifier`, `StandardScaler`), [NumPy](https://numpy.org/) |
| **API layer** | REST/JSON over HTTP, CORS-enabled (`fastapi.middleware.cors`) |
| **Frontend** | Single-file HTML5 + vanilla JS + CSS (Tailwind utility classes), `<canvas>`-based live visualizations |
| **Packaging** | [PyInstaller](https://pyinstaller.org/) — bundles the app (including the HTML) into a standalone Windows `.exe` |
| **Logging** | Python `logging` module, structured `[ACCESS GRANTED]` / `[ACCESS DENIED]` / `[BOT BLOCKED]` log lines |

---

## 📂 Project Structure

```
.
├── main.py                  # FastAPI app — telemetry pipeline, bot detector, supervised matcher, routes
├── bioprint_merged.html     # Frontend — Live Studio, Intelligence dashboard, SDK Setup modal
├── BioPrintAI.spec          # PyInstaller build spec (entry point + bundled HTML resource)
├── build/                   # PyInstaller intermediate build artifacts (generated, safe to .gitignore)
├── __pycache__/             # Python bytecode cache (generated, safe to .gitignore)
└── dist/
    └── BioPrintAI.exe       # Packaged standalone executable (~100 MB, self-contained)
```

**Suggested `.gitignore` additions** if not already present:
```
__pycache__/
build/
*.pyc
```
(Keep `dist/BioPrintAI.exe` tracked, or use [Git LFS](https://git-lfs.github.com/) for it, since it's a large
binary and is part of the required deliverables.)

---

## 🚀 Getting Started

### Option 1 — Just run the executable (fastest, no setup)

No Python, no dependencies, no `pip install`. Just clone/download the repo and double-click:

```
dist/BioPrintAI.exe
```

This launches the FastAPI server locally on **http://127.0.0.1:5000** and automatically opens the app in your
default browser after ~1.2 seconds. This is the fastest way for **judges or first-time reviewers** to try the demo
— everything (backend + bundled frontend) is packed into the single `.exe`.

> If Windows SmartScreen warns about an unrecognized publisher (expected for an unsigned hackathon build), click
> **"More info" → "Run anyway."**

### Option 2 — Run from source

#### Prerequisites
- Python 3.9+
- `pip`

#### Installation

```bash
git clone <your-repo-url>
cd bioprint-ai
pip install fastapi uvicorn numpy scikit-learn
```

#### Run

```bash
python main.py
```

The app starts on **http://127.0.0.1:5000** and auto-opens in your default browser. Logs are printed to the
console with timestamps for every enrollment, grant, and denial event.

#### Build the executable yourself (optional)

```bash
pip install pyinstaller
pyinstaller BioPrintAI.spec
```

The packaged app will be produced at `dist/BioPrintAI.exe`. The `.spec` file ensures `bioprint_merged.html` is
bundled as a resource and correctly located at runtime via `get_resource_path()` (which resolves paths relative to
PyInstaller's `sys._MEIPASS` when frozen).

---

## 🖥️ Frontend Walkthrough

The UI (`bioprint_merged.html`) is organized into three navigable sections:

| Nav item | Section | What it shows |
|---|---|---|
| **Live Studio** | `#studio-section` | The actual enroll/login form — username & password fields, real-time keystroke/pointer HUD (`hud-keys`, `hud-moves`, `hud-clicks`, `hud-flags`), and the **attack simulation panel** (`🤖 Script Bot`, `📋 Paste Attack`, `👤 Impostor` buttons) |
| **Intelligence** | `#intel-section` | Live decision dashboard: confidence **gauge** (`gauge-canvas`, `gauge-score`), **risk badge** (LOW/MEDIUM/HIGH RISK), **keystroke rhythm chart** (`keystroke-canvas`), **entropy score**, **decision latency** in ms, and raw **payload size** |
| **SDK Setup** | modal (`sdk-modal`) | Integration documentation for wiring the telemetry capture script into a third-party site, with copy/download buttons for the generated JSON payload |

**Key HUD elements while typing/moving:**
- `badge-user-dynamics` — live keystroke counter
- `badge-pass-dynamics` — "Masked Dynamics" indicator confirming password content itself is never transmitted, only
  timing metadata
- `feat-dwell`, `feat-flight`, `feat-straight`, `feat-synthetic` — live feature readouts as you type/move
- `enroll-progress` / `sample-count-text` — enrollment trial progress (e.g. "4/7 samples recorded")

**Attack simulation buttons** (for live demos without needing a second physical device):
- **🤖 Script Bot** — synthetically injects `robotic_key_interval` / `fixed_dwell_10ms` flags to demonstrate Tier 1
  blocking a scripted login.
- **📋 Paste Attack** — simulates a `paste` event into the password field to demonstrate `paste_event_detected`
  blocking.
- **👤 Impostor** — simulates a genuine-but-different user's rhythm (`flight_rhythm_mismatch`) to demonstrate Tier 2
  blocking a correct-password-wrong-behavior attempt.

---

## 🔌 API Reference

All endpoints accept and return `application/json`. The server enables permissive CORS for local demo purposes.

### `GET /`
Serves the bundled frontend (`bioprint_merged.html`). Returns `404` with an inline error message if the file isn't
found next to the executable/script.

### `POST /api/enroll`
Submits one of the 7 required enrollment trials for a user. Internally calls the shared pipeline with
`mode_override="enroll"`.

### `POST /api/verify`
Verifies a login attempt against a previously enrolled profile. Internally calls the shared pipeline with
`mode_override="verify"`.

### `POST /api/bioprint/telemetry`
Generic ingestion endpoint — the `mode` (`"enroll"` / `"verify"`) is read from `session.mode` in the payload body
instead of the URL. Useful for a single unified client-side POST call.

### Response fields (common to all modes)

| Field | Type | Description |
|---|---|---|
| `status` | string | `"success"` \| `"granted"` \| `"blocked"` \| `"error"` |
| `reason` | string | Present on non-success outcomes: `BOT_DETECTED`, `PASSWORD_MISMATCH`, `USER_NOT_ENROLLED`, `INCORRECT_PASSWORD`, `BIOMETRIC_MISMATCH` |
| `is_bot` | boolean | Whether Tier 1 flagged this request as automated |
| `score` / `confidence_score` | float | 0–100 confidence/match percentage |
| `explainability` | string[] | Human-readable reasons behind the decision (present on blocks) |
| `message` | string | Human-readable summary, safe to display directly in the UI |
| `sample_count` | int | *(enroll mode only)* trials recorded so far, out of 7 |

### Example — successful enrollment trial

**Request** `POST /api/enroll`
```json
{
  "subject": { "username": "alex", "password": "Tr0ub4dor&3" },
  "session": { "mode": "enroll" },
  "keystrokes": {
    "password": {
      "derived": {
        "dwell": { "mean": 98.4, "sd": 19.2 },
        "ud":    { "mean": 145.1, "sd": 31.7 },
        "typingSpeedCps": 4.1
      }
    }
  },
  "pointer": {
    "derived": {
      "speedPxPerMs": { "mean": 0.42 },
      "straightness": { "mean": 0.68 },
      "turnAngleRad": { "sd": 0.91 }
    }
  },
  "signals": { "flags": [] }
}
```

**Response**
```json
{
  "status": "success",
  "mode": "enroll",
  "username": "alex",
  "sample_count": 3,
  "is_bot": false,
  "score": 98.0,
  "confidence_score": 98.0,
  "message": "Sample 3/7 recorded. Complete remaining attempts with the same password."
}
```

### Example — access granted

```json
{
  "status": "granted",
  "username": "alex",
  "score": 94.3,
  "confidence_score": 94.3,
  "is_bot": false,
  "message": "ACCESS GRANTED: Behavioral signature match (94.3%)"
}
```

### Example — blocked: correct password, behavioral mismatch

```json
{
  "status": "blocked",
  "reason": "BIOMETRIC_MISMATCH",
  "score": 41.2,
  "confidence_score": 41.2,
  "is_bot": false,
  "explainability": [
    "Dwell Mean deviated by 46.8% from baseline (55.0 vs 98.4)",
    "Flight Mean deviated by 38.2% from baseline (89.6 vs 145.1)"
  ],
  "message": "ACCESS DENIED: Behavioral match 41.2% is below required 80.0%"
}
```

### Example — blocked: bot / automation detected

```json
{
  "status": "blocked",
  "reason": "BOT_DETECTED",
  "is_bot": true,
  "score": 10.0,
  "confidence_score": 10.0,
  "explainability": [
    "Robotic fixed key interval detected",
    "Synthetic linear pointer path detected"
  ],
  "message": "ACCESS DENIED: Bot behavior detected (Robotic fixed key interval detected | Synthetic linear pointer path detected)"
}
```

### Error responses

| HTTP status | Cause |
|---|---|
| `400` | Malformed/non-JSON request body, or missing `username` |
| `200` (with `status: "error"` / `"blocked"`) | All *business-logic* failures (wrong password, not enrolled, bot detected, biometric mismatch) are returned as `200 OK` with a descriptive body — by design, so the frontend can render them without special-casing HTTP error handling |

---

## 📦 Telemetry Payload Schema

The frontend builds a payload roughly shaped like this before every POST (fields are read defensively on the
backend, with fallbacks, so partial payloads degrade gracefully rather than erroring):

```jsonc
{
  "subject": { "username": "string", "password": "string" },
  "session": { "mode": "enroll | verify" },
  "keystrokes": {
    "password": {
      "derived": {
        "dwell": { "mean": "number (ms)", "sd": "number (ms)" },
        "ud":    { "mean": "number (ms)", "sd": "number (ms)" },
        "typingSpeedCps": "number (chars/sec)"
      }
    }
  },
  "pointer": {
    "derived": {
      "speedPxPerMs": { "mean": "number" },
      "straightness": { "mean": "number, 0-1" },
      "turnAngleRad":  { "sd": "number" }
    }
  },
  "signals": {
    "flags": [
      "robotic_key_interval",
      "fixed_dwell_10ms",
      "paste_event_detected",
      "zero_pointer_curvature",
      "flight_rhythm_mismatch"
    ]
  }
}
```

> Note: raw keystroke *content* (the actual password characters) never needs to leave the browser for the
> behavioral engine to work — only **timing metadata** is required. The password itself is sent solely for the
> separate, traditional credential check (Step 2 of verification), matching how the extractor only ever touches
> `derived` statistics.

---

## 🧪 Demo Flow / Script

A suggested 3–5 minute live demo sequence for judges:

1. **Enroll** a new user — complete all 7 password-entry trials with natural typing/mouse behavior, showing the
   live `sample_count` progress tick up to `7/7` and the model training confirmation.
2. **Genuine login** — log in as that user with the correct password and normal behavior → **`ACCESS GRANTED`**,
   watch the confidence gauge and decision latency badge update live.
3. **Impostor attempt** — click the **👤 Impostor** simulation button (or have a teammate type the same correct
   password with a deliberately different rhythm) → **`ACCESS DENIED — BIOMETRIC_MISMATCH`**, with itemized
   per-feature deviations shown in the Intelligence panel — proving the correct password alone isn't enough.
4. **Bot attempt** — click **🤖 Script Bot** or **📋 Paste Attack** → **`ACCESS DENIED — BOT_DETECTED`**, shown as a
   distinct reason/badge from a biometric mismatch, proving bot detection is a separate signal from identity
   matching.
5. *(Optional)* Open **SDK Setup** to show how the telemetry capture script could be dropped into any third-party
   login page, and use **Download JSON** to show the judges the exact raw payload driving a decision.

---

## 🏆 Judging Criteria Alignment

| Criterion | How this project addresses it |
|---|---|
| **Creativity** | Combines keystroke *and* pointer dynamics into one 8-D fingerprint; live attack-simulation studio instead of needing a second real attacker |
| **Reliability** | Per-user supervised model trained on noise-augmented real samples vs. a synthetic impostor background population; deterministic `random_state` for reproducible demo scoring |
| **Ease of Use** | Guided 7-trial enrollment with live progress feedback; masked-dynamics badge reassures users their password content isn't being profiled |
| **Customer Satisfaction** | Transparent, explainable denials (`explainability` list) rather than an opaque "access denied" |
| **Novelty in Algorithms** | Two-tier pipeline: unsupervised anomaly screen (bots) feeding into a supervised, per-identity classifier (behavioral match) — not a single generic threshold |
| **Latency** | In-memory models, single forward pass per request; live-rendered decision latency badge in milliseconds |
| **Overall Innovation** | Fully demo-ready: live simulation buttons let judges see both attack classes blocked without needing multiple physical testers |

---

## 🔒 Security Considerations

- **No password transmission beyond what's necessary** — the behavioral engine only ever consumes *derived timing
  statistics*, never raw keystrokes; password content is used exclusively for the traditional credential check.
- **Fail-closed on ambiguity** — both `USER_NOT_ENROLLED` and `INCORRECT_PASSWORD` deny access outright rather than
  silently falling back to a weaker check.
- **No secondary-verification bypass** — by design, there is no OTP/email/SMS fallback path anywhere in
  `process_telemetry_pipeline`; a biometric mismatch is a hard deny.
- **Bot detection runs before identity logic** — automated traffic never has an opportunity to brute-force or
  probe the behavioral matcher, since it's rejected at Tier 1.
- This is a **hackathon prototype**, not a production auth system — see [Known Limitations](#-known-limitations--notes)
  for what would need to change before real-world deployment.

---

## ⚠️ Known Limitations / Notes

- `USER_DATABASE` is **in-memory only** (`Dict[str, Dict[str, Any]]`) — all enrolled users and trained models are
  lost on server restart. A production build should persist samples/models to a real datastore (SQLite,
  PostgreSQL, Redis, etc.).
- The Tier 1 bot baseline and the Tier 2 synthetic impostor background population are both `np.random`-generated
  at startup — a production system should train these on real captured human traffic instead of synthetic data.
- `VERIFY_THRESHOLD = 80.0` and the Random Forest hyperparameters (`n_estimators=100`, `max_depth=6`) are fixed
  constants tuned for demo responsiveness; a production system would tune/calibrate these per deployment.
- CORS is currently wide open (`allow_origins=["*"]`) for local demo convenience — restrict this to known origins
  before any public deployment.
- Enrollment currently requires the **exact same password** across all 7 trials; there's no password-change flow
  that re-triggers re-enrollment.
- The engine is delivered as a **website login page**, not a Chrome extension (the problem statement allowed
  either) — see the frontend's SDK Setup modal for how the same telemetry client could be adapted into an
  extension's content script.

---

## 🗺️ Roadmap / Stretch Goals

From the original problem statement, potential next steps beyond the current build:

- [ ] **Adaptive profiles** — gently update a user's baseline as their genuine behavior naturally drifts over time
      (e.g. rolling re-training on recent successful verifications).
- [ ] **Live confidence dashboard** — persist historical scores per user to visualize trend lines, not just the
      current session.
- [ ] **Multi-modality support** — unify touchpad, external mouse, and mobile touch signals into one profile.
- [ ] **Persistent storage** — replace the in-memory `USER_DATABASE` with a real database.
- [ ] **Chrome extension packaging** — port the existing telemetry client into a content script for drop-in use on
      third-party login pages.

---

## 🛠️ Troubleshooting / FAQ

**The `.exe` won't launch / Windows blocks it.**
This is an unsigned build; click "More info" → "Run anyway" in the SmartScreen prompt, or run from source instead
(Option 2 above).

**"Error: bioprint_merged.html not found" in the browser.**
Make sure `bioprint_merged.html` sits in the **same directory** as `main.py` (or, for the packaged `.exe`, that it
was correctly bundled — rebuild with `pyinstaller BioPrintAI.spec` if you've modified the HTML).

**Enrollment keeps failing with `PASSWORD_MISMATCH`.**
All 7 enrollment trials must use the **exact same password**, character for character. Restart enrollment for that
username if you need to change the password (currently there's no partial-reset endpoint — restart the server to
clear `USER_DATABASE`, or enroll under a different username).

**Verification always returns `USER_NOT_ENROLLED`.**
The user must complete all **7/7** enrollment trials — a model is only trained (and the user considered "enrolled")
once `sample_count >= 7`.

**Score seems inconsistent between runs.**
Both models use fixed `random_state=42` seeds, so scoring is deterministic for identical inputs — variation you
see between attempts reflects genuine differences in your captured typing/mouse telemetry, not model randomness.

**Port 5000 already in use.**
Edit the `uvicorn.run(...)` call at the bottom of `main.py` to use a different `port=`, and update `open_browser()`
to match.

---

## 🤝 Contributing

This was built under a 36-hour hackathon deadline — contributions, bug reports, and suggestions are welcome after
the event:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with clear messages
4. Open a pull request describing the change and why

---

## 👥 Team

- Chaitanya Gupta
- Deepak Chowdhary Nekkalapu
- Krishna Kumar Roy

---

## 🙏 Acknowledgments

Built for the **Root 36** hackathon — *Build. Break. Defend.* — under the
**BioPrint: Behavior-Based Login Security** track.
