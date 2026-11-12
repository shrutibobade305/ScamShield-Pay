# 🛡️ ScamShield Pay — AI-Powered Payment Fraud Protection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![PayPal Sandbox](https://img.shields.io/badge/PayPal-Orders_v2_Sandbox-003087.svg)](https://developer.paypal.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-52%20Passed%20(100%25)-success.svg)](tests/)

> **ScamShield Pay** is an autonomous, AI-driven payment fraud detection firewall. It intercepts online transaction requests, extracts behavioral risk signals, predicts fraud probabilities in real time using a Scikit-Learn **Random Forest Classifier (200 Decision Trees)**, and strictly gates **PayPal Sandbox Orders v2** checkout based on server-side risk decisions and cryptographic Step-Up MFA verification.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Risk Policy Corridors](#-risk-policy-corridors)
- [Technology Stack](#-technology-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Installation Guide](#-installation-guide)
- [Environment Configuration](#-environment-configuration)
- [ML Model Training & Pipeline](#-ml-model-training--pipeline)
- [Running the Application](#-running-the-application)
- [PayPal Sandbox Testing Guide](#-paypal-sandbox-testing-guide)
- [Automated Testing & QA Audit](#-automated-testing--qa-audit)
- [Deployment Guide (Render / Cloud)](#-deployment-guide-render--cloud)
- [Limitations & Future Roadmap](#-limitations--future-roadmap)

---

## 🎯 Overview

Traditional payment workflows process customer transactions without evaluating behavioral risk, exposing merchants and consumers to account takeovers, unauthorized card usage, and chargeback fraud.

**ScamShield Pay** introduces a **zero-trust, pre-settlement payment firewall**:
1. **Pre-Payment Analysis:** Before any payment gateway session is opened, payment parameters (amount, hour, geo-routing, device state, failure velocity) are evaluated.
2. **Machine Learning Inference:** A 200-tree Random Forest classifier calculates a non-linear fraud probability score ($0.0\% - 100.0\%$) with $<15\text{ ms}$ latency.
3. **Server-Side Policy Enforcement:**
   - 🟢 **Safe ($< 35\%$):** Directly authorized to create and capture a PayPal Sandbox payment.
   - 🟡 **Suspicious ($35\% - 69.9\%$):** Payment is locked under review until the user completes a time-limited Step-Up Security MFA challenge evaluated against an auditable policy.
   - 🔴 **High Risk ($\ge 70\%$):** Payment is strictly blocked on the backend. Frontend scores and tamper attempts are rejected with HTTP 403.
4. **Idempotent Payment Settlement:** Validates currency, amount, and reference IDs before capturing the order and archiving an immutable receipt to persistent SQLite/PostgreSQL.

---

## ✨ Key Features

- **⚡ Real-Time ML Fraud Inference:**
  - Evaluates 5 core behavioral features simultaneously.
  - Predicts continuous risk percentages using `predict_proba()`.
  - Serialized via Joblib with zlib compression (`models/scam_model.pkl`).
- **🔍 Explainable Risk Indicators (XAI):**
  - Generates human-understandable explanations for all 5 dimensions:
    - *Transaction Amount Tier* (Normal, Elevated, High-Value Spike)
    - *Device Hardware Fingerprint* (Trusted known device vs. Unrecognized device)
    - *Authentication Velocity* (0–1 failures vs. repeated failure anomaly)
    - *Geographic Routing* (Domestic vs. Cross-Border international transfer)
    - *Temporal Anomaly* (Standard daytime retail vs. off-peak nighttime 02:00–04:00)
  - Explicitly distinguishes contextual rule heuristics from non-linear model probabilities.
- **🔐 Server-Gated PayPal Sandbox Integration:**
  - Never trusts client-side scores, amounts, or approval states.
  - Dynamically binds PayPal `reference_id` and `custom_id` to internal transaction IDs.
  - Prevents double-charging with a deterministic idempotency key layer (`PayPal-Request-Id`).
- **🛡️ Step-Up Security Verification (MFA):**
  - Dynamic 6-digit cryptographic OTP generation.
  - Salted SHA-256 OTP hashing (plain-text OTP is never stored in the database).
  - 10-minute expiration TTL and 3-attempt lockout defense.
  - Explicit auditable policy evaluation post-verification (no blind auto-approvals).
- **📊 Real-Time Admin Intelligence Dashboard:**
  - Aggregate metrics: Total Transactions, Safe, Suspicious, High Risk, Completed, Blocked, Total Volume ($), Average Risk Score, and Approval Rate.
  - Interactive Chart.js visualizations: Risk Tier Distribution, Daily Volume Trends, Timeline Analytics, and Settlement Outcomes.
  - One-click demo dataset seeding and clearing for live hackathon presentations.
- **📜 Auditable Transaction Ledger:**
  - Paginated SQLite transaction history with live search by ID prefix.
  - Multi-parameter filtering by Risk Classification, Payment Status, and Date Range.
  - Deep inspection modal detailing model weights, decision trees, and PayPal capture IDs.
- **🎨 Dark-Mode Fintech Design System:**
  - Responsive layouts (320px mobile to 4K desktop).
  - Micro-interactions: Radar scanning inference animation, animated SVG radial gauge.

---

## 🏛️ System Architecture

```
                                 [ USER BROWSER ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
          [ HTML5 / CSS3 / JS ]                       [ PayPal SDK UI ]
                   │                                           │
                   ▼                                           ▼
       POST /predict {params}                        GET /payment?tx_id=...
                   │                                           │
                   ▼                                           ▼
      ┌─────────────────────────────────────────────────────────────┐
      │                    FLASK BACKEND ENGINE                     │
      │                                                             │
      │   1. Input Bounds & Sanity Validation                       │
      │   2. Random Forest Model Inference (models/scam_model.pkl)   │
      │   3. Heuristic Explainability Generator                     │
      │   4. Server-Side Policy Gating Decision                     │
      │   5. Persistent SQLite / PostgreSQL Audit Archival          │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
      [ Safe <35% ]         [ Suspicious 35-70% ]       [ High Risk ≥70% ]
           │                         │                         │
     Auto-Approved            Requires Step-Up         Strictly Blocked
           │                   MFA Verification                │
           │                         │                   HTTP 403 Forbidden
           │               POST /api/verify-otp                │
           │                         │                    (No PayPal Order)
           │                   Policy Passed?                  │
           │                  ┌──────┴──────┐                  │
           │                Yes             No                 │
           │                 │               │                 │
           │           Approved Post-MFA  Blocked              │
           │                 │               │                 │
           └────────┬────────┘               ▼                 ▼
                    │                  Payment Denied    Payment Denied
                    ▼
      ┌─────────────────────────────────────────────────────────────┐
      │              PAYPAL ORDERS v2 SANDBOX GATEWAY               │
      │                                                             │
      │   1. POST /create-paypal-order (Validates Server DB State)  │
      │   2. Buyer Approves Sandbox Payment                         │
      │   3. POST /capture-paypal-order (Cross-Order Verification)   │
      │   4. Idempotency Check & Ledger Finalization                │
      └─────────────────────────────────────────────────────────────┘
```

---

## 🚦 Risk Policy Corridors

| Tier | Risk Score | Initial Status | Payment Eligibility | Action Required |
| :--- | :---: | :---: | :---: | :--- |
| **Safe** | $0.0\% - 34.9\%$ | `APPROVED` | **Allowed** | Instant redirect to PayPal Sandbox Checkout. |
| **Suspicious** | $35.0\% - 69.9\%$ | `UNDER_REVIEW` | **Gated** | Requires Step-Up Security MFA and policy compliance check. |
| **High Risk** | $70.0\% - 100.0\%$ | `BLOCKED` | **Blocked (403)** | Transaction locked; payment processing strictly prevented. |

---

## 💻 Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.10+, Flask 3.0.0+ | REST API, security middleware, session handling |
| **Persistence** | SQLite / Flask-SQLAlchemy 3.1+ | Transaction audit ledger, MFA state, PayPal refs |
| **Machine Learning** | Scikit-Learn 1.4+, Pandas, NumPy | 200-tree Random Forest Classifier, stratified split |
| **Model Storage** | Joblib 1.3+ | Serialized model artifact with compression |
| **Payments** | PayPal Orders v2 REST API & JS SDK | Sandbox checkout, OAuth 2.0 token, server captures |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Responsive Fintech UI, radar animations, SVG gauge |
| **Analytics** | Chart.js 4.4 | Real-time interactive dashboard charts |
| **Production Server** | Gunicorn 21.2+ | WSGI HTTP Server for production deployment |
| **Testing** | Pytest 9.1+ | 52 automated unit, integration, and security tests |

---

## 📂 Project Directory Structure

```
ScamShield-Pay/
├── .env                       # Local environment secrets (ignored by git)
├── .env.example               # Template for configuration variables
├── .gitignore                 # Strict VCS exclusions (.env, db, pycache)
├── Procfile                   # Cloud web process entrypoint (gunicorn app:app)
├── README.md                  # Comprehensive documentation and setup guide
├── requirements.txt           # Production Python dependencies
├── app.py                     # Main Flask application and REST routes
├── database.py                # SQLAlchemy ORM Transaction model
├── train_model.py             # ML pipeline, synthetic data & model trainer
├── models/
│   └── scam_model.pkl         # Trained Random Forest serialized model artifact
├── instance/
│   └── scamshield.db          # Persistent SQLite database (local/dev)
├── static/
│   ├── script.js              # Client interactions, radar scanner, presets
│   └── style.css              # Master Fintech design system & media queries
├── templates/
│   ├── index.html             # Homepage, interactive analyzer, features
│   ├── result.html            # Radial risk meter, indicators, decisions
│   ├── dashboard.html         # Admin intelligence dashboard with Chart.js
│   ├── transactions.html      # Paginated audit ledger with search & filters
│   ├── transaction_detail.html# Deep audit inspection & model transparency
│   ├── verify.html            # Step-Up Security MFA simulator & countdown
│   └── payment.html           # Server-gated PayPal Sandbox checkout screen
└── tests/
    ├── test_app.py            # 37 core application, ML, and payment tests
    └── test_qa_audit.py       # 15 comprehensive pre-hackathon QA audit tests
```

---

## ⚙️ Installation Guide

### Prerequisites
- **Python 3.10** or higher
- **Git**
- A **PayPal Developer Account** (free at [developer.paypal.com](https://developer.paypal.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/shrutibobade305/ScamShield-Pay.git
cd ScamShield-Pay
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Environment Configuration

Create your `.env` file from the provided `.env.example`:

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` with your preferred configuration:

```env
# Flask Security
SECRET_KEY=replace_with_a_secure_random_key_e.g._secrets_token_hex_32
FLASK_DEBUG=False
PORT=5000

# Persistent Database URI
# Default (Local/Dev): sqlite:///instance/scamshield.db
# Cloud PostgreSQL: postgresql://user:password@hostname:5432/dbname
DATABASE_URL=sqlite:///instance/scamshield.db

# PayPal Sandbox API Credentials
# Get these from: https://developer.paypal.com/dashboard/applications/sandbox
PAYPAL_CLIENT_ID=your_paypal_sandbox_client_id_here
PAYPAL_CLIENT_SECRET=your_paypal_sandbox_client_secret_here

# PayPal Endpoint (api-m.sandbox.paypal.com for Sandbox)
PAYPAL_BASE_URL=https://api-m.sandbox.paypal.com

# Step-Up Verification Simulator (true for Hackathon Demo UI)
DEMO_VERIFICATION_MODE=true
```

---

## 🧠 ML Model Training & Pipeline

The machine learning model is trained on a structured synthetic behavioral dataset ($10,000$ transactions) using stratified train/test splitting ($80/20$) to eliminate data leakage.

### Run Training Pipeline
```bash
python train_model.py
```

### Evaluation Output & Metrics
```
============================================================
 SCAMSHIELD PAY - RANDOM FOREST MODEL EVALUATION REPORT
============================================================
Total Dataset Records   : 10,000
Training Split Records  : 8,000 (80%)
Test Split Records      : 2,000 (20%)
Fraud Class Prevalence  : 46.74%

--- OVERALL PERFORMANCE METRICS ---
Accuracy               : 66.05%
Precision (Class 1)    : 63.11%
Recall (Class 1)       : 65.88%
F1-Score (Class 1)     : 64.47%
ROC-AUC Score          : 72.23%

--- CONFUSION MATRIX ---
True Negatives  (Legitimate Correct) :   705
False Positives (Legitimate Flagged) :   360
False Negatives (Fraud Missed)       :   319
True Positives  (Fraud Caught)       :   616

--- FEATURE IMPORTANCE BREAKDOWN ---
  • amount              :  49.97%
  • transaction_hour    :  21.56%
  • failed_attempts     :  15.54%
  • new_device          :   8.92%
  • international       :   4.01%
============================================================
Model saved successfully at: models/scam_model.pkl (7.92 MB)
============================================================
```

---

## 🚀 Running the Application

### Development Server
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

### Production Server (WSGI via Gunicorn)
```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 3
```

---

## 🧪 PayPal Sandbox Testing Guide

To test the end-to-end payment flow in PayPal Sandbox:

1. **Log into PayPal Developer Dashboard:** Navigate to [developer.paypal.com](https://developer.paypal.com/) and go to **Testing Tools → Sandbox Accounts**.
2. **Note Sandbox Credentials:**
   - **Buyer Account:** Use the personal sandbox email and password to log into the checkout modal.
   - **App Credentials:** Copy the Sandbox `Client ID` and `Client Secret` into your `.env`.

### Testing Pre-Set Hackathon Scenarios
On the homepage ([http://127.0.0.1:5000/#analyzer](http://127.0.0.1:5000/#analyzer)), use the Quick Preset buttons:

1. 🟢 **Safe Scenario ($45.50):**
   - Click **Safe ($45.50)** → Click **Analyze Risk**.
   - Result: Risk $< 35\%$, Status `Safe`.
   - Click **Proceed to PayPal Sandbox Checkout** → Authenticate with your Sandbox Buyer account → Complete purchase.
   - Status updates to `COMPLETED` with an official capture ID.

2. 🟡 **Suspicious Scenario ($4,850.00):**
   - Click **Suspicious ($4,850)** → Click **Analyze Risk**.
   - Result: Risk $35\% - 70\%$, Status `Suspicious`.
   - Direct payment is **blocked**. Click **Complete Security Verification**.
   - View the generated demo OTP on screen, enter it, and submit.
   - Server validates OTP and auditable policy → Grants `APPROVED_POST_MFA` clearance.
   - Payment checkout is now enabled.

3. 🔴 **High Risk Scenario ($18,500.00):**
   - Click **High Risk ($18,500)** → Click **Analyze Risk**.
   - Result: Risk $\ge 70\%$, Status `High Risk`.
   - Payment is strictly **blocked**. Navigating directly to `/payment` or sending POST requests to `/create-paypal-order` returns `HTTP 403 Forbidden`.

4. 🔁 **Idempotency & Duplicate Capture Test:**
   - Submitting duplicate capture requests for a completed transaction returns `HTTP 200` with `"idempotent": true` and returns the existing capture ID without duplicate charging.

---

## 📋 Automated Testing & QA Audit

The test suite includes **52 automated unit, integration, and security tests** with 100% pass rate.

### Run All Tests
```bash
python -m pytest -v
```

### Test Suites Breakdown
- **`tests/test_app.py` (37 tests):**
  - Homepage & health check endpoints
  - Safe, Suspicious, and High Risk prediction flows & SQLite persistence
  - Form validation bounds and error responses
  - Explainable risk indicators structure
  - PayPal order creation, cancellation, failure, and capture flows
  - Dashboard KPI metrics aggregation and chart data endpoints
  - Ledger search, filtering, and pagination
  - MFA OTP request, expiration, attempt lockout, and policy re-evaluation
  - Post-MFA payment clearance flow
- **`tests/test_qa_audit.py` (15 tests):**
  - Full audit of all 15 pre-hackathon quality assurance tasks
  - Navigation links and anchor target integrity
  - ML model loading, feature alignment, and probability sum verification
  - Strict server-side blocking of unauthorized PayPal orders (HTTP 403)
  - Duplicate capture idempotency protection
  - Responsive viewport meta tag and CSS media query validation
  - Missing environment variable resilience (graceful degradation)
  - Custom 404 and 500 error handlers for HTML and JSON API requests

---

## ☁️ Deployment Guide (Render / Cloud)

The application is fully configured for deployment on [Render](https://render.com/) or any container/PaaS provider.

### Render Configuration
1. **Repository:** Connect your GitHub repository to Render as a **Web Service**.
2. **Environment:** `Python 3`
3. **Build Command:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Start Command:**
   ```bash
   gunicorn app:app
   ```
   *(Matches the included `Procfile`)*

### Environment Variables on Render
Add these variables in **Render Dashboard → Environment**:
- `SECRET_KEY`: A cryptographically secure random string (`python -c "import secrets; print(secrets.token_hex(32))"`).
- `FLASK_DEBUG`: `False`
- `PAYPAL_CLIENT_ID`: Your PayPal Sandbox Client ID.
- `PAYPAL_CLIENT_SECRET`: Your PayPal Sandbox Client Secret.
- `PAYPAL_BASE_URL`: `https://api-m.sandbox.paypal.com`
- `DEMO_VERIFICATION_MODE`: `true`

### ⚠️ Persistent Database Storage on Render
- **Default SQLite Behavior:** Standard Render web services run on ephemeral disks, meaning an SQLite file (`instance/scamshield.db`) is reset whenever the web service restarts or rebuilds.
- **Production Recommendation:**
  1. **Option A (Managed PostgreSQL):** Create a free Render PostgreSQL database and copy the Internal Database URL into the `DATABASE_URL` environment variable. `app.py` automatically detects and uses PostgreSQL:
     ```python
     DATABASE_URL=postgresql://user:password@dpg-xxxxx:5432/scamshield
     ```
  2. **Option B (Persistent Disk):** Attach a Render Persistent Disk mounted at `/opt/render/project/src/instance` to persist SQLite files across deploys.

---

## 🔮 Limitations & Future Roadmap

### Current Limitations
- **Synthetic Training Data:** The model is trained on a synthetic dataset designed to demonstrate behavioral interaction patterns. Real-world financial institutions require proprietary fraud and chargeback datasets.
- **Sandbox Latency:** Live PayPal Sandbox API responses typically incur 800ms–1500ms latency depending on network conditions.
- **Simulated SMS/Email:** In `DEMO_VERIFICATION_MODE=true`, the OTP is displayed in the simulator UI for judging convenience. In production, this connects to Twilio / SendGrid.

### Future Roadmap
- [ ] **Adaptive Online Learning:** Continuous model retraining pipeline incorporating verified merchant chargeback feedback.
- [ ] **Advanced Device Telemetry:** Client-side canvas, WebGL, and behavioral biometrics (keystroke velocity, mouse trajectories).
- [ ] **Multi-Currency Dynamic Conversion:** Real-time FX exchange rate normalization before risk evaluation.
- [ ] **WebSocket Push Alerts:** Live push notifications to compliance analysts when high-risk transactions are detected.
- [ ] **Apple Pay & Google Pay Integration:** Expand the pre-settlement firewall to modern tokenized mobile wallets.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

---

## 👥 Hackathon Team & Acknowledgements
- Developed for the **Hackathon 2026** by the **ScamShield Pay Team**.
- Built with Python, Flask, Scikit-Learn, and PayPal Developer Platform.

