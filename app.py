from datetime import datetime, timedelta
import hashlib
import logging
import math
import os
import re
import secrets
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from sqlalchemy import desc, func, or_, text
from sqlalchemy.exc import SQLAlchemyError

from database import db, Transaction

# --------------------------------------------------
# 1. CONFIGURATION & LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger("scamshield")

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

# Secret key handling
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    logger.warning("SECRET_KEY not found in .env; generating a temporary session key.")
    secret_key = secrets.token_hex(32)

app.secret_key = secret_key

# Database Configuration (SQLite default, PostgreSQL via DATABASE_URL on Render)
db_url = os.getenv("DATABASE_URL", "").strip()
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url or f"sqlite:///{BASE_DIR / 'instance' / 'scamshield.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Ensure instance folder exists
os.makedirs(BASE_DIR / "instance", exist_ok=True)

# Initialize Flask-SQLAlchemy
db.init_app(app)

with app.app_context():
    # create_all creates tables if they don't exist; does NOT delete existing records on restart
    db.create_all()

    # Ensure secondary verification schema columns exist in SQLite if table was already created
    if "sqlite" in app.config["SQLALCHEMY_DATABASE_URI"]:
        try:
            existing_cols = [
                row[1] for row in db.session.execute(text("PRAGMA table_info(transactions)")).fetchall()
            ]
            new_column_definitions = [
                ("verification_status", "VARCHAR(32) NOT NULL DEFAULT 'NONE'"),
                ("otp_code_hash", "VARCHAR(128)"),
                ("otp_expires_at", "DATETIME"),
                ("otp_attempts_left", "INTEGER NOT NULL DEFAULT 3"),
                ("otp_requested_at", "DATETIME"),
                ("verification_notes", "VARCHAR(255)")
            ]
            for col_name, col_sql in new_column_definitions:
                if col_name not in existing_cols:
                    db.session.execute(text(f"ALTER TABLE transactions ADD COLUMN {col_name} {col_sql}"))
                    logger.info("Migrated SQLite schema: Added column %s to transactions table", col_name)
            db.session.commit()
        except Exception as mig_err:
            db.session.rollback()
            logger.warning("Table schema migration check notice: %s", mig_err)

    logger.info("Database initialized successfully at %s", app.config["SQLALCHEMY_DATABASE_URI"])

# Demo Mode Configuration
# If true (default when third-party SMS/Email gateway is not connected), the server provides
# a simulated OTP challenge visible in the verification simulator UI for testing & judging.
DEMO_VERIFICATION_MODE = os.getenv("DEMO_VERIFICATION_MODE", "true").lower() in ("true", "1", "yes")
logger.info("Step-Up Verification Demo Mode: %s", DEMO_VERIFICATION_MODE)

# Machine Learning Model Loading
MODEL_PATH = BASE_DIR / "models" / "scam_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Please run: python train_model.py"
    )

logger.info("Loading machine learning model from %s ...", MODEL_PATH)
model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = [
    "amount",
    "transaction_hour",
    "international",
    "new_device",
    "failed_attempts"
]
logger.info("Model loaded successfully. Features: %s", FEATURE_COLUMNS)

# PayPal Configuration
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "").strip()
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
PAYPAL_BASE_URL = os.getenv(
    "PAYPAL_BASE_URL",
    "https://api-m.sandbox.paypal.com"
).strip().rstrip("/")
CURRENCY = "USD"

if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
    logger.warning(
        "PayPal credentials not configured in .env. PayPal payments will fail until configured."
    )

# Legacy alias kept for backward compatibility (no routes rely on it)
approved_transactions = {}


# --------------------------------------------------
# 2. VALIDATION & EXPLAINABILITY HELPERS
# --------------------------------------------------

def is_valid_transaction_id(tx_id):
    """Validates that a transaction ID string is a well-formed token."""
    if not isinstance(tx_id, str):
        return False
    clean_id = tx_id.strip()
    return bool(re.match(r"^[A-Za-z0-9_-]{10,64}$", clean_id))


def is_valid_paypal_order_id(order_id):
    """Validates that a PayPal order ID is a well-formed token."""
    if not isinstance(order_id, str):
        return False
    clean_id = order_id.strip()
    return bool(re.match(r"^[A-Z0-9]{10,32}$", clean_id))


TRANSACTION_EXPIRATION_HOURS = 3


def is_transaction_expired(transaction):
    """Checks whether a transaction has exceeded its checkout validity window."""
    if not transaction or not transaction.created_at:
        return False
    age = datetime.utcnow() - transaction.created_at
    return age > timedelta(hours=TRANSACTION_EXPIRATION_HOURS)


def generate_secure_otp():
    """Generates a cryptographically random 6-digit numeric string."""
    return f"{secrets.randbelow(900000) + 100000:06d}"


def hash_otp(otp_code):
    """Computes SHA-256 hash of an OTP string."""
    return hashlib.sha256(str(otp_code).strip().encode("utf-8")).hexdigest()


def evaluate_verification_policy(transaction):
    """
    Explicit, auditable risk re-evaluation policy for suspicious transactions.

    IMPORTANT AUDITING REQUIREMENT:
    Do not automatically approve a suspicious transaction just because OTP verification succeeds.
    Require an explicit, auditable re-evaluation policy:
    1. The transaction must have been classified in the Suspicious corridor (35% <= risk < 70%).
       High-risk transactions (>= 70%) are NEVER approved via step-up verification.
    2. Velocity check: Prior failed attempts must not exceed critical anomaly threshold (<= 5).
    3. State check: Transaction must be in an active review state ('UNDER_REVIEW').
    4. Time check: The transaction must not be expired.

    Returns:
        dict: {'approved': bool, 'reason': str, 'notes': str}
    """
    if not transaction:
        return {
            "approved": False,
            "reason": "Transaction record not found.",
            "notes": "Policy rejection: null transaction reference."
        }

    # Policy Rule 1: High Risk transactions cannot be overridden by OTP
    if transaction.risk_classification != "Suspicious" or transaction.risk_score < 35.0 or transaction.risk_score >= 70.0:
        return {
            "approved": False,
            "reason": f"Transaction risk tier '{transaction.risk_classification}' ({transaction.risk_score:.2f}%) is outside the allowable MFA re-evaluation policy bounds.",
            "notes": f"Policy rejection: Risk score {transaction.risk_score:.2f}% is outside the review corridor (35%-70%)."
        }

    # Policy Rule 2: Brute force / velocity check
    if transaction.failed_attempts > 5:
        return {
            "approved": False,
            "reason": f"Excessive velocity anomaly detected ({transaction.failed_attempts} prior failed attempts). Requires compliance officer review.",
            "notes": f"Policy rejection: velocity threshold exceeded ({transaction.failed_attempts} > 5)."
        }

    # Policy Rule 3: Valid active review state
    if transaction.transaction_status not in ("UNDER_REVIEW", "PENDING"):
        return {
            "approved": False,
            "reason": f"Transaction state '{transaction.transaction_status}' is invalid for step-up clearance.",
            "notes": f"Policy rejection: unexpected transaction status '{transaction.transaction_status}'."
        }

    return {
        "approved": True,
        "reason": "Secondary authentication verified. Risk re-evaluation policy criteria fully satisfied.",
        "notes": f"MFA Verified: OTP matched within TTL. Policy satisfied for Suspicious tier (Risk: {transaction.risk_score:.2f}%). Clearance granted for PayPal Sandbox checkout."
    }


def is_transaction_authorized_for_payment(transaction):
    """
    Server-controlled payment eligibility evaluation.
    Never trusts frontend approval status or client scores.
    Authorized ONLY IF:
      1. Safe transaction: risk_classification == 'Safe', risk_score < 35.0, status == 'APPROVED'
      OR
      2. Verified Suspicious transaction: risk_classification == 'Suspicious', status == 'APPROVED_POST_MFA',
         and verification_status == 'VERIFIED'
    All High Risk transactions (>= 70%) are strictly disallowed under all circumstances.
    """
    if not transaction:
        return False
    if (
        transaction.risk_classification == "Safe"
        and transaction.risk_score < 35.0
        and transaction.transaction_status == "APPROVED"
    ):
        return True
    if (
        transaction.risk_classification == "Suspicious"
        and transaction.transaction_status == "APPROVED_POST_MFA"
        and transaction.verification_status == "VERIFIED"
    ):
        return True
    return False


def analyze_risk_indicators(amount, transaction_hour, international, new_device, failed_attempts):
    """
    Evaluates observable transaction attributes against standard operational security heuristics.

    NOTE ON EXPLAINABILITY (XAI):
    These indicators represent contextual rule-based risk factors observed in the transaction
    details. They provide human-readable operational visibility and must be distinguished from
    the non-linear ensemble predictions generated by the Random Forest model.
    """
    indicators = []

    # 1. Transaction Amount Analysis
    if amount >= 20000.0:
        indicators.append({
            "category": "Transaction Amount",
            "level": "critical",
            "icon": "🚨",
            "badge": "High-Value Spike",
            "title": "Unusually High Transaction Amount",
            "description": f"${amount:,.2f} USD significantly exceeds standard consumer transfer thresholds ($20,000 threshold).",
            "metric": f"${amount:,.2f} USD"
        })
    elif amount >= 5000.0:
        indicators.append({
            "category": "Transaction Amount",
            "level": "warning",
            "icon": "⚠️",
            "badge": "Elevated Value",
            "title": "Elevated Transaction Amount",
            "description": f"${amount:,.2f} USD is higher than average consumer purchase volumes ($5,000 - $20,000 range).",
            "metric": f"${amount:,.2f} USD"
        })
    else:
        indicators.append({
            "category": "Transaction Amount",
            "level": "positive",
            "icon": "✅",
            "badge": "Standard Value",
            "title": "Normal Transaction Amount",
            "description": f"${amount:,.2f} USD is within normal consumer retail payment limits (< $5,000).",
            "metric": f"${amount:,.2f} USD"
        })

    # 2. Device Identity & Hardware Fingerprint
    if new_device == 1:
        indicators.append({
            "category": "Device Verification",
            "level": "warning",
            "icon": "📱",
            "badge": "New Fingerprint",
            "title": "Unrecognized Device Detected",
            "description": "Payment initiated from a device, browser, or hardware profile not previously linked to this user.",
            "metric": "New Device"
        })
    else:
        indicators.append({
            "category": "Device Verification",
            "level": "positive",
            "icon": "✅",
            "badge": "Trusted Device",
            "title": "Known Device Profile",
            "description": "Transaction initiated from a recognized device fingerprint with historical consistency.",
            "metric": "Recognized Device"
        })

    # 3. Prior Authentication & Payment Attempts
    if failed_attempts >= 3:
        indicators.append({
            "category": "Authentication History",
            "level": "critical",
            "icon": "🔐",
            "badge": "Velocity Anomaly",
            "title": "Multiple Prior Failed Attempts",
            "description": f"{failed_attempts} failed payment or authentication attempts recorded immediately prior to this transaction.",
            "metric": f"{failed_attempts} Failed Attempts"
        })
    elif failed_attempts > 0:
        indicators.append({
            "category": "Authentication History",
            "level": "warning",
            "icon": "⚠️",
            "badge": "Minor Velocity",
            "title": "Prior Failed Attempt Recorded",
            "description": f"{failed_attempts} prior unsuccessful attempt detected before submission.",
            "metric": f"{failed_attempts} Failed Attempt"
        })
    else:
        indicators.append({
            "category": "Authentication History",
            "level": "positive",
            "icon": "✅",
            "badge": "Clean Velocity",
            "title": "Clean Attempt History",
            "description": "No prior failed attempts recorded. First-pass transaction initiation.",
            "metric": "0 Failed Attempts"
        })

    # 4. Geographic Routing
    if international == 1:
        indicators.append({
            "category": "Geographic Origin",
            "level": "warning",
            "icon": "🌐",
            "badge": "Cross-Border",
            "title": "International Transaction Detected",
            "description": "Payment involves cross-border entity routing, which historically presents higher fraud exposure.",
            "metric": "International"
        })
    else:
        indicators.append({
            "category": "Geographic Origin",
            "level": "positive",
            "icon": "✅",
            "badge": "Domestic",
            "title": "Domestic Transaction Routing",
            "description": "Transaction originates within the local domestic financial jurisdiction.",
            "metric": "Domestic"
        })

    # 5. Transaction Hour / Timing
    if transaction_hour <= 4 or transaction_hour >= 23:
        indicators.append({
            "category": "Temporal Anomaly",
            "level": "warning",
            "icon": "🌙",
            "badge": "Off-Peak Time",
            "title": "Unusual Transaction Timing",
            "description": f"Attempted at {transaction_hour:02d}:00 (late-night / off-peak hours when unauthorized takeovers peak).",
            "metric": f"{transaction_hour:02d}:00 UTC"
        })
    else:
        indicators.append({
            "category": "Temporal Anomaly",
            "level": "positive",
            "icon": "☀️",
            "badge": "Standard Hours",
            "title": "Normal Daytime Transaction Hour",
            "description": f"Attempted at {transaction_hour:02d}:00 during normal active daytime business hours.",
            "metric": f"{transaction_hour:02d}:00 UTC"
        })

    return indicators


def get_recommended_action(decision, risk_score):
    """Returns the recommended operational security action based on risk tier."""
    if decision == "approved":
        return {
            "title": "Authorize Standard Payment",
            "severity": "safe",
            "summary": "Low risk profile detected. Automated clearance granted.",
            "guidance": "The transaction satisfies all security baseline checks (Risk < 35%). The customer is cleared to proceed directly to PayPal Sandbox checkout.",
            "badge": "AUTO-APPROVED"
        }
    elif decision == "review":
        return {
            "title": "Hold for Step-Up Verification (MFA Required)",
            "severity": "suspicious",
            "summary": "Moderate risk signals detected. Checkout held pending secondary verification.",
            "guidance": "Payment is held within the review corridor (35% to 70% risk). Recommended operational protocol: Issue a secondary OTP/SMS multi-factor challenge or conduct manual review before releasing payment.",
            "badge": "CHALLENGE / REVIEW"
        }
    else:
        return {
            "title": "Decline & Block Transaction",
            "severity": "danger",
            "summary": "Severe fraud probability detected. Immediate payment blocked.",
            "guidance": "High probability of fraudulent activity (Risk ≥ 70%). Payment processing is strictly prohibited to protect account funds and prevent chargebacks. Recommended action: Lock session and notify account owner.",
            "badge": "STRICTLY BLOCKED"
        }


def get_paypal_access_token():
    """Retrieve an OAuth 2.0 access token from PayPal Sandbox."""
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise RuntimeError("PayPal credentials are not configured on the server.")

    response = requests.post(
        f"{PAYPAL_BASE_URL}/v1/oauth2/token",
        auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
        headers={
            "Accept": "application/json",
            "Accept-Language": "en_US"
        },
        data={"grant_type": "client_credentials"},
        timeout=20
    )

    if not response.ok:
        logger.error(
            "PayPal OAuth token request failed: %s - %s",
            response.status_code,
            response.text
        )
        response.raise_for_status()

    token_data = response.json()
    token = token_data.get("access_token")
    if not token:
        raise RuntimeError("PayPal did not return an access token.")

    return token


# --------------------------------------------------
# 3. CORE APPLICATION ROUTES
# --------------------------------------------------

@app.route("/")
def index():
    """Homepage: Displays marketing hero and transaction risk analyzer."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint for container/cloud deployment monitoring."""
    db_connected = True
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as err:
        logger.warning("Database connectivity check failed: %s", err)
        db_connected = False

    return jsonify({
        "status": "healthy" if db_connected else "degraded",
        "database": "connected" if db_connected else "disconnected",
        "database_backend": "PostgreSQL" if "postgresql" in app.config["SQLALCHEMY_DATABASE_URI"] else "SQLite",
        "model_loaded": model is not None,
        "paypal_configured": bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET),
        "environment": os.getenv("FLASK_ENV", "production")
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Evaluates transaction risk using the Random Forest Classifier and stores every
    analyzed transaction persistently in the database.

    Risk Thresholds:
      - Safe: Risk < 35% -> Status 'APPROVED', Payment 'PENDING' (Eligible for PayPal).
      - Suspicious: 35% <= Risk < 70% -> Status 'UNDER_REVIEW', Payment 'BLOCKED_SUSPICIOUS'.
      - High Risk: Risk >= 70% -> Status 'BLOCKED', Payment 'BLOCKED_HIGH_RISK'.
    """
    is_json = request.is_json

    try:
        # Extract fields from either JSON or Form data
        if is_json:
            data = request.get_json(silent=True) or {}
            raw_amount = data.get("amount")
            raw_hour = data.get("transaction_hour")
            raw_intl = data.get("international", 0)
            raw_device = data.get("new_device", 0)
            raw_fails = data.get("failed_attempts", 0)
        else:
            raw_amount = request.form.get("amount")
            raw_hour = request.form.get("transaction_hour")
            raw_intl = request.form.get("international", 0)
            raw_device = request.form.get("new_device", 0)
            raw_fails = request.form.get("failed_attempts", 0)

        # Input validation
        if raw_amount is None or raw_amount == "":
            raise ValueError("Transaction amount is required.")

        amount = float(raw_amount)
        if amount <= 0 or amount > 10_000_000:
            raise ValueError("Amount must be between $0.01 and $10,000,000.")

        if raw_hour is None or raw_hour == "":
            transaction_hour = 12
        else:
            transaction_hour = int(raw_hour)
            if not 0 <= transaction_hour <= 23:
                raise ValueError("Transaction hour must be between 0 and 23.")

        international = int(raw_intl)
        if international not in (0, 1):
            raise ValueError("International transaction flag must be 0 or 1.")

        new_device = int(raw_device)
        if new_device not in (0, 1):
            raise ValueError("New device flag must be 0 or 1.")

        failed_attempts = int(raw_fails)
        if failed_attempts < 0 or failed_attempts > 20:
            raise ValueError("Failed attempts must be between 0 and 20.")

        # Prepare feature DataFrame matching exact model feature schema
        features_df = pd.DataFrame([{
            "amount": amount,
            "transaction_hour": transaction_hour,
            "international": international,
            "new_device": new_device,
            "failed_attempts": failed_attempts
        }], columns=FEATURE_COLUMNS)

        # 1. Model inference: Random Forest risk probability
        probabilities = model.predict_proba(features_df)[0]
        classes = list(model.classes_)

        if 1 in classes:
            fraud_index = classes.index(1)
            risk_score = float(probabilities[fraud_index]) * 100.0
        else:
            risk_score = 0.0

        risk_score = round(risk_score, 2)

        # 2. Strict Risk Classification Thresholds
        if risk_score >= 70.0:
            status = "High Risk"
            decision = "blocked"
            tx_status = "BLOCKED"
            pay_status = "BLOCKED_HIGH_RISK"
        elif risk_score >= 35.0:
            status = "Suspicious"
            decision = "review"
            tx_status = "UNDER_REVIEW"
            pay_status = "BLOCKED_SUSPICIOUS"
        else:
            status = "Safe"
            decision = "approved"
            tx_status = "APPROVED"
            pay_status = "PENDING"

        # 3. Contextual Risk Factor Analysis (Rule-Based Heuristics)
        risk_indicators = analyze_risk_indicators(
            amount=amount,
            transaction_hour=transaction_hour,
            international=international,
            new_device=new_device,
            failed_attempts=failed_attempts
        )

        # 4. Recommended Security Action
        recommended_action = get_recommended_action(decision, risk_score)

        transaction_id = secrets.token_urlsafe(16)

        # 5. Persist EVERY analyzed transaction into the database
        transaction_record = Transaction(
            transaction_id=transaction_id,
            amount=amount,
            currency=CURRENCY,
            transaction_hour=transaction_hour,
            international=international,
            new_device=new_device,
            failed_attempts=failed_attempts,
            risk_score=risk_score,
            risk_classification=status,
            transaction_status=tx_status,
            payment_status=pay_status
        )

        try:
            db.session.add(transaction_record)
            db.session.commit()
            logger.info(
                "Persisted transaction %s to database (Class: %s, Risk: %.2f%%, Status: %s)",
                transaction_id,
                status,
                risk_score,
                tx_status
            )
        except SQLAlchemyError as db_err:
            db.session.rollback()
            logger.exception("Database error persisting transaction: %s", db_err)
            raise RuntimeError("Database error while saving transaction record.")

        result = {
            "amount": amount,
            "risk_score": risk_score,
            "status": status,
            "decision": decision,
            "transaction_id": transaction_id,
            "transaction_hour": transaction_hour,
            "international": international,
            "new_device": new_device,
            "failed_attempts": failed_attempts,
            "risk_indicators": risk_indicators,
            "recommended_action": recommended_action,
            "model_metadata": {
                "name": "Random Forest Classifier",
                "trees": 200,
                "evaluated_features": FEATURE_COLUMNS,
                "target_class": "Fraud (1)",
                "confidence_tier": "High" if (risk_score >= 80.0 or risk_score <= 20.0) else "Moderate"
            },
            "explanation_disclaimer": (
                "Notice on Explainability: The risk percentage is generated by the Random Forest model "
                "evaluating multi-feature interactions. Contextual risk indicators represent rule-based "
                "behavioral heuristics observed in the transaction details; they provide transparency and "
                "are distinct from direct causal claims of model branch activations."
            )
        }

        if is_json:
            return jsonify(result)

        return render_template("result.html", result=result)

    except (ValueError, TypeError) as val_err:
        logger.warning("Invalid input submitted: %s", val_err)
        if is_json:
            return jsonify({"error": str(val_err)}), 400
        return render_template("result.html", result={
            "status": "Invalid Input",
            "decision": "error",
            "message": str(val_err)
        }), 400

    except Exception as exc:
        logger.exception("Unexpected error in prediction: %s", exc)
        if is_json:
            return jsonify({"error": "Unable to process transaction assessment."}), 500
        return render_template("result.html", result={
            "status": "Error",
            "decision": "error",
            "message": "An error occurred while evaluating transaction risk. Please try again."
        }), 500


# --------------------------------------------------
# 3.5 STEP-UP SECURITY VERIFICATION (SUSPICIOUS MFA)
# --------------------------------------------------

@app.route("/verify/<transaction_id>")
def verify_transaction(transaction_id):
    """
    Renders Step-Up Identity Verification page for Suspicious transactions.

    Security Gating:
    - Safe transactions (<35%) bypass MFA and are redirected directly to /payment.
    - Suspicious transactions (35%-70%) are presented with the MFA challenge.
    - High-risk transactions (>=70%) are strictly prohibited from step-up verification.
    - Expired, locked, or completed transactions are handled with clear informative states.
    """
    clean_tx_id = (transaction_id or "").strip()
    if not clean_tx_id or not is_valid_transaction_id(clean_tx_id):
        return render_template("result.html", result={
            "status": "Invalid Transaction ID",
            "decision": "error",
            "message": "The transaction identifier provided is malformed or invalid."
        }), 400

    try:
        transaction = Transaction.query.filter_by(transaction_id=clean_tx_id).first()
    except SQLAlchemyError as db_err:
        logger.exception("Database error loading transaction %s for verification: %s", clean_tx_id, db_err)
        return render_template("result.html", result={
            "status": "Database Error",
            "decision": "error",
            "message": "Unable to load transaction records due to a database error."
        }), 500

    if not transaction:
        return render_template("result.html", result={
            "status": "Transaction Not Found",
            "decision": "error",
            "message": f"Transaction '{clean_tx_id}' was not found in the persistent ledger."
        }), 404

    # Expiration check
    if is_transaction_expired(transaction):
        if transaction.payment_status not in ("COMPLETED", "EXPIRED"):
            try:
                transaction.payment_status = "EXPIRED"
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
        return render_template("result.html", result={
            "status": "Transaction Expired",
            "decision": "error",
            "message": "This transaction assessment has expired (validity window: 3 hours). Please submit a new transaction analysis."
        }), 410

    # Safe transactions do not need step-up verification; redirect directly to payment
    if transaction.risk_classification == "Safe" and transaction.risk_score < 35.0:
        return redirect(url_for("payment", transaction_id=transaction.transaction_id))

    # High Risk transactions cannot be verified
    if transaction.risk_classification == "High Risk" or transaction.risk_score >= 70.0:
        return render_template("result.html", result={
            "status": "Verification Not Permitted",
            "decision": "blocked",
            "risk_score": transaction.risk_score,
            "amount": transaction.amount,
            "transaction_id": transaction.transaction_id,
            "message": f"Transaction risk tier is High Risk ({transaction.risk_score:.2f}%). Secondary verification is strictly prohibited to prevent financial compromise."
        }), 403

    # Already verified transactions can proceed directly
    if transaction.verification_status == "VERIFIED" and transaction.transaction_status == "APPROVED_POST_MFA":
        return redirect(url_for("payment", transaction_id=transaction.transaction_id))

    # Render verification screen
    indicators = analyze_risk_indicators(
        amount=transaction.amount,
        transaction_hour=transaction.transaction_hour,
        international=transaction.international,
        new_device=transaction.new_device,
        failed_attempts=transaction.failed_attempts
    )

    return render_template(
        "verify.html",
        transaction=transaction,
        indicators=indicators,
        demo_mode=DEMO_VERIFICATION_MODE,
        attempts_left=transaction.otp_attempts_left,
        verification_status=transaction.verification_status
    )


@app.route("/api/verification/request-otp", methods=["POST"])
def api_request_otp():
    """
    Generates a secure 6-digit OTP on the server and stores its SHA-256 hash in SQLite.
    Never stores or exposes plaintext OTP in production responses.

    In Demo Mode (DEMO_VERIFICATION_MODE=True), returns demo_code in the JSON payload
    so evaluators can test the verification workflow immediately without external SMS.
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_tx_id = (data.get("transaction_id") or "").strip()

        if not raw_tx_id or not is_valid_transaction_id(raw_tx_id):
            return jsonify({"error": "Invalid or missing transaction_id.", "status": "error"}), 400

        transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
        if not transaction:
            return jsonify({"error": "Transaction not found.", "status": "error"}), 404

        if is_transaction_expired(transaction):
            return jsonify({"error": "Transaction assessment has expired.", "status": "error"}), 410

        if transaction.risk_classification == "High Risk" or transaction.risk_score >= 70.0:
            return jsonify({
                "error": "High Risk transactions are not eligible for step-up verification.",
                "status": "error"
            }), 403

        if transaction.risk_classification == "Safe" and transaction.risk_score < 35.0:
            return jsonify({
                "status": "already_verified",
                "message": "Safe transactions are already cleared.",
                "redirect_url": url_for("payment", transaction_id=transaction.transaction_id)
            }), 200

        if transaction.verification_status == "VERIFIED" and transaction.transaction_status == "APPROVED_POST_MFA":
            return jsonify({
                "status": "already_verified",
                "message": "Transaction has already been verified.",
                "redirect_url": url_for("payment", transaction_id=transaction.transaction_id)
            }), 200

        if transaction.otp_attempts_left <= 0 or transaction.verification_status == "FAILED":
            return jsonify({
                "status": "locked",
                "error": "Maximum verification attempts exceeded. This transaction has been locked.",
                "attempts_left": 0
            }), 403

        # Generate secure random 6-digit OTP
        otp_code = generate_secure_otp()
        otp_hash = hash_otp(otp_code)
        expires_at = datetime.utcnow() + timedelta(minutes=10)

        transaction.otp_code_hash = otp_hash
        transaction.otp_expires_at = expires_at
        transaction.otp_requested_at = datetime.utcnow()
        transaction.verification_status = "PENDING"
        db.session.commit()

        logger.info(
            "Generated step-up OTP challenge for suspicious transaction %s (expires in 10m). Demo mode: %s",
            transaction.transaction_id,
            DEMO_VERIFICATION_MODE
        )

        resp = {
            "status": "success",
            "message": "Verification code generated and dispatched.",
            "transaction_id": transaction.transaction_id,
            "expires_in_seconds": 600,
            "attempts_left": transaction.otp_attempts_left,
            "demo_mode": DEMO_VERIFICATION_MODE,
            "masked_destination": "+1 (•••) •••-4921 / user•••@example.com"
        }

        # In demo mode, provide the code to the frontend simulator box
        if DEMO_VERIFICATION_MODE:
            resp["demo_code"] = otp_code
            resp["demo_notice"] = "DEMO SIMULATOR ACTIVE: Use this code to test verification without real cellular SMS."

        return jsonify(resp), 200

    except SQLAlchemyError as err:
        db.session.rollback()
        logger.exception("Database error during OTP generation: %s", err)
        return jsonify({"error": "Database error while generating verification challenge.", "status": "error"}), 500
    except Exception as exc:
        logger.exception("Unexpected error requesting OTP: %s", exc)
        return jsonify({"error": "Internal server error.", "status": "error"}), 500


@app.route("/api/verification/verify-otp", methods=["POST"])
def api_verify_otp():
    """
    Validates submitted 6-digit OTP against the stored SHA-256 hash.
    Enforces TTL expiration, attempt rate-limiting, and the explicit auditable re-evaluation policy.

    Upon success, updates transaction_status to 'APPROVED_POST_MFA', verification_status to 'VERIFIED',
    and payment_status to 'PENDING', releasing the payment hold.
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_tx_id = (data.get("transaction_id") or "").strip()
        raw_otp = str(data.get("otp") or "").strip()

        if not raw_tx_id or not is_valid_transaction_id(raw_tx_id):
            return jsonify({"error": "Invalid or missing transaction_id.", "status": "error"}), 400

        if not re.match(r"^\d{6}$", raw_otp):
            return jsonify({"error": "Verification code must be exactly 6 digits.", "status": "error"}), 400

        transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
        if not transaction:
            return jsonify({"error": "Transaction not found.", "status": "error"}), 404

        if is_transaction_expired(transaction):
            return jsonify({"error": "Transaction assessment has expired.", "status": "error"}), 410

        if transaction.risk_classification == "High Risk" or transaction.risk_score >= 70.0:
            return jsonify({"error": "High Risk transactions cannot be verified.", "status": "error"}), 403

        if transaction.verification_status == "VERIFIED" and transaction.transaction_status == "APPROVED_POST_MFA":
            return jsonify({
                "status": "success",
                "verified": True,
                "message": "Transaction already verified.",
                "redirect_url": url_for("payment", transaction_id=transaction.transaction_id)
            }), 200

        # Check lockout
        if transaction.otp_attempts_left <= 0 or transaction.verification_status == "FAILED":
            return jsonify({
                "status": "locked",
                "error": "Maximum verification attempts exceeded. Transaction is permanently locked.",
                "attempts_left": 0
            }), 403

        # Check expiration of current OTP
        now = datetime.utcnow()
        if not transaction.otp_expires_at or transaction.otp_expires_at < now:
            transaction.verification_status = "EXPIRED"
            db.session.commit()
            return jsonify({
                "status": "expired",
                "error": "Verification code has expired. Please request a new code.",
                "attempts_left": transaction.otp_attempts_left
            }), 400

        # Verify cryptographic hash of submitted OTP
        submitted_hash = hash_otp(raw_otp)
        if secrets.compare_digest(submitted_hash, transaction.otp_code_hash or ""):
            # Code matches! Now execute explicit auditable re-evaluation policy
            policy = evaluate_verification_policy(transaction)

            if policy["approved"]:
                transaction.verification_status = "VERIFIED"
                transaction.transaction_status = "APPROVED_POST_MFA"
                transaction.payment_status = "PENDING"
                transaction.verification_notes = policy["notes"]
                db.session.commit()

                logger.info(
                    "Transaction %s passed MFA verification and risk policy re-evaluation.",
                    transaction.transaction_id
                )

                return jsonify({
                    "status": "success",
                    "verified": True,
                    "message": "Identity verification successful! Transaction cleared for PayPal Sandbox checkout.",
                    "redirect_url": url_for("payment", transaction_id=transaction.transaction_id),
                    "policy_notes": transaction.verification_notes
                }), 200
            else:
                transaction.verification_status = "FAILED"
                transaction.transaction_status = "BLOCKED"
                transaction.payment_status = "BLOCKED_SUSPICIOUS"
                transaction.verification_notes = policy["notes"]
                db.session.commit()

                logger.warning(
                    "Transaction %s matched OTP but was rejected by policy: %s",
                    transaction.transaction_id,
                    policy["reason"]
                )

                return jsonify({
                    "status": "blocked",
                    "verified": False,
                    "error": policy["reason"],
                    "policy_notes": transaction.verification_notes
                }), 403

        else:
            # Code mismatch! Decrement attempts counter
            transaction.otp_attempts_left = max(0, transaction.otp_attempts_left - 1)

            if transaction.otp_attempts_left == 0:
                transaction.verification_status = "FAILED"
                transaction.transaction_status = "BLOCKED"
                transaction.payment_status = "BLOCKED_SUSPICIOUS"
                transaction.verification_notes = "MFA locked: Maximum invalid OTP attempts exceeded."
                db.session.commit()

                logger.warning(
                    "Transaction %s locked after 3 failed OTP attempts.",
                    transaction.transaction_id
                )

                return jsonify({
                    "status": "locked",
                    "attempts_left": 0,
                    "error": "Maximum verification attempts exceeded. This transaction has been locked."
                }), 403
            else:
                db.session.commit()
                logger.warning(
                    "Transaction %s invalid OTP attempt (%d attempts remaining)",
                    transaction.transaction_id,
                    transaction.otp_attempts_left
                )

                return jsonify({
                    "status": "invalid",
                    "attempts_left": transaction.otp_attempts_left,
                    "error": f"Invalid verification code. {transaction.otp_attempts_left} attempt{'s' if transaction.otp_attempts_left != 1 else ''} remaining."
                }), 400

    except SQLAlchemyError as err:
        db.session.rollback()
        logger.exception("Database error verifying OTP: %s", err)
        return jsonify({"error": "Database error during verification.", "status": "error"}), 500
    except Exception as exc:
        logger.exception("Unexpected error verifying OTP: %s", exc)
        return jsonify({"error": "Internal server error.", "status": "error"}), 500


@app.route("/api/verification/status/<transaction_id>")
def api_verification_status(transaction_id):
    """Returns current verification state and eligibility for a transaction."""
    clean_tx_id = (transaction_id or "").strip()
    if not clean_tx_id or not is_valid_transaction_id(clean_tx_id):
        return jsonify({"error": "Invalid transaction ID format.", "status": "error"}), 400

    transaction = Transaction.query.filter_by(transaction_id=clean_tx_id).first()
    if not transaction:
        return jsonify({"error": "Transaction not found.", "status": "error"}), 404

    is_auth = is_transaction_authorized_for_payment(transaction)
    now = datetime.utcnow()
    expires_in = max(0, int((transaction.otp_expires_at - now).total_seconds())) if transaction.otp_expires_at else 0

    return jsonify({
        "status": "success",
        "transaction_id": transaction.transaction_id,
        "verification_status": transaction.verification_status,
        "transaction_status": transaction.transaction_status,
        "risk_classification": transaction.risk_classification,
        "risk_score": transaction.risk_score,
        "attempts_left": transaction.otp_attempts_left,
        "is_authorized": is_auth,
        "payment_url": url_for("payment", transaction_id=transaction.transaction_id) if is_auth else None,
        "expires_in_seconds": expires_in,
        "verification_notes": transaction.verification_notes
    })


# --------------------------------------------------
# 4. PAYMENT GATING & CHECKOUT (DATABASE-DRIVEN)
# --------------------------------------------------

@app.route("/payment")
def payment():
    """
    Renders PayPal payment screen ONLY for transactions verified as Safe in the database.
    Rejects missing, unapproved, suspicious, high-risk, expired, or already completed transactions.
    """
    raw_tx_id = (request.args.get("transaction_id") or "").strip()

    if not raw_tx_id:
        return render_template("result.html", result={
            "status": "Access Denied",
            "decision": "error",
            "message": "No transaction ID provided. Please submit your transaction through the Risk Analyzer first."
        }), 400

    if not is_valid_transaction_id(raw_tx_id):
        return render_template("result.html", result={
            "status": "Invalid Transaction ID",
            "decision": "error",
            "message": "The transaction ID format is invalid."
        }), 400

    try:
        transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
    except SQLAlchemyError as db_err:
        logger.exception("Database error retrieving transaction %s: %s", raw_tx_id, db_err)
        return render_template("result.html", result={
            "status": "Database Error",
            "decision": "error",
            "message": "A database error occurred while retrieving your transaction record."
        }), 500

    if not transaction:
        return render_template("result.html", result={
            "status": "Transaction Not Found",
            "decision": "error",
            "message": "This transaction does not exist in the database or has expired."
        }), 404

    # Expiration check
    if is_transaction_expired(transaction):
        if transaction.payment_status not in ("COMPLETED", "EXPIRED"):
            try:
                transaction.payment_status = "EXPIRED"
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
        return render_template("result.html", result={
            "status": "Transaction Expired",
            "decision": "error",
            "message": "This transaction assessment has expired (validity window: 3 hours). Please submit a new transaction analysis."
        }), 410

    # Strict Server-Side Gating: Only Safe (<35% risk) or Approved Suspicious Post-MFA
    if not is_transaction_authorized_for_payment(transaction):
        return render_template("result.html", result={
            "status": "Payment Disallowed",
            "decision": "error",
            "message": f"This transaction was evaluated as '{transaction.risk_classification}' (Risk: {transaction.risk_score:.2f}%). Payment processing is strictly blocked."
        }), 403

    if transaction.payment_status == "COMPLETED":
        return render_template("result.html", result={
            "status": "Already Paid",
            "decision": "completed",
            "amount": transaction.amount,
            "transaction_id": transaction.transaction_id,
            "capture_id": transaction.paypal_capture_id,
            "message": "This transaction has already been completed."
        })

    if transaction.payment_status not in ("PENDING", "NOT_ATTEMPTED"):
        return render_template("result.html", result={
            "status": "Payment Not Available",
            "decision": "error",
            "message": f"This transaction is not payable (Payment status: {transaction.payment_status})."
        }), 400

    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        return render_template("result.html", result={
            "status": "Gateway Not Configured",
            "decision": "error",
            "message": "PayPal Sandbox credentials are not configured in .env."
        }), 500

    return render_template(
        "payment.html",
        amount=transaction.amount,
        currency=transaction.currency,
        client_id=PAYPAL_CLIENT_ID,
        transaction_id=transaction.transaction_id
    )


# --------------------------------------------------
# 5. PAYPAL API: CREATE ORDER (DATABASE-DRIVEN)
# --------------------------------------------------

@app.route("/create-paypal-order", methods=["POST"])
def create_paypal_order():
    """
    Creates a PayPal order via PayPal Orders v2 API for an approved transaction stored in SQLite.
    Security Validations:
    - Verifies transaction exists in SQLite
    - Verifies server-side risk classification is Safe (risk_score < 35.0, APPROVED)
    - Blocks Suspicious and High-Risk transactions (HTTP 403)
    - Validates transaction is not expired, cancelled, failed, or already completed
    - Ignores all frontend-supplied amounts/risks/currencies
    - Attaches reference_id and custom_id to tie PayPal order directly to transaction_id
    - Persists order ID to SQLite
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_tx_id = (data.get("transaction_id") or "").strip()

        if not raw_tx_id:
            return jsonify({"error": "Missing transaction_id."}), 400

        if not is_valid_transaction_id(raw_tx_id):
            return jsonify({"error": "Invalid transaction_id format."}), 400

        try:
            transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
        except SQLAlchemyError as db_err:
            logger.exception("Database error querying transaction: %s", db_err)
            return jsonify({"error": "Database query error."}), 500

        if not transaction:
            return jsonify({"error": "Invalid or expired transaction."}), 404

        # Expiration validation
        if is_transaction_expired(transaction):
            if transaction.payment_status not in ("COMPLETED", "EXPIRED"):
                try:
                    transaction.payment_status = "EXPIRED"
                    db.session.commit()
                except SQLAlchemyError:
                    db.session.rollback()
            return jsonify({"error": "Transaction has expired. Please run a new transaction assessment."}), 410

        # Strict Server-Side Fraud Gating: Never trust frontend scores
        if not is_transaction_authorized_for_payment(transaction):
            logger.warning(
                "Security violation: Blocked PayPal order creation for unauthorized transaction %s (Risk: %s%%, Class: %s, Status: %s, MFA: %s)",
                transaction.transaction_id,
                transaction.risk_score,
                transaction.risk_classification,
                transaction.transaction_status,
                transaction.verification_status
            )
            return jsonify({
                "error": f"Payment order creation blocked: transaction risk is '{transaction.risk_classification}' (Risk: {transaction.risk_score:.2f}%)."
            }), 403

        if transaction.payment_status == "COMPLETED":
            return jsonify({"error": "Transaction has already been paid and completed."}), 400

        if transaction.payment_status in ("CANCELLED", "FAILED"):
            return jsonify({"error": f"Payment order cannot be created for a {transaction.payment_status} transaction."}), 400

        if transaction.payment_status.startswith("BLOCKED"):
            return jsonify({"error": "Payment is strictly blocked for this transaction."}), 403

        # Reuse existing order if already generated and still pending
        if transaction.paypal_order_id:
            return jsonify({"id": transaction.paypal_order_id})

        access_token = get_paypal_access_token()
        formatted_amount = f"{transaction.amount:.2f}"

        # Bind transaction ID into both reference_id and custom_id
        order_payload = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": transaction.transaction_id,
                    "custom_id": transaction.transaction_id,
                    "description": f"ScamShield Pay - Tx {transaction.transaction_id}",
                    "amount": {
                        "currency_code": transaction.currency,
                        "value": formatted_amount
                    }
                }
            ]
        }

        # Deterministic Idempotency Key for order creation
        order_idempotency_key = f"ord-{transaction.transaction_id}"

        response = requests.post(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
                "PayPal-Request-Id": order_idempotency_key
            },
            json=order_payload,
            timeout=20
        )

        if not response.ok:
            logger.error(
                "PayPal order creation failed: %s - %s",
                response.status_code,
                response.text
            )
            return jsonify({
                "error": "Failed to create order with PayPal Sandbox.",
                "details": response.json() if response.headers.get("content-type") == "application/json" else response.text
            }), response.status_code

        order = response.json()
        order_id = order.get("id")

        if not order_id:
            return jsonify({"error": "PayPal did not return an order ID."}), 502

        # Persist PayPal Order ID to Database
        try:
            transaction.paypal_order_id = order_id
            transaction.payment_status = "PENDING"
            db.session.commit()
            logger.info("Persisted PayPal order %s for transaction %s", order_id, transaction.transaction_id)
        except SQLAlchemyError as db_err:
            db.session.rollback()
            logger.exception("Failed to commit PayPal order ID to database: %s", db_err)
            return jsonify({"error": "Database error saving PayPal order."}), 500

        return jsonify({"id": order_id})

    except requests.exceptions.Timeout:
        logger.error("PayPal order request timed out.")
        return jsonify({"error": "PayPal request timed out. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        logger.error("Unable to connect to PayPal gateway.")
        return jsonify({"error": "Unable to connect to PayPal gateway. Check connection."}), 502
    except requests.RequestException as req_err:
        logger.exception("PayPal order request error: %s", req_err)
        return jsonify({"error": "Network error communicating with PayPal."}), 502
    except Exception as exc:
        logger.exception("Unexpected error creating order: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


# --------------------------------------------------
# 6. PAYPAL API: CAPTURE PAYMENT (DATABASE-DRIVEN)
# --------------------------------------------------

@app.route("/capture-paypal-order", methods=["POST"])
def capture_paypal_order():
    """
    Captures and validates a PayPal payment following buyer approval, updating the DB record.
    Security Validations:
    - Verifies transaction exists in SQLite
    - Verifies server-side risk classification is Safe (risk_score < 35.0, APPROVED)
    - Validates order ID format and matches registered paypal_order_id
    - Inspects PayPal order state in PayPal Sandbox
    - Validates PayPal order reference_id/custom_id matches the expected transaction
    - Validates currency and amount strictly against stored SQLite transaction
    - Idempotent: Handles duplicate capture calls safely without duplicate charges
    - Verifies PayPal capture response status is COMPLETED before updating database
    - Persists capture ID and updates transaction status to COMPLETED
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_tx_id = (data.get("transaction_id") or "").strip()
        raw_order_id = (data.get("orderID") or "").strip()

        if not raw_tx_id or not raw_order_id:
            return jsonify({"error": "Missing transaction_id or orderID."}), 400

        if not is_valid_transaction_id(raw_tx_id):
            return jsonify({"error": "Invalid transaction_id format."}), 400

        if not is_valid_paypal_order_id(raw_order_id):
            return jsonify({"error": "Invalid PayPal order ID format."}), 400

        try:
            transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
        except SQLAlchemyError as db_err:
            logger.exception("Database error querying transaction: %s", db_err)
            return jsonify({"error": "Database query error."}), 500

        if not transaction:
            return jsonify({"error": "Invalid or expired transaction."}), 404

        # Idempotency Check #1: If already completed in SQLite, return existing capture immediately
        if transaction.payment_status == "COMPLETED":
            logger.info(
                "Idempotency check: Transaction %s is already completed with capture %s.",
                transaction.transaction_id,
                transaction.paypal_capture_id
            )
            return jsonify({
                "status": "COMPLETED",
                "capture_id": transaction.paypal_capture_id,
                "message": "Payment has already been completed.",
                "idempotent": True
            }), 200

        # Expiration validation
        if is_transaction_expired(transaction):
            try:
                transaction.payment_status = "EXPIRED"
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
            return jsonify({"error": "Transaction has expired."}), 410

        # Strict Server-Side Gating: Authorized transactions only (Safe or Approved Post-MFA)
        if not is_transaction_authorized_for_payment(transaction):
            logger.warning(
                "Security violation: Attempted payment capture for unauthorized transaction %s (Risk: %s%%, Status: %s, MFA: %s)",
                transaction.transaction_id,
                transaction.risk_score,
                transaction.transaction_status,
                transaction.verification_status
            )
            return jsonify({"error": "Transaction is not eligible for payment capture."}), 403

        # Verify order ID matches database record
        if transaction.paypal_order_id and raw_order_id != transaction.paypal_order_id:
            logger.warning(
                "Order ID mismatch: Request order %s != database order %s for tx %s",
                raw_order_id,
                transaction.paypal_order_id,
                transaction.transaction_id
            )
            return jsonify({"error": "Order ID mismatch with database record."}), 400

        access_token = get_paypal_access_token()

        # Step A: Inspect the order state in PayPal Sandbox
        order_response = requests.get(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders/{raw_order_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20
        )

        if order_response.status_code == 404:
            return jsonify({"error": "PayPal order not found in PayPal Sandbox."}), 404

        if not order_response.ok:
            logger.error("Failed to inspect PayPal order: %s - %s", order_response.status_code, order_response.text)
            return jsonify({"error": "Unable to verify PayPal order state with PayPal Sandbox."}), 502

        order_data = order_response.json()
        order_status = order_data.get("status")

        # Step B: Validate purchase unit and cross-order verification
        purchase_units = order_data.get("purchase_units", [])
        if not purchase_units:
            return jsonify({"error": "PayPal order has no purchase units."}), 400

        pu = purchase_units[0]

        # Verify that the PayPal order belongs to this transaction
        pu_ref = pu.get("reference_id") or pu.get("custom_id")
        if pu_ref and pu_ref != transaction.transaction_id:
            logger.error(
                "Security violation: PayPal order reference '%s' does not match transaction '%s'.",
                pu_ref,
                transaction.transaction_id
            )
            return jsonify({"error": "Order verification failed: PayPal order does not belong to this transaction."}), 400

        # Validate Currency and Amount
        pu_amount = pu.get("amount", {})
        if pu_amount.get("currency_code") != transaction.currency:
            logger.error(
                "Currency mismatch: expected %s, got %s",
                transaction.currency,
                pu_amount.get("currency_code")
            )
            return jsonify({"error": f"Currency mismatch: expected {transaction.currency}."}), 400

        expected_amount = f"{transaction.amount:.2f}"
        if pu_amount.get("value") != expected_amount:
            logger.error(
                "Amount mismatch: expected %s, got %s",
                expected_amount,
                pu_amount.get("value")
            )
            return jsonify({"error": f"Payment amount mismatch: expected {expected_amount}."}), 400

        # Step C: Idempotency Check #2: If PayPal order is already COMPLETED
        if order_status == "COMPLETED":
            captures = pu.get("payments", {}).get("captures", [])
            capture_id = captures[0].get("id") if captures else "CAPTURED"
            try:
                transaction.payment_status = "COMPLETED"
                transaction.transaction_status = "COMPLETED"
                transaction.paypal_order_id = raw_order_id
                transaction.paypal_capture_id = capture_id
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
            return jsonify({
                "status": "COMPLETED",
                "capture_id": capture_id,
                "message": "Payment already completed in PayPal.",
                "idempotent": True
            }), 200

        if order_status != "APPROVED":
            logger.warning("Order %s status is '%s', not APPROVED.", raw_order_id, order_status)
            return jsonify({
                "error": f"PayPal order status is '{order_status}'. Buyer approval is required before capture."
            }), 400

        # Step D: Capture payment with Gateway Idempotency Key
        capture_idempotency_key = f"cap-{transaction.transaction_id}-{raw_order_id}"

        capture_response = requests.post(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders/{raw_order_id}/capture",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
                "PayPal-Request-Id": capture_idempotency_key
            },
            json={},
            timeout=30
        )

        # Handle ORDER_ALREADY_CAPTURED (422) as idempotent success
        if capture_response.status_code == 422:
            try:
                err_data = capture_response.json()
                issues = [d.get("issue") for d in err_data.get("details", [])]
                if "ORDER_ALREADY_CAPTURED" in issues:
                    logger.info("Order %s was already captured by PayPal; reconciling state.", raw_order_id)
                    reinspect = requests.get(
                        f"{PAYPAL_BASE_URL}/v2/checkout/orders/{raw_order_id}",
                        headers={"Authorization": f"Bearer {access_token}"},
                        timeout=15
                    )
                    if reinspect.ok:
                        r_data = reinspect.json()
                        caps = r_data.get("purchase_units", [{}])[0].get("payments", {}).get("captures", [])
                        cap_id = caps[0].get("id") if caps else "CAPTURED"
                        transaction.payment_status = "COMPLETED"
                        transaction.transaction_status = "COMPLETED"
                        transaction.paypal_capture_id = cap_id
                        db.session.commit()
                        return jsonify({
                            "status": "COMPLETED",
                            "capture_id": cap_id,
                            "message": "Payment already completed.",
                            "idempotent": True
                        }), 200
            except Exception as e:
                logger.warning("Could not reconcile 422 already captured: %s", e)

        if not capture_response.ok:
            logger.error("PayPal capture failed: %s - %s", capture_response.status_code, capture_response.text)
            try:
                transaction.payment_status = "FAILED"
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
            return jsonify({
                "error": "PayPal capture call failed.",
                "details": capture_response.text
            }), capture_response.status_code

        capture_data = capture_response.json()
        cap_status = capture_data.get("status")

        if cap_status != "COMPLETED":
            try:
                transaction.payment_status = "FAILED"
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
            return jsonify({"error": f"Payment capture status was '{cap_status}', expected 'COMPLETED'."}), 400

        # Step E: Validate captured amount and currency in capture response
        captured_units = capture_data.get("purchase_units", [])
        capture_id = None
        if captured_units:
            caps = captured_units[0].get("payments", {}).get("captures", [])
            if caps:
                cap = caps[0]
                capture_id = cap.get("id")
                c_amount = cap.get("amount", {})
                if c_amount.get("value") != expected_amount or c_amount.get("currency_code") != transaction.currency:
                    logger.error(
                        "Captured amount mismatch: expected %s %s, got %s %s",
                        expected_amount,
                        transaction.currency,
                        c_amount.get("value"),
                        c_amount.get("currency_code")
                    )
                    return jsonify({"error": "Captured amount mismatch with transaction record."}), 400

        if not capture_id:
            capture_id = capture_data.get("id") or "CAPTURED"

        # Update Database Record: COMPLETED
        try:
            transaction.payment_status = "COMPLETED"
            transaction.transaction_status = "COMPLETED"
            transaction.paypal_order_id = raw_order_id
            transaction.paypal_capture_id = capture_id
            db.session.commit()
            logger.info(
                "Transaction %s successfully captured and marked COMPLETED in database! (Capture ID: %s)",
                transaction.transaction_id,
                capture_id
            )
        except SQLAlchemyError as db_err:
            db.session.rollback()
            logger.exception("Database error updating captured transaction: %s", db_err)
            return jsonify({"error": "Database error updating transaction."}), 500

        return jsonify({
            "status": "COMPLETED",
            "capture_id": capture_id,
            "message": "Payment captured successfully."
        }), 200

    except requests.exceptions.Timeout:
        logger.error("PayPal capture request timed out.")
        return jsonify({"error": "PayPal capture request timed out. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        logger.error("Unable to connect to PayPal gateway during capture.")
        return jsonify({"error": "Unable to connect to PayPal gateway. Please check connection."}), 502
    except requests.RequestException as req_err:
        logger.exception("PayPal capture request error: %s", req_err)
        return jsonify({"error": "Unable to verify payment with PayPal Sandbox."}), 502
    except Exception as exc:
        logger.exception("Unexpected error capturing payment: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


# --------------------------------------------------
# 7. PAYMENT SUCCESS, CANCEL & LIFECYCLE HOOKS
# --------------------------------------------------

@app.route("/cancel-paypal-order", methods=["POST"])
def cancel_paypal_order():
    """
    Handles buyer cancellation notification from the frontend PayPal SDK.
    Marks the transaction payment_status as CANCELLED in SQLite if still PENDING.
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_tx_id = (data.get("transaction_id") or "").strip()

        if not raw_tx_id or not is_valid_transaction_id(raw_tx_id):
            return jsonify({"error": "Invalid transaction ID format."}), 400

        transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
        if not transaction:
            return jsonify({"error": "Transaction not found."}), 404

        if transaction.payment_status in ("PENDING", "NOT_ATTEMPTED"):
            transaction.payment_status = "CANCELLED"
            db.session.commit()
            logger.info("Marked transaction %s as CANCELLED via API.", raw_tx_id)

        return jsonify({
            "status": "CANCELLED",
            "message": "Payment was cancelled. No funds were transferred."
        })
    except SQLAlchemyError as db_err:
        db.session.rollback()
        logger.exception("Database error recording cancellation: %s", db_err)
        return jsonify({"error": "Database error recording cancellation."}), 500


@app.route("/fail-paypal-order", methods=["POST"])
def fail_paypal_order():
    """
    Handles payment failure notification from the frontend PayPal SDK.
    Marks the transaction payment_status as FAILED in SQLite if still PENDING.
    """
    try:
        data = request.get_json(silent=True) or {}
        raw_tx_id = (data.get("transaction_id") or "").strip()
        error_msg = (data.get("error") or "Payment processing failed").strip()

        if not raw_tx_id or not is_valid_transaction_id(raw_tx_id):
            return jsonify({"error": "Invalid transaction ID format."}), 400

        transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
        if not transaction:
            return jsonify({"error": "Transaction not found."}), 404

        if transaction.payment_status in ("PENDING", "NOT_ATTEMPTED"):
            transaction.payment_status = "FAILED"
            db.session.commit()
            logger.info("Marked transaction %s as FAILED via API (Reason: %s).", raw_tx_id, error_msg)

        return jsonify({
            "status": "FAILED",
            "message": "Payment failure recorded in database."
        })
    except SQLAlchemyError as db_err:
        db.session.rollback()
        logger.exception("Database error recording failure: %s", db_err)
        return jsonify({"error": "Database error recording payment failure."}), 500


@app.route("/payment/success")
def payment_success():
    """
    Renders payment completion confirmation ONLY after successful server-side capture verification.
    If the transaction is missing, invalid, or not verified as COMPLETED in the database,
    it strictly rejects the request and displays a payment incomplete warning.
    """
    raw_tx_id = (request.args.get("transaction_id") or "").strip()
    if not raw_tx_id or not is_valid_transaction_id(raw_tx_id):
        return render_template("result.html", result={
            "status": "Invalid Transaction",
            "decision": "error",
            "message": "Invalid or missing transaction identifier."
        }), 400

    try:
        transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
    except SQLAlchemyError as db_err:
        logger.exception("Database error loading completed transaction: %s", db_err)
        return render_template("result.html", result={
            "status": "Database Error",
            "decision": "error",
            "message": "Unable to verify payment status due to a database error."
        }), 500

    if not transaction:
        return render_template("result.html", result={
            "status": "Transaction Not Found",
            "decision": "error",
            "message": "Transaction record not found in the database."
        }), 404

    # Strict Server-Side Verification: Must be verified as COMPLETED in SQLite
    if transaction.payment_status != "COMPLETED":
        logger.warning(
            "Access denied to /payment/success for non-completed transaction %s (Payment status: %s)",
            raw_tx_id,
            transaction.payment_status
        )
        return render_template("result.html", result={
            "status": "Payment Not Completed",
            "decision": "error",
            "transaction_id": raw_tx_id,
            "message": f"Payment has not been completed or verified by PayPal Sandbox. Current status: '{transaction.payment_status}'."
        }), 400

    return render_template("result.html", result={
        "status": "Payment Successful",
        "decision": "completed",
        "risk_score": transaction.risk_score,
        "amount": transaction.amount,
        "transaction_id": transaction.transaction_id,
        "capture_id": transaction.paypal_capture_id
    })


@app.route("/payment/cancel")
def payment_cancel():
    """Handles cancellation by updating payment status in database and redirecting home."""
    raw_tx_id = (request.args.get("transaction_id") or "").strip()
    if raw_tx_id and is_valid_transaction_id(raw_tx_id):
        try:
            transaction = Transaction.query.filter_by(transaction_id=raw_tx_id).first()
            if transaction and transaction.payment_status in ("PENDING", "NOT_ATTEMPTED"):
                transaction.payment_status = "CANCELLED"
                db.session.commit()
                logger.info("Marked transaction %s as CANCELLED in database", raw_tx_id)
        except SQLAlchemyError as db_err:
            db.session.rollback()
            logger.exception("Database error recording cancellation: %s", db_err)

    return redirect(url_for("index"))


# --------------------------------------------------
# 8. ADMIN DASHBOARD & ANALYTICS APIS
# --------------------------------------------------

def _apply_date_filter(query, date_range):
    """Applies a date range window filter to a SQLAlchemy query."""
    now = datetime.utcnow()
    if date_range == "today":
        start_date = datetime(now.year, now.month, now.day)
        return query.filter(Transaction.created_at >= start_date)
    elif date_range == "7d":
        start_date = now - timedelta(days=7)
        return query.filter(Transaction.created_at >= start_date)
    elif date_range == "30d":
        start_date = now - timedelta(days=30)
        return query.filter(Transaction.created_at >= start_date)
    return query


@app.route("/dashboard")
@app.route("/admin")
def dashboard():
    """Renders the ScamShield Pay Fintech Admin & Fraud Intelligence Dashboard."""
    return render_template("dashboard.html")


@app.route("/api/dashboard/stats")
def api_dashboard_stats():
    """
    Returns aggregated KPI metrics from the database.
    Query parameters:
        date_range: 'today' | '7d' | '30d' | 'all' (default: 'all')
    """
    date_range = request.args.get("date_range", "all").strip().lower()
    try:
        base_query = _apply_date_filter(Transaction.query, date_range)

        total_tx = base_query.count()
        safe_tx = base_query.filter_by(risk_classification="Safe").count()
        suspicious_tx = base_query.filter_by(risk_classification="Suspicious").count()
        high_risk_tx = base_query.filter_by(risk_classification="High Risk").count()

        successful_payments = base_query.filter_by(payment_status="COMPLETED").count()

        blocked_tx = base_query.filter(
            or_(
                Transaction.transaction_status == "BLOCKED",
                Transaction.payment_status.like("BLOCKED%")
            )
        ).count()

        # Aggregated financial volume and average risk score
        agg_query = db.session.query(
            func.coalesce(func.sum(Transaction.amount), 0.0).label("total_volume"),
            func.coalesce(func.avg(Transaction.risk_score), 0.0).label("avg_risk")
        )
        agg = _apply_date_filter(agg_query, date_range).first()

        total_volume = round(float(agg.total_volume), 2) if agg and agg.total_volume is not None else 0.0
        avg_risk = round(float(agg.avg_risk), 1) if agg and agg.avg_risk is not None else 0.0

        approval_rate = round((safe_tx / total_tx * 100), 1) if total_tx > 0 else 0.0

        return jsonify({
            "status": "success",
            "date_range": date_range,
            "metrics": {
                "total_transactions": total_tx,
                "safe_transactions": safe_tx,
                "suspicious_transactions": suspicious_tx,
                "high_risk_transactions": high_risk_tx,
                "successful_payments": successful_payments,
                "blocked_transactions": blocked_tx,
                "total_volume_usd": total_volume,
                "avg_risk_score": avg_risk,
                "approval_rate": approval_rate
            }
        })
    except SQLAlchemyError as err:
        logger.exception("Database error fetching dashboard stats: %s", err)
        return jsonify({"error": "Database query error", "status": "error"}), 500


@app.route("/api/dashboard/charts")
def api_dashboard_charts():
    """
    Returns chart-ready datasets for Chart.js analytics:
    - Risk category distribution
    - Payment lifecycle outcomes
    - Daily transaction volume & amount
    - Risk trend / velocity line
    Query parameters:
        date_range: 'today' | '7d' | '30d' | 'all' (default: '7d')
    """
    date_range = request.args.get("date_range", "7d").strip().lower()
    try:
        base_query = _apply_date_filter(Transaction.query, date_range)

        # 1. Risk category distribution
        risk_distribution = {
            "Safe": base_query.filter_by(risk_classification="Safe").count(),
            "Suspicious": base_query.filter_by(risk_classification="Suspicious").count(),
            "High Risk": base_query.filter_by(risk_classification="High Risk").count()
        }

        # 2. Payment status outcomes
        status_query = db.session.query(
            Transaction.payment_status,
            func.count(Transaction.id)
        )
        status_query = _apply_date_filter(status_query, date_range)
        raw_statuses = dict(status_query.group_by(Transaction.payment_status).all())

        payment_outcomes = {
            "COMPLETED": raw_statuses.get("COMPLETED", 0),
            "PENDING": raw_statuses.get("PENDING", 0),
            "BLOCKED": sum(v for k, v in raw_statuses.items() if k.startswith("BLOCKED")),
            "FAILED": raw_statuses.get("FAILED", 0),
            "CANCELLED": raw_statuses.get("CANCELLED", 0),
            "NOT_ATTEMPTED": raw_statuses.get("NOT_ATTEMPTED", 0)
        }

        # 3. Daily volume and counts (database-agnostic date formatting)
        is_sqlite = db.engine.name == "sqlite"
        date_expr = (
            func.strftime("%Y-%m-%d", Transaction.created_at)
            if is_sqlite
            else func.to_char(Transaction.created_at, "YYYY-MM-DD")
        )

        daily_query = db.session.query(
            date_expr.label("day"),
            func.count(Transaction.id).label("count"),
            func.coalesce(func.sum(Transaction.amount), 0.0).label("volume"),
            func.coalesce(func.avg(Transaction.risk_score), 0.0).label("avg_risk")
        )
        daily_query = _apply_date_filter(daily_query, date_range if date_range != "all" else "30d")
        daily_results = daily_query.group_by("day").order_by("day").all()

        daily_data = [
            {
                "date": row.day,
                "count": int(row.count),
                "volume": round(float(row.volume), 2),
                "avg_risk": round(float(row.avg_risk), 1)
            }
            for row in daily_results
            if row.day is not None
        ]

        # 4. Recent Risk Trend (up to 20 most recent transactions)
        trend_query = _apply_date_filter(Transaction.query, date_range)
        recent_txs = trend_query.order_by(Transaction.created_at.desc()).limit(20).all()
        recent_txs.reverse()  # Chronological display

        risk_trend = {
            "labels": [t.transaction_id[:8] for t in recent_txs],
            "scores": [round(t.risk_score, 1) for t in recent_txs],
            "dates": [t.created_at.strftime("%b %d %H:%M") if t.created_at else "" for t in recent_txs],
            "amounts": [round(t.amount, 2) for t in recent_txs],
            "tiers": [t.risk_classification for t in recent_txs]
        }

        return jsonify({
            "status": "success",
            "date_range": date_range,
            "risk_distribution": risk_distribution,
            "payment_outcomes": payment_outcomes,
            "daily_volume": daily_data,
            "risk_trend": risk_trend
        })
    except SQLAlchemyError as err:
        logger.exception("Database error fetching chart data: %s", err)
        return jsonify({"error": "Database query error", "status": "error"}), 500


def apply_transaction_filters(query, search="", risk="all", status="all", date_range="all"):
    """
    Applies validated filter criteria to a Transaction query:
    - date_range: 'today' | '7d' | '30d' | 'all'
    - risk: 'Safe' | 'Suspicious' | 'High Risk' | 'all'
    - status: 'COMPLETED' (Payment Successful) | 'FAILED' (Payment Failed) | 'BLOCKED' (Blocked) | 'PENDING' | 'all'
    - search: string matched against transaction_id, paypal_order_id, paypal_capture_id
    """
    query = _apply_date_filter(query, date_range)

    # Risk classification filtering
    if risk and risk.lower() != "all":
        risk_map = {
            "safe": "Safe",
            "suspicious": "Suspicious",
            "high risk": "High Risk",
            "high_risk": "High Risk",
            "highrisk": "High Risk"
        }
        canonical_risk = risk_map.get(risk.lower(), risk)
        query = query.filter(Transaction.risk_classification == canonical_risk)

    # Payment status filtering
    if status and status.lower() != "all":
        norm_status = status.lower().strip()
        if norm_status in ("completed", "successful", "payment successful", "success"):
            query = query.filter(Transaction.payment_status == "COMPLETED")
        elif norm_status in ("failed", "payment failed", "failure"):
            query = query.filter(Transaction.payment_status == "FAILED")
        elif norm_status in ("blocked", "blocked_suspicious", "blocked_high_risk"):
            query = query.filter(
                or_(
                    Transaction.payment_status.like("BLOCKED%"),
                    Transaction.transaction_status == "BLOCKED"
                )
            )
        elif norm_status in ("pending",):
            query = query.filter(Transaction.payment_status == "PENDING")
        elif norm_status in ("cancelled", "canceled"):
            query = query.filter(Transaction.payment_status == "CANCELLED")
        elif norm_status in ("safe", "suspicious", "high risk", "high_risk"):
            risk_map = {"safe": "Safe", "suspicious": "Suspicious", "high risk": "High Risk", "high_risk": "High Risk"}
            query = query.filter(Transaction.risk_classification == risk_map[norm_status])
        else:
            query = query.filter(Transaction.payment_status == status.upper())

    # Search filter
    if search:
        clean_search = search.strip()
        term = f"%{clean_search}%"
        query = query.filter(
            or_(
                Transaction.transaction_id.ilike(term),
                Transaction.paypal_order_id.ilike(term),
                Transaction.paypal_capture_id.ilike(term)
            )
        )

    return query


@app.route("/api/dashboard/transactions")
def api_dashboard_transactions():
    """
    Returns filtered and paginated transactions from the database.
    Query parameters:
        risk: 'all' | 'Safe' | 'Suspicious' | 'High Risk'
        status: 'all' | 'COMPLETED' | 'PENDING' | 'BLOCKED' | 'FAILED' | 'CANCELLED'
        search: string (matching transaction_id, paypal_order_id, paypal_capture_id)
        date_range: 'today' | '7d' | '30d' | 'all'
        limit: int (1..100, default 25)
        offset: int (default 0)
    """
    try:
        risk_filter = request.args.get("risk", "all").strip()
        status_filter = request.args.get("status", "all").strip()
        search_query = request.args.get("search", "").strip()
        date_range = request.args.get("date_range", "all").strip().lower()

        try:
            limit = min(max(int(request.args.get("limit", 25)), 1), 100)
            offset = max(int(request.args.get("offset", 0)), 0)
        except (ValueError, TypeError):
            limit = 25
            offset = 0

        query = apply_transaction_filters(
            Transaction.query,
            search=search_query,
            risk=risk_filter,
            status=status_filter,
            date_range=date_range
        )

        total_count = query.count()
        records = query.order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()

        transactions_data = [
            {
                "id": t.id,
                "transaction_id": t.transaction_id,
                "amount": t.amount,
                "currency": t.currency,
                "transaction_hour": t.transaction_hour,
                "international": bool(t.international),
                "new_device": bool(t.new_device),
                "failed_attempts": t.failed_attempts,
                "risk_score": round(t.risk_score, 2),
                "risk_classification": t.risk_classification,
                "transaction_status": t.transaction_status,
                "verification_status": t.verification_status,
                "verification_notes": t.verification_notes,
                "payment_status": t.payment_status,
                "paypal_order_id": t.paypal_order_id,
                "paypal_capture_id": t.paypal_capture_id,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "created_at_formatted": t.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if t.created_at else ""
            }
            for t in records
        ]

        return jsonify({
            "status": "success",
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "transactions": transactions_data
        })

    except SQLAlchemyError as err:
        logger.exception("Database error querying transactions: %s", err)
        return jsonify({"error": "Database error retrieving transactions", "status": "error"}), 500


@app.route("/api/dashboard/transactions/<transaction_id>")
def api_dashboard_transaction_detail(transaction_id):
    """
    Returns deep inspection detail for a single transaction:
    - Input features
    - ML probability verdict
    - Rule-based explainable risk indicators
    - Actionable recommendation
    - PayPal lifecycle references
    """
    clean_tx_id = (transaction_id or "").strip()
    if not clean_tx_id or not is_valid_transaction_id(clean_tx_id):
        return jsonify({"error": "Invalid transaction ID format.", "status": "error"}), 400

    try:
        transaction = Transaction.query.filter_by(transaction_id=clean_tx_id).first()
        if not transaction:
            return jsonify({"error": "Transaction not found.", "status": "error"}), 404

        # Compute explainable risk indicators
        indicators = analyze_risk_indicators(
            amount=transaction.amount,
            transaction_hour=transaction.transaction_hour,
            international=transaction.international,
            new_device=transaction.new_device,
            failed_attempts=transaction.failed_attempts
        )

        decision = "approved" if transaction.risk_classification == "Safe" else (
            "review" if transaction.risk_classification == "Suspicious" else "blocked"
        )
        recommendation = get_recommended_action(decision, transaction.risk_score)

        is_payable = (
            is_transaction_authorized_for_payment(transaction)
            and transaction.payment_status in ("PENDING", "NOT_ATTEMPTED")
        )

        return jsonify({
            "status": "success",
            "transaction": {
                "id": transaction.id,
                "transaction_id": transaction.transaction_id,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "transaction_hour": transaction.transaction_hour,
                "international": bool(transaction.international),
                "new_device": bool(transaction.new_device),
                "failed_attempts": transaction.failed_attempts,
                "risk_score": round(transaction.risk_score, 2),
                "risk_classification": transaction.risk_classification,
                "transaction_status": transaction.transaction_status,
                "verification_status": transaction.verification_status,
                "verification_notes": transaction.verification_notes,
                "otp_attempts_left": transaction.otp_attempts_left,
                "payment_status": transaction.payment_status,
                "paypal_order_id": transaction.paypal_order_id,
                "paypal_capture_id": transaction.paypal_capture_id,
                "created_at": transaction.created_at.isoformat() if transaction.created_at else None,
                "created_at_formatted": transaction.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if transaction.created_at else "",
                "is_payable": is_payable,
                "payment_url": url_for("payment", transaction_id=transaction.transaction_id) if is_payable else None,
                "verify_url": url_for("verify_transaction", transaction_id=transaction.transaction_id) if transaction.risk_classification == "Suspicious" else None
            },
            "explainability": {
                "indicators": indicators,
                "recommendation": recommendation,
                "model_name": "Random Forest Classifier (200 Trees, Stratified)",
                "feature_weights": {
                    "amount": 0.457,
                    "transaction_hour": 0.222,
                    "failed_attempts": 0.178,
                    "international": 0.081,
                    "new_device": 0.062
                }
            }
        })

    except SQLAlchemyError as err:
        logger.exception("Database error loading transaction detail: %s", err)
        return jsonify({"error": "Database error", "status": "error"}), 500


@app.route("/api/dashboard/seed-demo", methods=["POST"])
def api_dashboard_seed_demo():
    """
    Seeds realistic demo transactions scored directly by the trained Random Forest model.
    Facilitates immediate visual evaluation of charts, filters, and analytics without manual inputs.
    """
    demo_scenarios = [
        # (amount, hour, intl, new_dev, failed, days_ago, hours_ago, payment_status, sample_order, sample_cap)
        (45.0, 14, 0, 0, 0, 5, 2, "COMPLETED", "DEMO-ORD-101", "DEMO-CAP-101"),
        (89.50, 16, 0, 0, 0, 4, 3, "COMPLETED", "DEMO-ORD-102", "DEMO-CAP-102"),
        (120.0, 12, 0, 0, 0, 3, 5, "COMPLETED", "DEMO-ORD-103", "DEMO-CAP-103"),
        (35.0, 10, 0, 0, 0, 2, 1, "PENDING", "DEMO-ORD-104", None),
        (250.0, 18, 0, 0, 0, 1, 4, "COMPLETED", "DEMO-ORD-105", "DEMO-CAP-105"),
        (65.0, 15, 0, 0, 0, 0, 2, "PENDING", None, None),
        (6500.0, 2, 1, 0, 1, 4, 1, "BLOCKED_SUSPICIOUS", None, None),
        (12000.0, 23, 0, 1, 2, 3, 7, "BLOCKED_SUSPICIOUS", None, None),
        (8500.0, 1, 1, 1, 1, 2, 4, "BLOCKED_SUSPICIOUS", None, None),
        (5400.0, 4, 0, 1, 2, 1, 2, "BLOCKED_SUSPICIOUS", None, None),
        (28000.0, 3, 1, 1, 3, 3, 2, "BLOCKED_HIGH_RISK", None, None),
        (45000.0, 1, 1, 1, 4, 2, 8, "BLOCKED_HIGH_RISK", None, None),
        (85000.0, 23, 1, 1, 5, 1, 6, "BLOCKED_HIGH_RISK", None, None),
        (32000.0, 4, 1, 0, 4, 0, 3, "BLOCKED_HIGH_RISK", None, None)
    ]

    now = datetime.utcnow()
    created_records = []

    try:
        for amount, hour, intl, new_dev, failed, d_ago, h_ago, pay_status, ord_id, cap_id in demo_scenarios:
            df = pd.DataFrame([{
                "amount": float(amount),
                "transaction_hour": int(hour),
                "international": int(intl),
                "new_device": int(new_dev),
                "failed_attempts": int(failed)
            }], columns=FEATURE_COLUMNS)

            # Predict probability using real Random Forest model
            prob = model.predict_proba(df)[0][1]
            risk_score = round(float(prob * 100), 2)

            if risk_score < 35.0:
                classification = "Safe"
                tx_status = "APPROVED" if pay_status != "COMPLETED" else "COMPLETED"
            elif risk_score < 70.0:
                classification = "Suspicious"
                tx_status = "UNDER_REVIEW"
                pay_status = "BLOCKED_SUSPICIOUS"
            else:
                classification = "High Risk"
                tx_status = "BLOCKED"
                pay_status = "BLOCKED_HIGH_RISK"

            tx_token = secrets.token_urlsafe(16)
            created_time = now - timedelta(days=d_ago, hours=h_ago)

            record = Transaction(
                transaction_id=tx_token,
                amount=amount,
                currency="USD",
                transaction_hour=hour,
                international=intl,
                new_device=new_dev,
                failed_attempts=failed,
                risk_score=risk_score,
                risk_classification=classification,
                transaction_status=tx_status,
                paypal_order_id=ord_id,
                paypal_capture_id=cap_id,
                payment_status=pay_status,
                created_at=created_time
            )
            db.session.add(record)
            created_records.append(tx_token)

        db.session.commit()
        logger.info("Successfully seeded %d demo transactions into database.", len(created_records))
        return jsonify({
            "status": "success",
            "message": f"Successfully seeded {len(created_records)} realistic demo transactions.",
            "count": len(created_records)
        })

    except Exception as exc:
        db.session.rollback()
        logger.exception("Failed to seed demo transactions: %s", exc)
        return jsonify({"error": "Failed to seed demo transactions.", "status": "error"}), 500


@app.route("/api/dashboard/clear-demo", methods=["POST"])
def api_dashboard_clear_demo():
    """Clears all transactions in the database (for testing and clean demo resets)."""
    try:
        deleted = db.session.query(Transaction).delete()
        db.session.commit()
        logger.info("Cleared %d transactions from database.", deleted)
        return jsonify({
            "status": "success",
            "message": f"Cleared {deleted} transactions.",
            "count": deleted
        })
    except SQLAlchemyError as exc:
        db.session.rollback()
        logger.exception("Failed to clear transactions: %s", exc)
        return jsonify({"error": "Failed to clear transactions.", "status": "error"}), 500


# --------------------------------------------------
# 9. TRANSACTION HISTORY & DETAILS ROUTES
# --------------------------------------------------

@app.route("/transactions")
@app.route("/history")
def transactions_history():
    """
    Renders the complete Transaction History page with:
    - Real-time search by transaction ID, PayPal Order ID, or Capture ID
    - Multi-factor filtering: Safe, Suspicious, High Risk, Payment Successful, Payment Failed, Blocked
    - Date-range filtering (Today, Last 7 Days, Last 30 Days, All Time)
    - Pagination with page size controls
    - Summary metrics banner
    - Responsive table with clear status badges
    - Quick-inspect modal and full detail view links
    Also returns JSON if requested (Accept: application/json or ?format=json).
    """
    search = request.args.get("search", "").strip()
    risk = request.args.get("risk", "all").strip()
    status = request.args.get("status", "all").strip()
    date_range = request.args.get("date_range", "all").strip().lower()

    try:
        page = max(int(request.args.get("page", 1)), 1)
    except (ValueError, TypeError):
        page = 1

    try:
        per_page = min(max(int(request.args.get("limit", request.args.get("per_page", 15))), 1), 100)
    except (ValueError, TypeError):
        per_page = 15

    try:
        base_query = Transaction.query
        filtered_query = apply_transaction_filters(
            base_query,
            search=search,
            risk=risk,
            status=status,
            date_range=date_range
        )

        total_records = filtered_query.count()
        total_pages = max(1, math.ceil(total_records / per_page))
        if page > total_pages and total_records > 0:
            page = total_pages

        offset = (page - 1) * per_page
        records = filtered_query.order_by(Transaction.created_at.desc()).offset(offset).limit(per_page).all()

        transactions_list = [
            {
                "id": t.id,
                "transaction_id": t.transaction_id,
                "amount": t.amount,
                "currency": t.currency,
                "transaction_hour": t.transaction_hour,
                "international": bool(t.international),
                "new_device": bool(t.new_device),
                "failed_attempts": t.failed_attempts,
                "risk_score": round(t.risk_score, 2),
                "risk_classification": t.risk_classification,
                "transaction_status": t.transaction_status,
                "verification_status": t.verification_status,
                "verification_notes": t.verification_notes,
                "payment_status": t.payment_status,
                "paypal_order_id": t.paypal_order_id,
                "paypal_capture_id": t.paypal_capture_id,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "created_at_formatted": t.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if t.created_at else ""
            }
            for t in records
        ]

        # Overall summary figures for filter badges
        summary_counts = {
            "total": Transaction.query.count(),
            "safe": Transaction.query.filter_by(risk_classification="Safe").count(),
            "suspicious": Transaction.query.filter_by(risk_classification="Suspicious").count(),
            "high_risk": Transaction.query.filter_by(risk_classification="High Risk").count(),
            "completed": Transaction.query.filter_by(payment_status="COMPLETED").count(),
            "failed": Transaction.query.filter_by(payment_status="FAILED").count(),
            "blocked": Transaction.query.filter(
                or_(
                    Transaction.payment_status.like("BLOCKED%"),
                    Transaction.transaction_status == "BLOCKED"
                )
            ).count()
        }

        is_json = (
            request.is_json
            or request.args.get("format") == "json"
            or request.headers.get("Accept") == "application/json"
        )

        if is_json:
            return jsonify({
                "status": "success",
                "total": total_records,
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "has_prev": page > 1,
                "has_next": page < total_pages,
                "transactions": transactions_list,
                "summary": summary_counts,
                "filters": {
                    "search": search,
                    "risk": risk,
                    "status": status,
                    "date_range": date_range
                }
            })

        return render_template(
            "transactions.html",
            transactions=transactions_list,
            total_records=total_records,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_prev=page > 1,
            has_next=page < total_pages,
            prev_page=page - 1,
            next_page=page + 1,
            search=search,
            risk=risk,
            status=status,
            date_range=date_range,
            summary=summary_counts
        )

    except SQLAlchemyError as err:
        logger.exception("Database error loading transactions history: %s", err)
        if request.is_json or request.args.get("format") == "json":
            return jsonify({"error": "Database error", "status": "error"}), 500
        return render_template("result.html", result={
            "status": "Database Error",
            "decision": "error",
            "message": "A database error occurred while retrieving transaction history."
        }), 500


@app.route("/transactions/<transaction_id>")
def transaction_detail(transaction_id):
    """
    Renders the dedicated Transaction Details page displaying:
    - Behavioral input features (amount, hour, international, new device, failed attempts)
    - ML probability verdict & risk classification
    - 5 explainable risk indicators (contextual heuristics)
    - Actionable security recommendation
    - PayPal order and capture references
    - Direct payment link if transaction is Safe and pending
    Also returns JSON if requested.
    """
    clean_tx_id = (transaction_id or "").strip()
    if not clean_tx_id or not is_valid_transaction_id(clean_tx_id):
        if request.is_json or request.args.get("format") == "json":
            return jsonify({"error": "Invalid transaction ID format.", "status": "error"}), 400
        return render_template("result.html", result={
            "status": "Invalid Transaction ID",
            "decision": "error",
            "message": "The transaction ID provided has an invalid format."
        }), 400

    try:
        transaction = Transaction.query.filter_by(transaction_id=clean_tx_id).first()
        if not transaction:
            if request.is_json or request.args.get("format") == "json":
                return jsonify({"error": "Transaction not found.", "status": "error"}), 404
            return render_template("result.html", result={
                "status": "Transaction Not Found",
                "decision": "error",
                "message": f"Transaction '{clean_tx_id}' was not found in the database."
            }), 404

        # Compute explainable risk indicators
        indicators = analyze_risk_indicators(
            amount=transaction.amount,
            transaction_hour=transaction.transaction_hour,
            international=transaction.international,
            new_device=transaction.new_device,
            failed_attempts=transaction.failed_attempts
        )

        decision = "approved" if transaction.risk_classification == "Safe" else (
            "review" if transaction.risk_classification == "Suspicious" else "blocked"
        )
        recommendation = get_recommended_action(decision, transaction.risk_score)

        is_payable = (
            is_transaction_authorized_for_payment(transaction)
            and transaction.payment_status in ("PENDING", "NOT_ATTEMPTED")
        )

        tx_dict = {
            "id": transaction.id,
            "transaction_id": transaction.transaction_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "transaction_hour": transaction.transaction_hour,
            "international": bool(transaction.international),
            "new_device": bool(transaction.new_device),
            "failed_attempts": transaction.failed_attempts,
            "risk_score": round(transaction.risk_score, 2),
            "risk_classification": transaction.risk_classification,
            "transaction_status": transaction.transaction_status,
            "verification_status": transaction.verification_status,
            "verification_notes": transaction.verification_notes,
            "otp_attempts_left": transaction.otp_attempts_left,
            "payment_status": transaction.payment_status,
            "paypal_order_id": transaction.paypal_order_id,
            "paypal_capture_id": transaction.paypal_capture_id,
            "created_at": transaction.created_at.isoformat() if transaction.created_at else None,
            "created_at_formatted": transaction.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if transaction.created_at else "",
            "is_payable": is_payable,
            "payment_url": url_for("payment", transaction_id=transaction.transaction_id) if is_payable else None,
            "verify_url": url_for("verify_transaction", transaction_id=transaction.transaction_id) if transaction.risk_classification == "Suspicious" else None
        }

        if request.is_json or request.args.get("format") == "json":
            return jsonify({
                "status": "success",
                "transaction": tx_dict,
                "explainability": {
                    "indicators": indicators,
                    "recommendation": recommendation,
                    "model_name": "Random Forest Classifier (200 Trees, Stratified)",
                    "feature_weights": {
                        "amount": 0.457,
                        "transaction_hour": 0.222,
                        "failed_attempts": 0.178,
                        "international": 0.081,
                        "new_device": 0.062
                    }
                }
            })

        return render_template(
            "transaction_detail.html",
            transaction=tx_dict,
            indicators=indicators,
            recommendation=recommendation,
            is_payable=is_payable
        )

    except SQLAlchemyError as err:
        logger.exception("Database error loading transaction detail: %s", err)
        if request.is_json or request.args.get("format") == "json":
            return jsonify({"error": "Database error", "status": "error"}), 500
        return render_template("result.html", result={
            "status": "Database Error",
            "decision": "error",
            "message": "A database error occurred while retrieving transaction details."
        }), 500


# --------------------------------------------------
# 10. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t", "yes")

    logger.info("Starting ScamShield Pay server on http://127.0.0.1:%s (debug=%s)", port, debug)
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )
