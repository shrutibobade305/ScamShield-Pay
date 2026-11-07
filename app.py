import logging
import os
import secrets
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

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

# Machine Learning Model Loading
MODEL_PATH = BASE_DIR / "models" / "scam_model.pkl"

if not MODEL_PATH.exists():
    # Fallback check if models directory has scam_model.pkl
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


# --------------------------------------------------
# 2. IN-MEMORY TRANSACTION REGISTRY
# --------------------------------------------------

# In-memory storage for transaction verification (production would use a persistent DB)
approved_transactions = {}


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

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
# 4. CORE ROUTES
# --------------------------------------------------

@app.route("/")
def index():
    """Homepage: Displays marketing hero and transaction risk analyzer."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint for container/cloud deployment monitoring."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "paypal_configured": bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET),
        "environment": os.getenv("FLASK_ENV", "production")
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Evaluates transaction risk using the Random Forest Classifier.
    Classifications:
      - Safe: Risk < 35% -> Eligible for PayPal payment.
      - Suspicious: 35% <= Risk < 70% -> Payment disabled (under review).
      - High Risk: Risk >= 70% -> Payment strictly blocked.
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

        # Model risk scoring
        probabilities = model.predict_proba(features_df)[0]
        classes = list(model.classes_)

        if 1 in classes:
            fraud_index = classes.index(1)
            risk_score = float(probabilities[fraud_index]) * 100.0
        else:
            risk_score = 0.0

        risk_score = round(risk_score, 2)

        # Decision thresholds
        if risk_score >= 70.0:
            status = "High Risk"
            decision = "blocked"
        elif risk_score >= 35.0:
            status = "Suspicious"
            decision = "review"
        else:
            status = "Safe"
            decision = "approved"

        transaction_id = secrets.token_urlsafe(16)

        result = {
            "amount": amount,
            "risk_score": risk_score,
            "status": status,
            "decision": decision,
            "transaction_id": transaction_id
        }

        # ONLY Safe / Approved transactions are registered as payable
        if decision == "approved":
            approved_transactions[transaction_id] = {
                "amount": amount,
                "currency": CURRENCY,
                "risk_score": risk_score,
                "status": "PENDING",
                "paypal_order_id": None,
                "capture_id": None
            }
            logger.info(
                "Transaction %s approved (Risk: %.2f%%). Registered for payment.",
                transaction_id,
                risk_score
            )
        else:
            logger.warning(
                "Transaction %s flagged as %s (Risk: %.2f%%). Payment disallowed.",
                transaction_id,
                status,
                risk_score
            )

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
# 5. PAYMENT GATING & CHECKOUT
# --------------------------------------------------

@app.route("/payment")
def payment():
    """
    Renders PayPal payment screen ONLY for transactions previously verified as Safe.
    Rejects missing, expired, or non-approved transactions.
    """
    transaction_id = request.args.get("transaction_id")

    if not transaction_id:
        return render_template("result.html", result={
            "status": "Access Denied",
            "decision": "error",
            "message": "No transaction ID provided. Please submit your transaction through the Risk Analyzer first."
        }), 400

    transaction = approved_transactions.get(transaction_id)

    if not transaction:
        return render_template("result.html", result={
            "status": "Transaction Expired or Not Found",
            "decision": "error",
            "message": "This transaction is either invalid, expired, or was not classified as Safe."
        }), 404

    if transaction["status"] == "COMPLETED":
        return render_template("result.html", result={
            "status": "Already Paid",
            "decision": "completed",
            "amount": transaction["amount"],
            "transaction_id": transaction_id,
            "capture_id": transaction.get("capture_id"),
            "message": "This transaction has already been completed."
        })

    if transaction["status"] != "PENDING":
        return render_template("result.html", result={
            "status": "Payment Not Available",
            "decision": "error",
            "message": f"This transaction cannot be paid (Status: {transaction['status']})."
        }), 400

    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        return render_template("result.html", result={
            "status": "Gateway Not Configured",
            "decision": "error",
            "message": "PayPal Sandbox credentials are not configured in .env."
        }), 500

    return render_template(
        "payment.html",
        amount=transaction["amount"],
        currency=transaction["currency"],
        client_id=PAYPAL_CLIENT_ID,
        transaction_id=transaction_id
    )


# --------------------------------------------------
# 6. PAYPAL API: CREATE ORDER
# --------------------------------------------------

@app.route("/create-paypal-order", methods=["POST"])
def create_paypal_order():
    """Creates a PayPal order via PayPal Orders v2 API for an approved transaction."""
    try:
        data = request.get_json(silent=True) or {}
        transaction_id = data.get("transaction_id")

        if not transaction_id:
            return jsonify({"error": "Missing transaction_id."}), 400

        transaction = approved_transactions.get(transaction_id)

        if not transaction:
            return jsonify({"error": "Invalid or expired transaction."}), 404

        if transaction["status"] != "PENDING":
            return jsonify({"error": "Transaction is not payable."}), 400

        # Reuse order if already created and valid
        if transaction.get("paypal_order_id"):
            return jsonify({"id": transaction["paypal_order_id"]})

        access_token = get_paypal_access_token()
        formatted_amount = f'{transaction["amount"]:.2f}'

        order_payload = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": transaction_id,
                    "amount": {
                        "currency_code": transaction["currency"],
                        "value": formatted_amount
                    }
                }
            ]
        }

        response = requests.post(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
                "PayPal-Request-Id": secrets.token_urlsafe(20)
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
                "error": "Failed to create order with PayPal.",
                "details": response.json() if response.headers.get("content-type") == "application/json" else response.text
            }), response.status_code

        order = response.json()
        order_id = order.get("id")

        if not order_id:
            return jsonify({"error": "PayPal did not return an order ID."}), 502

        transaction["paypal_order_id"] = order_id
        logger.info("PayPal order %s created for transaction %s", order_id, transaction_id)

        return jsonify({"id": order_id})

    except requests.RequestException as req_err:
        logger.exception("PayPal order request error: %s", req_err)
        return jsonify({"error": "Network error communicating with PayPal."}), 502

    except Exception as exc:
        logger.exception("Unexpected error creating order: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


# --------------------------------------------------
# 7. PAYPAL API: CAPTURE PAYMENT
# --------------------------------------------------

@app.route("/capture-paypal-order", methods=["POST"])
def capture_paypal_order():
    """Captures and validates a PayPal payment following buyer approval."""
    try:
        data = request.get_json(silent=True) or {}
        transaction_id = data.get("transaction_id")
        order_id = data.get("orderID")

        if not transaction_id or not order_id:
            return jsonify({"error": "Missing transaction_id or orderID."}), 400

        transaction = approved_transactions.get(transaction_id)

        if not transaction:
            return jsonify({"error": "Invalid or expired transaction."}), 404

        if transaction["status"] == "COMPLETED":
            return jsonify({
                "status": "COMPLETED",
                "capture_id": transaction.get("capture_id"),
                "message": "Payment has already been completed."
            })

        if transaction["status"] != "PENDING":
            return jsonify({"error": "Transaction is not eligible for capture."}), 400

        if transaction.get("paypal_order_id") and order_id != transaction["paypal_order_id"]:
            return jsonify({"error": "Order ID mismatch."}), 400

        access_token = get_paypal_access_token()

        # Step A: Inspect the order state in PayPal
        order_response = requests.get(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders/{order_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=20
        )

        if not order_response.ok:
            logger.error("Failed to inspect PayPal order: %s", order_response.text)
            return jsonify({"error": "Unable to verify PayPal order state."}), 502

        order_data = order_response.json()
        order_status = order_data.get("status")

        # Step B: Validate purchase unit currency and amount
        purchase_units = order_data.get("purchase_units", [])
        if not purchase_units:
            return jsonify({"error": "Order has no purchase units."}), 400

        pu = purchase_units[0]
        pu_amount = pu.get("amount", {})

        if pu_amount.get("currency_code") != transaction["currency"]:
            return jsonify({"error": "Currency mismatch."}), 400

        expected_amount = f'{transaction["amount"]:.2f}'
        if pu_amount.get("value") != expected_amount:
            return jsonify({"error": "Payment amount mismatch."}), 400

        # Step C: Capture if approved, or check if already captured
        if order_status == "COMPLETED":
            # Already captured
            captures = pu.get("payments", {}).get("captures", [])
            capture_id = captures[0].get("id") if captures else "CAPTURED"
            transaction["status"] = "COMPLETED"
            transaction["capture_id"] = capture_id
            return jsonify({
                "status": "COMPLETED",
                "capture_id": capture_id,
                "message": "Payment already completed."
            })

        if order_status != "APPROVED":
            return jsonify({
                "error": f"PayPal order status is '{order_status}'. Buyer approval is required before capture."
            }), 400

        capture_response = requests.post(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders/{order_id}/capture",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
                "PayPal-Request-Id": f"cap-{order_id}-{secrets.token_urlsafe(8)}"
            },
            json={},
            timeout=30
        )

        if not capture_response.ok:
            logger.error("PayPal capture failed: %s", capture_response.text)
            return jsonify({"error": "PayPal capture call failed."}), capture_response.status_code

        capture_data = capture_response.json()
        cap_status = capture_data.get("status")

        if cap_status != "COMPLETED":
            return jsonify({"error": f"Payment capture status was '{cap_status}'."}), 400

        # Retrieve capture ID
        captured_units = capture_data.get("purchase_units", [])
        capture_id = None
        if captured_units:
            caps = captured_units[0].get("payments", {}).get("captures", [])
            if caps:
                capture_id = caps[0].get("id")

        transaction["status"] = "COMPLETED"
        transaction["capture_id"] = capture_id
        logger.info(
            "Transaction %s successfully captured! (PayPal Order: %s, Capture: %s)",
            transaction_id,
            order_id,
            capture_id
        )

        return jsonify({
            "status": "COMPLETED",
            "capture_id": capture_id,
            "message": "Payment captured successfully."
        })

    except requests.RequestException as req_err:
        logger.exception("PayPal capture request error: %s", req_err)
        return jsonify({"error": "Unable to verify payment with PayPal."}), 502

    except Exception as exc:
        logger.exception("Unexpected error capturing payment: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


# --------------------------------------------------
# 8. PAYMENT SUCCESS & CANCEL
# --------------------------------------------------

@app.route("/payment/success")
def payment_success():
    """Displays payment completion confirmation."""
    transaction_id = request.args.get("transaction_id")
    transaction = approved_transactions.get(transaction_id) if transaction_id else None

    amount = transaction["amount"] if transaction else 0.0
    capture_id = transaction.get("capture_id") if transaction else None
    risk_score = transaction.get("risk_score", 0.0) if transaction else 0.0

    return render_template("result.html", result={
        "status": "Payment Successful",
        "decision": "completed",
        "risk_score": risk_score,
        "amount": amount,
        "transaction_id": transaction_id,
        "capture_id": capture_id
    })


@app.route("/payment/cancel")
def payment_cancel():
    """Handles cancellation by redirecting safely back to home."""
    return redirect(url_for("index"))


# --------------------------------------------------
# 9. RUN APPLICATION
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
