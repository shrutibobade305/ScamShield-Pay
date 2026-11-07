
import os
import secrets
from pathlib import Path

import joblib
import requests

from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

secret_key = os.getenv("SECRET_KEY")

if not secret_key:
    raise RuntimeError("SECRET_KEY is missing from your .env file.")

app.secret_key = secret_key

# ML model
MODEL_PATH = BASE_DIR / "models" / "scam_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)


# PayPal configuration
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET")

PAYPAL_BASE_URL = os.getenv(
    "PAYPAL_BASE_URL",
    "https://api-m.sandbox.paypal.com"
).rstrip("/")

CURRENCY = "USD"


# --------------------------------------------------
# 2. TEMPORARY TRANSACTION STORAGE
# --------------------------------------------------

# Demo only: transactions disappear when the server restarts.
# Use a database such as SQLite or PostgreSQL for production.

approved_transactions = {}


# --------------------------------------------------
# 3. HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# 4. FRAUD DETECTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        amount = float(request.form.get("amount", 0))
        transaction_hour = int(request.form.get("transaction_hour", 0))
        international = int(request.form.get("international", 0))
        new_device = int(request.form.get("new_device", 0))
        failed_attempts = int(request.form.get("failed_attempts", 0))

        # Validate input
        if amount <= 0:
            return "Amount must be greater than zero.", 400

        if not 0 <= transaction_hour <= 23:
            return "Transaction hour must be between 0 and 23.", 400

        if international not in (0, 1) or new_device not in (0, 1):
            return "Invalid transaction options.", 400

        if failed_attempts < 0:
            return "Failed attempts cannot be negative.", 400

        # Features must match the order used during model training.
        features = [[
            amount,
            transaction_hour,
            international,
            new_device,
            failed_attempts
        ]]

        # Predict fraud probability
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        fraud_index = classes.index(1)
        risk_score = float(probabilities[fraud_index]) * 100

        # Risk classification
        if risk_score >= 70:
            status = "High Risk"
            decision = "blocked"

        elif risk_score >= 35:
            status = "Suspicious"
            decision = "review"

        else:
            status = "Safe"
            decision = "approved"

        transaction_id = secrets.token_urlsafe(16)

        result = {
            "amount": amount,
            "risk_score": round(risk_score, 2),
            "status": status,
            "decision": decision,
            "transaction_id": transaction_id
        }

        # Only low-risk transactions are eligible for payment.
        if decision == "approved":
            approved_transactions[transaction_id] = {
                "amount": amount,
                "currency": CURRENCY,
                "risk_score": round(risk_score, 2),
                "status": "PENDING",
                "paypal_order_id": None
            }

        return render_template("result.html", result=result)

    except (ValueError, TypeError):
        return "Please enter valid transaction details.", 400

    except Exception as error:
        app.logger.exception("Prediction failed: %s", error)
        return "Unable to process the transaction.", 500


# --------------------------------------------------
# 5. PAYMENT PAGE
# --------------------------------------------------

@app.route("/payment")
def payment():

    transaction_id = request.args.get("transaction_id")

    transaction = approved_transactions.get(transaction_id)

    if not transaction:
        return "Invalid or expired transaction.", 404

    if transaction["status"] != "PENDING":
        return "This transaction cannot be paid again.", 400

    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        return "PayPal credentials are not configured.", 500

    return render_template(
        "payment.html",
        amount=transaction["amount"],
        currency=transaction["currency"],
        client_id=PAYPAL_CLIENT_ID,
        transaction_id=transaction_id
    )


# --------------------------------------------------
# 6. GET PAYPAL ACCESS TOKEN
# --------------------------------------------------

def get_paypal_access_token():

    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise RuntimeError("PayPal credentials are missing.")

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

    response.raise_for_status()

    token = response.json().get("access_token")

    if not token:
        raise RuntimeError("PayPal did not return an access token.")

    return token


# --------------------------------------------------
# 7. CREATE PAYPAL ORDER
# --------------------------------------------------

@app.route("/create-paypal-order", methods=["POST"])
def create_paypal_order():

    try:
        data = request.get_json(silent=True) or {}

        transaction_id = data.get("transaction_id")

        transaction = approved_transactions.get(transaction_id)

        if not transaction:
            return jsonify({"error": "Invalid transaction."}), 404

        if transaction["status"] != "PENDING":
            return jsonify({"error": "Transaction is not payable."}), 400

        # Reuse an existing order for this transaction.
        if transaction["paypal_order_id"]:
            return jsonify({
                "id": transaction["paypal_order_id"]
            })

        access_token = get_paypal_access_token()

        amount = f'{transaction["amount"]:.2f}'

        order_payload = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": transaction_id,
                    "amount": {
                        "currency_code": transaction["currency"],
                        "value": amount
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

        response.raise_for_status()

        order = response.json()

        if not order.get("id"):
            return jsonify({"error": "PayPal order creation failed."}), 502

        transaction["paypal_order_id"] = order["id"]

        return jsonify({"id": order["id"]})

    except requests.RequestException:
        app.logger.exception("PayPal order creation failed.")
        return jsonify({"error": "Unable to create PayPal order."}), 502

    except Exception:
        app.logger.exception("Unexpected order creation error.")
        return jsonify({"error": "Something went wrong."}), 500


# --------------------------------------------------
# 8. CAPTURE PAYPAL PAYMENT
# --------------------------------------------------

@app.route("/capture-paypal-order", methods=["POST"])
def capture_paypal_order():

    try:
        data = request.get_json(silent=True) or {}

        transaction_id = data.get("transaction_id")
        order_id = data.get("orderID")

        transaction = approved_transactions.get(transaction_id)

        if not transaction:
            return jsonify({"error": "Invalid transaction."}), 404

        if transaction["status"] == "COMPLETED":
            return jsonify({"status": "COMPLETED"})

        if transaction["status"] != "PENDING":
            return jsonify({"error": "Transaction cannot be captured."}), 400

        if not order_id or order_id != transaction["paypal_order_id"]:
            return jsonify({"error": "Invalid PayPal order."}), 400

        access_token = get_paypal_access_token()

        # Fetch order details before capturing.
        order_response = requests.get(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders/{order_id}",
            headers={
                "Authorization": f"Bearer {access_token}"
            },
            timeout=20
        )

        order_response.raise_for_status()
        order = order_response.json()

        purchase_units = order.get("purchase_units", [])

        if order.get("status") != "APPROVED" or len(purchase_units) != 1:
            return jsonify({"error": "PayPal order is not approved."}), 400

        purchase_unit = purchase_units[0]
        order_amount = purchase_unit.get("amount", {})

        if purchase_unit.get("reference_id") != transaction_id:
            return jsonify({"error": "Transaction reference mismatch."}), 400

        if order_amount.get("currency_code") != transaction["currency"]:
            return jsonify({"error": "Currency mismatch."}), 400

        if order_amount.get("value") != f'{transaction["amount"]:.2f}':
            return jsonify({"error": "Payment amount mismatch."}), 400

        # Capture payment.
        capture_response = requests.post(
            f"{PAYPAL_BASE_URL}/v2/checkout/orders/{order_id}/capture",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
                "PayPal-Request-Id": f"capture-{order_id}"
            },
            json={},
            timeout=30
        )

        capture_response.raise_for_status()

        capture_data = capture_response.json()

        if capture_data.get("status") != "COMPLETED":
            return jsonify({"error": "Payment was not completed."}), 400

        captured_units = capture_data.get("purchase_units", [])

        if len(captured_units) != 1:
            return jsonify({"error": "Invalid capture response."}), 400

        captures = (
            captured_units[0]
            .get("payments", {})
            .get("captures", [])
        )

        if len(captures) != 1:
            return jsonify({"error": "Invalid payment capture."}), 400

        capture = captures[0]
        capture_amount = capture.get("amount", {})

        if capture.get("status") != "COMPLETED":
            return jsonify({"error": "Capture was not completed."}), 400

        if capture_amount.get("currency_code") != transaction["currency"]:
            return jsonify({"error": "Captured currency mismatch."}), 400

        if capture_amount.get("value") != f'{transaction["amount"]:.2f}':
            return jsonify({"error": "Captured amount mismatch."}), 400

        # Mark transaction as completed.
        transaction["status"] = "COMPLETED"
        transaction["capture_id"] = capture.get("id")

        return jsonify({
            "status": "COMPLETED",
            "capture_id": capture.get("id"),
            "message": "Payment completed successfully."
        })

    except requests.RequestException:
        app.logger.exception("PayPal capture failed.")
        return jsonify({"error": "Unable to verify payment with PayPal."}), 502

    except Exception:
        app.logger.exception("Unexpected capture error.")
        return jsonify({"error": "Something went wrong."}), 500


# --------------------------------------------------
# 9. PAYMENT SUCCESS PAGE
# --------------------------------------------------

@app.route("/payment/success")
def payment_success():

    return render_template("result.html", result={
        "status": "Payment Successful",
        "decision": "completed",
        "risk_score": 0,
        "amount": 0
    })


# --------------------------------------------------
# 10. PAYMENT CANCEL PAGE
# --------------------------------------------------

@app.route("/payment/cancel")
def payment_cancel():

    return redirect(url_for("index"))


# --------------------------------------------------
# 11. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
