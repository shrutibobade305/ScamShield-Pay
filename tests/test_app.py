import json
import pytest
from app import app, approved_transactions, FEATURE_COLUMNS, model


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage(client):
    """Test that homepage loads successfully with HTTP 200 and expected markup."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"ScamShield Pay" in response.data
    assert b"Transaction Risk Analyzer" in response.data


def test_health_check(client):
    """Test the /health monitoring endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_safe_transaction(client):
    """
    Test a low-risk transaction.
    Expected: Risk < 35%, status 'Safe', decision 'approved', eligible for payment.
    """
    payload = {
        "amount": 45.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "Safe"
    assert data["decision"] == "approved"
    assert data["risk_score"] < 35.0
    assert "transaction_id" in data

    # Verify transaction is registered in memory as PENDING
    tx_id = data["transaction_id"]
    assert tx_id in approved_transactions
    assert approved_transactions[tx_id]["status"] == "PENDING"
    assert approved_transactions[tx_id]["amount"] == 45.0


def test_predict_suspicious_transaction(client):
    """
    Test a moderate-risk transaction.
    Expected: 35% <= Risk < 70%, status 'Suspicious', decision 'review'.
    Must NOT be eligible for payment.
    """
    payload = {
        "amount": 15000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "Suspicious"
    assert data["decision"] == "review"
    assert 35.0 <= data["risk_score"] < 70.0

    # Verify transaction is NOT registered for payment
    tx_id = data["transaction_id"]
    assert tx_id not in approved_transactions


def test_predict_high_risk_transaction(client):
    """
    Test a severe scam / high-risk transaction.
    Expected: Risk >= 70%, status 'High Risk', decision 'blocked'.
    Must NOT be eligible for payment.
    """
    payload = {
        "amount": 35000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "High Risk"
    assert data["decision"] == "blocked"
    assert data["risk_score"] >= 70.0

    # Verify transaction is NOT registered for payment
    tx_id = data["transaction_id"]
    assert tx_id not in approved_transactions


def test_predict_validation_errors(client):
    """Test input validation for non-positive amount and out-of-range values."""
    # Negative amount
    resp1 = client.post("/predict", json={"amount": -50.0, "transaction_hour": 10})
    assert resp1.status_code == 400

    # Zero amount
    resp2 = client.post("/predict", json={"amount": 0, "transaction_hour": 10})
    assert resp2.status_code == 400

    # Invalid hour
    resp3 = client.post("/predict", json={"amount": 100, "transaction_hour": 25})
    assert resp3.status_code == 400

    # Invalid international flag
    resp4 = client.post("/predict", json={"amount": 100, "transaction_hour": 12, "international": 5})
    assert resp4.status_code == 400


def test_payment_gating_missing_and_unapproved(client):
    """Test that /payment rejects requests without valid safe transaction_id."""
    # No transaction_id
    resp_none = client.get("/payment")
    assert resp_none.status_code == 400
    assert b"No transaction ID provided" in resp_none.data

    # Random / forged transaction_id
    resp_fake = client.get("/payment?transaction_id=fake_tx_123")
    assert resp_fake.status_code == 404
    assert b"not found" in resp_fake.data.lower() or b"invalid" in resp_fake.data.lower()


def test_payment_page_safe_transaction(client):
    """Test that a Safe approved transaction correctly opens the payment page."""
    # Submit a safe transaction
    post_res = client.post("/predict", json={
        "amount": 29.99,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = post_res.get_json()["transaction_id"]

    # Open payment page
    pay_res = client.get(f"/payment?transaction_id={tx_id}")
    assert pay_res.status_code == 200
    assert b"Secure Payment" in pay_res.data
    assert b"29.99" in pay_res.data
    assert tx_id.encode() in pay_res.data


def test_create_paypal_order_unapproved_rejected(client):
    """Test that create-paypal-order rejects invalid or non-approved transaction IDs."""
    res = client.post("/create-paypal-order", json={"transaction_id": "nonexistent_id"})
    assert res.status_code == 404


def test_create_paypal_order_safe_transaction(client):
    """Test create-paypal-order with a real safe transaction and verify PayPal order creation."""
    # Create approved transaction
    post_res = client.post("/predict", json={
        "amount": 10.00,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = post_res.get_json()["transaction_id"]

    # Request PayPal order
    order_res = client.post("/create-paypal-order", json={"transaction_id": tx_id})
    # If PayPal credentials are valid and live network is accessible:
    if order_res.status_code == 200:
        order_data = order_res.get_json()
        assert "id" in order_data
        assert approved_transactions[tx_id]["paypal_order_id"] == order_data["id"]
    else:
        # If offline or rate limited, ensure structured error response
        assert order_res.status_code in (500, 502)


def test_payment_success_and_cancel_routes(client):
    """Test /payment/success and /payment/cancel routes."""
    # Test success
    succ_res = client.get("/payment/success")
    assert succ_res.status_code == 200
    assert b"Payment Successful" in succ_res.data

    # Test cancel
    cancel_res = client.get("/payment/cancel")
    assert cancel_res.status_code == 302  # Redirects to home
