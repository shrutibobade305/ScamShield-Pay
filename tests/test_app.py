import json
import pytest
from app import app, FEATURE_COLUMNS, model
from database import db, Transaction


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.app_context():
        db.create_all()
        with app.test_client() as client:
            yield client
        db.session.remove()
        db.drop_all()


def test_homepage(client):
    """Test that homepage loads successfully with HTTP 200 and expected markup."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"ScamShield Pay" in response.data
    assert b"Transaction Risk Analyzer" in response.data
    assert b"Quick Test Scenarios" in response.data


def test_health_check(client):
    """Test the /health monitoring endpoint and database connectivity."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["model_loaded"] is True


def test_predict_safe_transaction_persists_to_db(client):
    """
    Test a low-risk transaction.
    Expected: Risk < 35%, status 'Safe', saved to database with status 'APPROVED', payment 'PENDING'.
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
    tx_id = data["transaction_id"]

    # Verify transaction is persisted in database
    with app.app_context():
        record = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert record is not None
        assert record.amount == 45.0
        assert record.risk_classification == "Safe"
        assert record.transaction_status == "APPROVED"
        assert record.payment_status == "PENDING"
        assert record.currency == "USD"


def test_predict_suspicious_transaction_persists_to_db(client):
    """
    Test a moderate-risk transaction.
    Expected: 35% <= Risk < 70%, status 'Suspicious', saved to DB with status 'UNDER_REVIEW', payment 'BLOCKED_SUSPICIOUS'.
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
    tx_id = data["transaction_id"]

    # Verify transaction is persisted in database with blocked payment status
    with app.app_context():
        record = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert record is not None
        assert record.risk_classification == "Suspicious"
        assert record.transaction_status == "UNDER_REVIEW"
        assert record.payment_status == "BLOCKED_SUSPICIOUS"


def test_predict_high_risk_transaction_persists_to_db(client):
    """
    Test a severe scam / high-risk transaction.
    Expected: Risk >= 70%, status 'High Risk', saved to DB with status 'BLOCKED', payment 'BLOCKED_HIGH_RISK'.
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
    tx_id = data["transaction_id"]

    # Verify transaction is persisted in database as strictly blocked
    with app.app_context():
        record = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert record is not None
        assert record.risk_classification == "High Risk"
        assert record.transaction_status == "BLOCKED"
        assert record.payment_status == "BLOCKED_HIGH_RISK"


def test_predict_validation_errors(client):
    """Test input validation for non-positive amount and out-of-range values."""
    resp1 = client.post("/predict", json={"amount": -50.0, "transaction_hour": 10})
    assert resp1.status_code == 400

    resp2 = client.post("/predict", json={"amount": 0, "transaction_hour": 10})
    assert resp2.status_code == 400

    resp3 = client.post("/predict", json={"amount": 100, "transaction_hour": 25})
    assert resp3.status_code == 400

    resp4 = client.post("/predict", json={"amount": 100, "transaction_hour": 12, "international": 5})
    assert resp4.status_code == 400


def test_explainable_risk_indicators_structure(client):
    """Verify that all 5 explainable risk indicators are properly structured."""
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

    indicators = data["risk_indicators"]
    categories = [ind["category"] for ind in indicators]

    assert "Transaction Amount" in categories
    assert "Device Verification" in categories
    assert "Authentication History" in categories
    assert "Geographic Origin" in categories
    assert "Temporal Anomaly" in categories


def test_payment_gating_unapproved_rejected(client):
    """Test that /payment strictly rejects suspicious or unapproved transactions from the DB."""
    # Create suspicious transaction
    post_res = client.post("/predict", json={
        "amount": 15000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })
    tx_id = post_res.get_json()["transaction_id"]

    # Try to access payment page for this suspicious transaction
    pay_res = client.get(f"/payment?transaction_id={tx_id}")
    assert pay_res.status_code == 403
    assert b"Payment is not authorized" in pay_res.data or b"Payment Disallowed" in pay_res.data


def test_payment_page_safe_transaction(client):
    """Test that a Safe approved transaction correctly opens the payment page from the DB."""
    post_res = client.post("/predict", json={
        "amount": 29.99,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = post_res.get_json()["transaction_id"]

    pay_res = client.get(f"/payment?transaction_id={tx_id}")
    assert pay_res.status_code == 200
    assert b"Secure Payment" in pay_res.data
    assert b"29.99" in pay_res.data
    assert tx_id.encode() in pay_res.data


def test_create_paypal_order_safe_transaction_updates_db(client):
    """Test create-paypal-order with a safe transaction and verify order ID is saved in DB."""
    post_res = client.post("/predict", json={
        "amount": 10.00,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = post_res.get_json()["transaction_id"]

    order_res = client.post("/create-paypal-order", json={"transaction_id": tx_id})
    if order_res.status_code == 200:
        order_data = order_res.get_json()
        assert "id" in order_data
        order_id = order_data["id"]

        with app.app_context():
            record = Transaction.query.filter_by(transaction_id=tx_id).first()
            assert record.paypal_order_id == order_id
            assert record.payment_status == "PENDING"
    else:
        assert order_res.status_code in (500, 502)


def test_payment_cancel_updates_db_status(client):
    """Test that cancelling payment updates the database payment_status to CANCELLED."""
    post_res = client.post("/predict", json={
        "amount": 25.00,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = post_res.get_json()["transaction_id"]

    # Cancel payment
    cancel_res = client.get(f"/payment/cancel?transaction_id={tx_id}")
    assert cancel_res.status_code == 302

    with app.app_context():
        record = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert record.payment_status == "CANCELLED"


def test_payment_success_displays_db_record(client):
    """Test that payment/success retrieves and displays details from the database."""
    post_res = client.post("/predict", json={
        "amount": 55.00,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = post_res.get_json()["transaction_id"]

    # Manually mark as completed with capture ID for test verification
    with app.app_context():
        rec = Transaction.query.filter_by(transaction_id=tx_id).first()
        rec.payment_status = "COMPLETED"
        rec.paypal_capture_id = "TEST_CAP_98765"
        db.session.commit()

    succ_res = client.get(f"/payment/success?transaction_id={tx_id}")
    assert succ_res.status_code == 200
    assert b"Payment Successful" in succ_res.data
    assert b"55.00" in succ_res.data
    assert b"TEST_CAP_98765" in succ_res.data
