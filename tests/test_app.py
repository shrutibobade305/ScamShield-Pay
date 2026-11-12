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


# --------------------------------------------------
# DASHBOARD TESTS
# --------------------------------------------------

def test_dashboard_page(client):
    """Test that /dashboard and /admin routes load successfully with 200."""
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert b"ScamShield Pay" in res.data
    assert b"Executive Overview" in res.data
    assert b"Random Forest" in res.data

    res_admin = client.get("/admin")
    assert res_admin.status_code == 200
    assert b"ScamShield Pay" in res_admin.data


def test_dashboard_stats_empty_db(client):
    """Test dashboard stats endpoint handles an empty database gracefully."""
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    metrics = data["metrics"]
    assert metrics["total_transactions"] == 0
    assert metrics["safe_transactions"] == 0
    assert metrics["suspicious_transactions"] == 0
    assert metrics["high_risk_transactions"] == 0
    assert metrics["successful_payments"] == 0
    assert metrics["blocked_transactions"] == 0
    assert metrics["total_volume_usd"] == 0.0
    assert metrics["avg_risk_score"] == 0.0
    assert metrics["approval_rate"] == 0.0


def test_dashboard_stats_with_data(client):
    """Test that dashboard stats aggregate real transaction data accurately."""
    # 1. Safe transaction ($50)
    client.post("/predict", json={
        "amount": 50.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })

    # 2. Suspicious transaction ($12,000)
    client.post("/predict", json={
        "amount": 12000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })

    # 3. High-Risk transaction ($35,000)
    client.post("/predict", json={
        "amount": 35000.0,
        "transaction_hour": 1,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 4
    })

    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.get_json()
    metrics = data["metrics"]
    assert metrics["total_transactions"] == 3
    assert metrics["safe_transactions"] == 1
    assert metrics["suspicious_transactions"] == 1
    assert metrics["high_risk_transactions"] == 1
    assert metrics["total_volume_usd"] == 47050.0
    assert metrics["approval_rate"] == 33.3


def test_dashboard_charts_api(client):
    """Test that /api/dashboard/charts returns chart datasets."""
    # Add a transaction
    client.post("/predict", json={
        "amount": 45.0,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })

    res = client.get("/api/dashboard/charts?date_range=7d")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "risk_distribution" in data
    assert "payment_outcomes" in data
    assert "daily_volume" in data
    assert "risk_trend" in data
    assert data["risk_distribution"]["Safe"] >= 1


def test_dashboard_transactions_filter_and_pagination(client):
    """Test filtering transactions by risk classification and pagination."""
    # Add safe transaction
    res1 = client.post("/predict", json={
        "amount": 25.0,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    safe_tx_id = res1.get_json()["transaction_id"]

    # Add high-risk transaction
    res2 = client.post("/predict", json={
        "amount": 40000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 5
    })
    danger_tx_id = res2.get_json()["transaction_id"]

    # Filter by Safe
    res_safe = client.get("/api/dashboard/transactions?risk=Safe")
    assert res_safe.status_code == 200
    data_safe = res_safe.get_json()
    assert data_safe["total"] == 1
    assert data_safe["transactions"][0]["transaction_id"] == safe_tx_id

    # Filter by High Risk
    res_high = client.get("/api/dashboard/transactions?risk=High%20Risk")
    assert res_high.status_code == 200
    data_high = res_high.get_json()
    assert data_high["total"] == 1
    assert data_high["transactions"][0]["transaction_id"] == danger_tx_id

    # Search by ID prefix
    prefix = safe_tx_id[:8]
    res_search = client.get(f"/api/dashboard/transactions?search={prefix}")
    assert res_search.status_code == 200
    assert res_search.get_json()["total"] == 1


def test_dashboard_transaction_detail(client):
    """Test deep transaction inspection endpoint."""
    res = client.post("/predict", json={
        "amount": 100.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = res.get_json()["transaction_id"]

    detail_res = client.get(f"/api/dashboard/transactions/{tx_id}")
    assert detail_res.status_code == 200
    data = detail_res.get_json()
    assert data["status"] == "success"
    tx = data["transaction"]
    assert tx["transaction_id"] == tx_id
    assert tx["amount"] == 100.0

    exp = data["explainability"]
    assert "indicators" in exp
    assert "recommendation" in exp
    assert len(exp["indicators"]) == 5

    # Test invalid and non-existent IDs
    assert client.get("/api/dashboard/transactions/invalid!id").status_code == 400
    assert client.get("/api/dashboard/transactions/nonexistent_tx_1234567890").status_code == 404


def test_dashboard_seed_and_clear_demo(client):
    """Test seeding demo transactions and clearing them."""
    seed_res = client.post("/api/dashboard/seed-demo")
    assert seed_res.status_code == 200
    seed_data = seed_res.get_json()
    assert seed_data["status"] == "success"
    assert seed_data["count"] == 14

    # Verify transactions count
    stats_res = client.get("/api/dashboard/stats")
    assert stats_res.get_json()["metrics"]["total_transactions"] == 14

    # Clear demo
    clear_res = client.post("/api/dashboard/clear-demo")
    assert clear_res.status_code == 200
    clear_data = clear_res.get_json()
    assert clear_data["status"] == "success"
    assert clear_data["count"] == 14

    # Verify cleared
    stats_res2 = client.get("/api/dashboard/stats")
    assert stats_res2.get_json()["metrics"]["total_transactions"] == 0


# --------------------------------------------------
# TRANSACTION HISTORY & DETAILS TESTS
# --------------------------------------------------

def test_transactions_history_page_loads(client):
    """Test that /transactions and /history load successfully."""
    res1 = client.get("/transactions")
    assert res1.status_code == 200
    assert b"Transaction History" in res1.data
    assert b"All Records" in res1.data

    res2 = client.get("/history")
    assert res2.status_code == 200
    assert b"Transaction History" in res2.data


def test_transactions_history_empty_db(client):
    """Test that transaction history handles empty database cleanly."""
    res = client.get("/transactions")
    assert res.status_code == 200
    assert b"No transactions found" in res.data

    # JSON response
    res_json = client.get("/transactions?format=json")
    assert res_json.status_code == 200
    data = res_json.get_json()
    assert data["status"] == "success"
    assert data["total"] == 0
    assert len(data["transactions"]) == 0


def test_transactions_history_filtering_and_search(client):
    """
    Test filtering by Safe, Suspicious, High Risk, Payment Successful,
    Payment Failed, Blocked, and search by transaction ID.
    """
    # 1. Safe + Completed transaction
    res_safe = client.post("/predict", json={
        "amount": 25.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    safe_id = res_safe.get_json()["transaction_id"]
    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=safe_id).first()
        t.payment_status = "COMPLETED"
        t.paypal_capture_id = "TEST-CAP-111"
        db.session.commit()

    # 2. Suspicious + Under Review + Blocked Suspicious transaction
    res_susp = client.post("/predict", json={
        "amount": 10000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 1
    })
    susp_id = res_susp.get_json()["transaction_id"]

    # 3. High Risk + Blocked transaction
    res_danger = client.post("/predict", json={
        "amount": 50000.0,
        "transaction_hour": 1,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 4
    })
    danger_id = res_danger.get_json()["transaction_id"]

    # 4. Another transaction manually set to FAILED
    res_failed = client.post("/predict", json={
        "amount": 35.0,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    failed_id = res_failed.get_json()["transaction_id"]
    with app.app_context():
        tf = Transaction.query.filter_by(transaction_id=failed_id).first()
        tf.payment_status = "FAILED"
        db.session.commit()

    # Test Filter: Safe
    safe_query = client.get("/transactions?risk=Safe&format=json")
    assert safe_query.status_code == 200
    safe_data = safe_query.get_json()
    assert safe_data["total"] == 2  # safe_id and failed_id both have low risk
    assert any(t["transaction_id"] == safe_id for t in safe_data["transactions"])

    # Test Filter: Suspicious
    susp_query = client.get("/transactions?risk=Suspicious&format=json")
    assert susp_query.status_code == 200
    assert susp_query.get_json()["total"] == 1
    assert susp_query.get_json()["transactions"][0]["transaction_id"] == susp_id

    # Test Filter: High Risk
    danger_query = client.get("/transactions?risk=High%20Risk&format=json")
    assert danger_query.status_code == 200
    assert danger_query.get_json()["total"] == 1
    assert danger_query.get_json()["transactions"][0]["transaction_id"] == danger_id

    # Test Filter: Payment Successful (COMPLETED)
    comp_query = client.get("/transactions?status=COMPLETED&format=json")
    assert comp_query.status_code == 200
    assert comp_query.get_json()["total"] == 1
    assert comp_query.get_json()["transactions"][0]["transaction_id"] == safe_id

    # Test Filter: Payment Failed
    fail_query = client.get("/transactions?status=FAILED&format=json")
    assert fail_query.status_code == 200
    assert fail_query.get_json()["total"] == 1
    assert fail_query.get_json()["transactions"][0]["transaction_id"] == failed_id

    # Test Filter: Blocked
    block_query = client.get("/transactions?status=BLOCKED&format=json")
    assert block_query.status_code == 200
    # Suspicious (BLOCKED_SUSPICIOUS) + High Risk (BLOCKED_HIGH_RISK) are both blocked
    assert block_query.get_json()["total"] == 2

    # Test Search by Transaction ID
    search_query = client.get(f"/transactions?search={safe_id[:10]}&format=json")
    assert search_query.status_code == 200
    assert search_query.get_json()["total"] == 1
    assert search_query.get_json()["transactions"][0]["transaction_id"] == safe_id


def test_transactions_history_pagination(client):
    """Test pagination on transactions history."""
    # Seed 5 transactions
    for i in range(5):
        client.post("/predict", json={
            "amount": 10.0 + i,
            "transaction_hour": 12,
            "international": 0,
            "new_device": 0,
            "failed_attempts": 0
        })

    # Page 1, limit 2
    p1 = client.get("/transactions?page=1&limit=2&format=json")
    assert p1.status_code == 200
    d1 = p1.get_json()
    assert d1["total"] == 5
    assert d1["page"] == 1
    assert d1["per_page"] == 2
    assert d1["total_pages"] == 3
    assert len(d1["transactions"]) == 2
    assert d1["has_prev"] is False
    assert d1["has_next"] is True

    # Page 2, limit 2
    p2 = client.get("/transactions?page=2&limit=2&format=json")
    assert p2.status_code == 200
    d2 = p2.get_json()
    assert d2["page"] == 2
    assert len(d2["transactions"]) == 2
    assert d2["has_prev"] is True
    assert d2["has_next"] is True


def test_transaction_detail_page_and_json(client):
    """Test the standalone transaction details page and API."""
    res = client.post("/predict", json={
        "amount": 75.0,
        "transaction_hour": 15,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = res.get_json()["transaction_id"]

    # Test HTML view
    html_res = client.get(f"/transactions/{tx_id}")
    assert html_res.status_code == 200
    assert b"Transaction Inspection" in html_res.data
    assert b"75.00" in html_res.data
    assert b"Behavioral Input Features" in html_res.data
    assert b"Explainable AI Risk Indicators" in html_res.data
    assert b"Security Protocol" in html_res.data

    # Test JSON view
    json_res = client.get(f"/transactions/{tx_id}?format=json")
    assert json_res.status_code == 200
    data = json_res.get_json()
    assert data["status"] == "success"
    assert data["transaction"]["transaction_id"] == tx_id
    assert len(data["explainability"]["indicators"]) == 5
    assert "recommendation" in data["explainability"]

    # Test invalid format ID (400) and nonexistent ID (404)
    assert client.get("/transactions/invalid!id_").status_code == 400
    assert client.get("/transactions/nonexistent_1234567890").status_code == 404


# --------------------------------------------------
# PAYPAL PAYMENT SECURITY AUDIT TESTS
# --------------------------------------------------

def test_create_paypal_order_blocks_suspicious_and_high_risk(client):
    """Verify that Suspicious and High-Risk transactions cannot create PayPal orders (HTTP 403)."""
    # 1. Suspicious transaction
    susp_res = client.post("/predict", json={
        "amount": 12000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })
    susp_tx_id = susp_res.get_json()["transaction_id"]

    order_susp = client.post("/create-paypal-order", json={"transaction_id": susp_tx_id})
    assert order_susp.status_code == 403
    assert "blocked" in order_susp.get_json()["error"].lower()

    # 2. High-Risk transaction
    danger_res = client.post("/predict", json={
        "amount": 35000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 5
    })
    danger_tx_id = danger_res.get_json()["transaction_id"]

    order_danger = client.post("/create-paypal-order", json={"transaction_id": danger_tx_id})
    assert order_danger.status_code == 403
    assert "blocked" in order_danger.get_json()["error"].lower()


def test_create_paypal_order_missing_or_expired_transaction(client):
    """Verify 404 for missing transaction and 410 for expired transaction."""
    # Missing transaction
    res_missing = client.post("/create-paypal-order", json={"transaction_id": "nonexistent_tx_1234567890"})
    assert res_missing.status_code == 404

    # Expired transaction (> 3 hours old)
    safe_res = client.post("/predict", json={
        "amount": 30.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = safe_res.get_json()["transaction_id"]

    from datetime import datetime, timedelta
    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=tx_id).first()
        t.created_at = datetime.utcnow() - timedelta(hours=4)
        db.session.commit()

    order_expired = client.post("/create-paypal-order", json={"transaction_id": tx_id})
    assert order_expired.status_code == 410
    assert "expired" in order_expired.get_json()["error"].lower()


def test_capture_paypal_order_blocks_unsafe_and_mismatched(client):
    """Verify that capture rejects non-safe transactions and mismatched order IDs."""
    # 1. Block capture for Suspicious transaction
    susp_res = client.post("/predict", json={
        "amount": 8000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })
    susp_tx_id = susp_res.get_json()["transaction_id"]

    cap_susp = client.post("/capture-paypal-order", json={
        "transaction_id": susp_tx_id,
        "orderID": "5C123456789012345"
    })
    assert cap_susp.status_code == 403

    # 2. Block capture with invalid order ID format
    safe_res = client.post("/predict", json={
        "amount": 20.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    safe_tx_id = safe_res.get_json()["transaction_id"]

    cap_invalid_id = client.post("/capture-paypal-order", json={
        "transaction_id": safe_tx_id,
        "orderID": "short!"
    })
    assert cap_invalid_id.status_code == 400


def test_capture_paypal_order_idempotency_prevents_duplicate_charge(client):
    """
    Verify that if a transaction is already COMPLETED, calling /capture-paypal-order
    returns the existing capture ID with HTTP 200 without issuing duplicate charges.
    """
    res = client.post("/predict", json={
        "amount": 50.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = res.get_json()["transaction_id"]

    # Mark as completed in SQLite
    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=tx_id).first()
        t.payment_status = "COMPLETED"
        t.transaction_status = "COMPLETED"
        t.paypal_order_id = "5C987654321012345"
        t.paypal_capture_id = "TEST-CAPTURE-ID-999"
        db.session.commit()

    # Second capture attempt
    dup_res = client.post("/capture-paypal-order", json={
        "transaction_id": tx_id,
        "orderID": "5C987654321012345"
    })
    assert dup_res.status_code == 200
    dup_data = dup_res.get_json()
    assert dup_data["status"] == "COMPLETED"
    assert dup_data["capture_id"] == "TEST-CAPTURE-ID-999"
    assert dup_data.get("idempotent") is True


def test_cancel_and_fail_paypal_order_endpoints(client):
    """Test /cancel-paypal-order and /fail-paypal-order lifecycle hooks."""
    # Cancel endpoint
    res1 = client.post("/predict", json={
        "amount": 40.0,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id1 = res1.get_json()["transaction_id"]

    cancel_res = client.post("/cancel-paypal-order", json={"transaction_id": tx_id1})
    assert cancel_res.status_code == 200
    assert cancel_res.get_json()["status"] == "CANCELLED"

    with app.app_context():
        rec1 = Transaction.query.filter_by(transaction_id=tx_id1).first()
        assert rec1.payment_status == "CANCELLED"

    # Fail endpoint
    res2 = client.post("/predict", json={
        "amount": 40.0,
        "transaction_hour": 12,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id2 = res2.get_json()["transaction_id"]

    fail_res = client.post("/fail-paypal-order", json={
        "transaction_id": tx_id2,
        "error": "Card declined in PayPal Sandbox"
    })
    assert fail_res.status_code == 200
    assert fail_res.get_json()["status"] == "FAILED"

    with app.app_context():
        rec2 = Transaction.query.filter_by(transaction_id=tx_id2).first()
        assert rec2.payment_status == "FAILED"


def test_payment_success_blocks_unverified_transactions(client):
    """
    Verify that /payment/success strictly blocks any transaction that has NOT
    been verified as COMPLETED on the server.
    """
    res = client.post("/predict", json={
        "amount": 60.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    tx_id = res.get_json()["transaction_id"]

    # Transaction is PENDING (not captured)
    succ_res = client.get(f"/payment/success?transaction_id={tx_id}")
    assert succ_res.status_code == 400
    assert b"Payment Not Completed" in succ_res.data

    # Now verify that once COMPLETED, it renders successfully
    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=tx_id).first()
        t.payment_status = "COMPLETED"
        t.paypal_capture_id = "VERIFIED-CAP-777"
        db.session.commit()

    succ_res2 = client.get(f"/payment/success?transaction_id={tx_id}")
    assert succ_res2.status_code == 200
    assert b"Payment Successful" in succ_res2.data
    assert b"VERIFIED-CAP-777" in succ_res2.data


# --------------------------------------------------
# STEP-UP VERIFICATION WORKFLOW TESTS
# --------------------------------------------------

def test_verify_page_gating(client):
    """
    Test access control on /verify/<tx_id>:
    - Safe transactions redirect directly to /payment (MFA bypass)
    - Suspicious transactions render verify.html (HTTP 200)
    - High-Risk transactions are blocked from verification (HTTP 403)
    """
    # 1. Safe transaction
    res_safe = client.post("/predict", json={
        "amount": 50.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    safe_tx_id = res_safe.get_json()["transaction_id"]

    v_safe = client.get(f"/verify/{safe_tx_id}")
    assert v_safe.status_code == 302
    assert f"/payment?transaction_id={safe_tx_id}" in v_safe.headers["Location"]

    # 2. Suspicious transaction
    res_susp = client.post("/predict", json={
        "amount": 15000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })
    susp_tx_id = res_susp.get_json()["transaction_id"]

    v_susp = client.get(f"/verify/{susp_tx_id}")
    assert v_susp.status_code == 200
    assert b"Step-Up Security Verification" in v_susp.data
    assert b"Demo Verification Simulator" in v_susp.data

    # 3. High Risk transaction
    res_high = client.post("/predict", json={
        "amount": 85000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 5
    })
    high_tx_id = res_high.get_json()["transaction_id"]

    v_high = client.get(f"/verify/{high_tx_id}")
    assert v_high.status_code == 403
    assert b"Verification Not Permitted" in v_high.data or b"High Risk" in v_high.data


def test_request_otp_suspicious_transaction(client):
    """
    Test generating an OTP challenge for a suspicious transaction:
    - Returns JSON with success, attempts_left: 3, demo_mode: True
    - Stored in SQLite as SHA-256 hash (never plain text)
    """
    res = client.post("/predict", json={
        "amount": 12000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 1
    })
    tx_id = res.get_json()["transaction_id"]

    otp_res = client.post("/api/verification/request-otp", json={"transaction_id": tx_id})
    assert otp_res.status_code == 200
    data = otp_res.get_json()

    assert data["status"] == "success"
    assert data["attempts_left"] == 3
    assert "demo_code" in data
    demo_code = data["demo_code"]
    assert len(demo_code) == 6
    assert demo_code.isdigit()

    # Check database persistence and SHA-256 hash
    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert t.verification_status == "PENDING"
        assert t.otp_attempts_left == 3
        assert t.otp_code_hash is not None
        assert t.otp_expires_at is not None
        # Verify plain text OTP is NOT stored in database
        assert t.otp_code_hash != demo_code
        # Verify hash match
        from datetime import datetime
        import hashlib
        assert hashlib.sha256(demo_code.encode("utf-8")).hexdigest() == t.otp_code_hash


def test_verify_otp_attempt_limits_and_locking(client):
    """
    Test rate-limiting and lockout:
    - Submitting 3 incorrect OTPs decrements attempts to 0 and locks the transaction.
    - Locked transaction cannot request OTP or verify further.
    """
    res = client.post("/predict", json={
        "amount": 14000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })
    tx_id = res.get_json()["transaction_id"]

    # Request initial OTP
    client.post("/api/verification/request-otp", json={"transaction_id": tx_id})

    # Wrong attempt 1
    r1 = client.post("/api/verification/verify-otp", json={"transaction_id": tx_id, "otp": "000000"})
    assert r1.status_code == 400
    assert r1.get_json()["attempts_left"] == 2

    # Wrong attempt 2
    r2 = client.post("/api/verification/verify-otp", json={"transaction_id": tx_id, "otp": "111111"})
    assert r2.status_code == 400
    assert r2.get_json()["attempts_left"] == 1

    # Wrong attempt 3 (Lockout!)
    r3 = client.post("/api/verification/verify-otp", json={"transaction_id": tx_id, "otp": "222222"})
    assert r3.status_code == 403
    d3 = r3.get_json()
    assert d3["status"] == "locked"
    assert d3["attempts_left"] == 0

    # Verify locked in DB
    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert t.verification_status == "FAILED"
        assert t.transaction_status == "BLOCKED"
        assert t.otp_attempts_left == 0

    # Further OTP requests are blocked
    lock_req = client.post("/api/verification/request-otp", json={"transaction_id": tx_id})
    assert lock_req.status_code == 403


def test_verify_otp_expiration(client):
    """Test that expired OTPs are rejected with HTTP 400 and expired status."""
    from datetime import datetime, timedelta

    res = client.post("/predict", json={
        "amount": 10000.0,
        "transaction_hour": 4,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 1
    })
    tx_id = res.get_json()["transaction_id"]

    otp_res = client.post("/api/verification/request-otp", json={"transaction_id": tx_id})
    code = otp_res.get_json()["demo_code"]

    # Manually expire the OTP in DB
    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=tx_id).first()
        t.otp_expires_at = datetime.utcnow() - timedelta(minutes=5)
        db.session.commit()

    ver_res = client.post("/api/verification/verify-otp", json={"transaction_id": tx_id, "otp": code})
    assert ver_res.status_code == 400
    assert ver_res.get_json()["status"] == "expired"


def test_verify_otp_success_and_auditable_policy(client):
    """
    Test successful OTP verification:
    - Verifies cryptographic hash
    - Evaluates explicit auditable policy
    - Updates transaction_status to 'APPROVED_POST_MFA', verification_status to 'VERIFIED'
    - Updates payment_status to 'PENDING'
    """
    res = client.post("/predict", json={
        "amount": 9500.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 1
    })
    tx_id = res.get_json()["transaction_id"]

    otp_res = client.post("/api/verification/request-otp", json={"transaction_id": tx_id})
    code = otp_res.get_json()["demo_code"]

    ver_res = client.post("/api/verification/verify-otp", json={"transaction_id": tx_id, "otp": code})
    assert ver_res.status_code == 200
    d = ver_res.get_json()
    assert d["status"] == "success"
    assert d["verified"] is True
    assert "redirect_url" in d

    with app.app_context():
        t = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert t.verification_status == "VERIFIED"
        assert t.transaction_status == "APPROVED_POST_MFA"
        assert t.payment_status == "PENDING"
        assert "MFA Verified" in t.verification_notes


def test_auditable_policy_rejects_excessive_velocity_anomaly(client):
    """
    Test that even if the OTP matches, the auditable policy rejects a suspicious
    transaction if failed_attempts exceeds allowable threshold (> 5).
    """
    with app.app_context():
        import secrets
        import hashlib
        from datetime import datetime, timedelta

        tx_id = secrets.token_urlsafe(16)
        code = "789123"
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()

        # Suspicious risk score but excessive prior failed attempts (6)
        t = Transaction(
            transaction_id=tx_id,
            amount=8000.0,
            currency="USD",
            transaction_hour=3,
            international=1,
            new_device=1,
            failed_attempts=6,  # Exceeds policy threshold (<= 5)
            risk_score=55.0,
            risk_classification="Suspicious",
            transaction_status="UNDER_REVIEW",
            payment_status="BLOCKED_SUSPICIOUS",
            verification_status="PENDING",
            otp_code_hash=code_hash,
            otp_expires_at=datetime.utcnow() + timedelta(minutes=10),
            otp_attempts_left=3
        )
        db.session.add(t)
        db.session.commit()

    ver_res = client.post("/api/verification/verify-otp", json={"transaction_id": tx_id, "otp": code})
    assert ver_res.status_code == 403
    d = ver_res.get_json()
    assert d["status"] == "blocked"
    assert d["verified"] is False
    assert "velocity" in d["error"].lower()

    with app.app_context():
        rec = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert rec.verification_status == "FAILED"
        assert rec.transaction_status == "BLOCKED"


def test_post_mfa_payment_authorization_flow(client, monkeypatch):
    """
    Test complete lifecycle from Suspicious -> Blocked -> MFA Verified -> Payment Allowed:
    1. Initial suspicious transaction cannot access /payment or create PayPal order.
    2. Step-up verification succeeds.
    3. Transaction can now access /payment and create PayPal order.
    """
    # Create suspicious transaction (Risk ~53.1%)
    res = client.post("/predict", json={
        "amount": 12000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })
    data = res.get_json()
    assert data["status"] == "Suspicious"
    tx_id = data["transaction_id"]

    # 1. Access to /payment is BLOCKED before verification
    pay_res = client.get(f"/payment?transaction_id={tx_id}")
    assert pay_res.status_code == 403

    # Order creation is BLOCKED before verification
    ord_res = client.post("/create-paypal-order", json={"transaction_id": tx_id})
    assert ord_res.status_code == 403

    # 2. Request OTP and verify
    req_res = client.post("/api/verification/request-otp", json={"transaction_id": tx_id})
    code = req_res.get_json()["demo_code"]

    ver_res = client.post("/api/verification/verify-otp", json={"transaction_id": tx_id, "otp": code})
    assert ver_res.status_code == 200

    # 3. Access to /payment is now ALLOWED
    pay_res_after = client.get(f"/payment?transaction_id={tx_id}")
    assert pay_res_after.status_code == 200
    assert b"Secure Payment" in pay_res_after.data
    assert tx_id.encode() in pay_res_after.data

    # Mock PayPal API to test PayPal order creation for post-MFA approved transaction
    monkeypatch.setattr("app.get_paypal_access_token", lambda: "fake-mock-token-xyz")

    class MockPayPalOrderResponse:
        status_code = 201
        ok = True
        headers = {"content-type": "application/json"}

        def json(self):
            return {"id": "MOCK-MFA-ORDER-999", "status": "CREATED"}

    monkeypatch.setattr("requests.post", lambda url, **kwargs: MockPayPalOrderResponse())

    ord_res_after = client.post("/create-paypal-order", json={"transaction_id": tx_id})
    assert ord_res_after.status_code == 200
    assert ord_res_after.get_json()["id"] == "MOCK-MFA-ORDER-999"

    with app.app_context():
        rec = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert rec.paypal_order_id == "MOCK-MFA-ORDER-999"
        assert rec.payment_status == "PENDING"



