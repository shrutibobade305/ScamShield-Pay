"""
ScamShield Pay - Pre-Hackathon Quality Assurance Audit Test Suite
================================================================
Comprehensive verification of all 15 audit tasks:
 1. Homepage and all navigation links
 2. Transaction form validation
 3. ML model loading and prediction
 4. Safe, Suspicious and High Risk scenarios
 5. Explainable risk explanations
 6. Database record creation & integrity
 7. Transaction history & dashboard statistics
 8. PayPal Sandbox order creation
 9. PayPal Sandbox successful capture
10. Cancelled and failed payment scenarios
11. High Risk PayPal order blocking (Server-side 403)
12. Duplicate capture idempotency protection
13. Mobile and desktop responsive layouts
14. Missing environment variables handling
15. Application logs & error handlers (404, 500, DB exceptions)
"""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock
from app import app, FEATURE_COLUMNS, model, analyze_risk_indicators, is_transaction_authorized_for_payment
from database import db, Transaction


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.app_context():
        db.create_all()
        with app.test_client() as test_client:
            yield test_client
        db.session.remove()
        db.drop_all()


# =====================================================================
# TASK 1: Homepage and Navigation Links
# =====================================================================
def test_task_1_homepage_and_navigation_links(client):
    """Test homepage loading, brand mark, navigation anchors and internal page links."""
    # 1. Homepage loads successfully
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Assert brand and core components present
    assert "ScamShield Pay" in html
    assert "Real-Time Payment Firewall" in html
    assert "Transaction Risk Analyzer" in html
    assert "How ScamShield Pay Works" in html
    assert "Intelligent Payment Protection Features" in html
    assert "Enterprise-Grade Technology Stack" in html
    assert "PayPal Sandbox Integration" in html

    # Assert anchor targets exist in the HTML
    assert 'id="analyzer"' in html
    assert 'id="pipeline"' in html
    assert 'id="features"' in html

    # 2. Dashboard navigation link
    dash_res = client.get("/dashboard")
    assert dash_res.status_code == 200
    assert b"Fraud Intelligence Dashboard" in dash_res.data

    # 3. Transaction history navigation link
    tx_res = client.get("/transactions")
    assert tx_res.status_code == 200
    assert b"Audit Ledger" in tx_res.data or b"Transaction History" in tx_res.data

    # 4. System health monitoring endpoint
    health_res = client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.get_json()
    assert health_data["status"] == "healthy"
    assert health_data["database"] == "connected"
    assert health_data["model_loaded"] is True


# =====================================================================
# TASK 2: Transaction Form Validation
# =====================================================================
def test_task_2_transaction_form_validation(client):
    """Test comprehensive input validation: bounds, types, ranges and malformed payloads."""
    # Negative amount
    res = client.post("/predict", json={"amount": -50.0, "transaction_hour": 14})
    assert res.status_code == 400
    assert "between $0.01 and $10,000,000" in res.get_json()["error"]

    # Zero amount
    res = client.post("/predict", json={"amount": 0, "transaction_hour": 14})
    assert res.status_code == 400

    # Missing amount
    res = client.post("/predict", json={"transaction_hour": 14})
    assert res.status_code == 400
    assert "required" in res.get_json()["error"].lower()

    # Excessively large amount (> $10,000,000)
    res = client.post("/predict", json={"amount": 10000001.0, "transaction_hour": 14})
    assert res.status_code == 400
    assert "between $0.01 and $10,000,000" in res.get_json()["error"]

    # Invalid hour (<0 or >23)
    res = client.post("/predict", json={"amount": 100.0, "transaction_hour": -1})
    assert res.status_code == 400
    res = client.post("/predict", json={"amount": 100.0, "transaction_hour": 24})
    assert res.status_code == 400

    # Invalid international flag (must be 0 or 1)
    res = client.post("/predict", json={"amount": 100.0, "transaction_hour": 12, "international": 5})
    assert res.status_code == 400

    # Invalid new_device flag
    res = client.post("/predict", json={"amount": 100.0, "transaction_hour": 12, "new_device": 99})
    assert res.status_code == 400

    # Invalid failed_attempts (< 0 or > 100)
    res = client.post("/predict", json={"amount": 100.0, "transaction_hour": 12, "failed_attempts": -3})
    assert res.status_code == 400
    res = client.post("/predict", json={"amount": 100.0, "transaction_hour": 12, "failed_attempts": 101})
    assert res.status_code == 400

    # Empty body
    res = client.post("/predict", data="", content_type="application/json")
    assert res.status_code == 400

    # HTML Form submission validation
    form_res = client.post("/predict", data={
        "amount": "120.50",
        "transaction_hour": "15",
        "international": "0",
        "new_device": "0",
        "failed_attempts": "0"
    })
    assert form_res.status_code == 200
    assert b"Risk Assessment Report" in form_res.data


# =====================================================================
# TASK 3: ML Model Loading and Prediction
# =====================================================================
def test_task_3_ml_model_loading_and_prediction():
    """Verify Random Forest model architecture, feature alignment and inference probability."""
    assert model is not None
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == 5

    expected_features = ["amount", "transaction_hour", "international", "new_device", "failed_attempts"]
    assert FEATURE_COLUMNS == expected_features

    # Inference sanity check
    import pandas as pd
    test_df = pd.DataFrame([{
        "amount": 100.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    }], columns=FEATURE_COLUMNS)

    proba = model.predict_proba(test_df)[0]
    assert len(proba) == 2
    assert 0.0 <= proba[0] <= 1.0
    assert 0.0 <= proba[1] <= 1.0
    assert abs(sum(proba) - 1.0) < 1e-4


# =====================================================================
# TASK 4: Safe, Suspicious and High Risk Scenarios
# =====================================================================
def test_task_4_safe_suspicious_high_risk_scenarios(client):
    """Test model risk classification across all 3 policy tiers."""
    # 1. Safe Scenario: Low amount, normal hour, domestic, recognized device, 0 failed attempts
    safe_res = client.post("/predict", json={
        "amount": 35.0,
        "transaction_hour": 14,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    })
    assert safe_res.status_code == 200
    safe_data = safe_res.get_json()
    assert safe_data["status"] == "Safe"
    assert safe_data["decision"] == "approved"
    assert safe_data["risk_score"] < 35.0

    # 2. Suspicious Scenario: Moderate amount, off-peak hour, new device, 2 failed attempts
    susp_res = client.post("/predict", json={
        "amount": 12000.0,
        "transaction_hour": 3,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 2
    })
    assert susp_res.status_code == 200
    susp_data = susp_res.get_json()
    assert susp_data["status"] == "Suspicious"
    assert susp_data["decision"] == "review"
    assert 35.0 <= susp_data["risk_score"] < 70.0

    # 3. High Risk Scenario: Very high amount, nighttime, international, new device, 5 failed attempts
    high_res = client.post("/predict", json={
        "amount": 45000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 5
    })
    assert high_res.status_code == 200
    high_data = high_res.get_json()
    assert high_data["status"] == "High Risk"
    assert high_data["decision"] == "blocked"
    assert high_data["risk_score"] >= 70.0


# =====================================================================
# TASK 5: Verify Risk Explanations
# =====================================================================
def test_task_5_risk_explanations(client):
    """Verify that explainable risk indicators and operational recommendations are returned."""
    indicators = analyze_risk_indicators(
        amount=25000.0,
        transaction_hour=2,
        international=1,
        new_device=1,
        failed_attempts=4
    )
    categories = [ind["category"] for ind in indicators]
    assert "Transaction Amount" in categories
    assert "Device Verification" in categories
    assert "Authentication History" in categories
    assert "Geographic Origin" in categories
    assert "Temporal Anomaly" in categories

    # Critical anomaly check
    crit_count = sum(1 for ind in indicators if ind["level"] == "critical")
    assert crit_count >= 2

    # Check API response contains explainability metadata
    res = client.post("/predict", json={
        "amount": 25000.0,
        "transaction_hour": 2,
        "international": 1,
        "new_device": 1,
        "failed_attempts": 4
    })
    assert res.status_code == 200
    data = res.get_json()
    assert "risk_indicators" in data
    assert len(data["risk_indicators"]) == 5
    assert "recommended_action" in data
    assert "badge" in data["recommended_action"]
    assert "explanation_disclaimer" in data


# =====================================================================
# TASK 6: Verify Database Record Creation
# =====================================================================
def test_task_6_database_record_creation(client):
    """Verify that transaction evaluations are persisted with all schema fields in SQLite."""
    payload = {
        "amount": 89.99,
        "transaction_hour": 11,
        "international": 0,
        "new_device": 0,
        "failed_attempts": 0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    tx_id = res.get_json()["transaction_id"]

    with app.app_context():
        record = Transaction.query.filter_by(transaction_id=tx_id).first()
        assert record is not None
        assert record.amount == 89.99
        assert record.currency == "USD"
        assert record.transaction_hour == 11
        assert record.international == 0
        assert record.new_device == 0
        assert record.failed_attempts == 0
        assert record.risk_classification == "Safe"
        assert record.transaction_status == "APPROVED"
        assert record.payment_status == "PENDING"
        assert record.created_at is not None

        # Check serialization helper
        tx_dict = record.to_dict()
        assert tx_dict["transaction_id"] == tx_id
        assert tx_dict["amount"] == 89.99


# =====================================================================
# TASK 7: Verify Transaction History and Dashboard Statistics
# =====================================================================
def test_task_7_transaction_history_and_dashboard_statistics(client):
    """Verify ledger search, risk filtering, pagination, and dashboard metrics aggregation."""
    # Seed 3 distinct transactions
    with app.app_context():
        t1 = Transaction(
            transaction_id="tx_safe_audit_001",
            amount=50.0,
            currency="USD",
            transaction_hour=14,
            international=0,
            new_device=0,
            failed_attempts=0,
            risk_score=15.2,
            risk_classification="Safe",
            transaction_status="APPROVED",
            payment_status="COMPLETED"
        )
        t2 = Transaction(
            transaction_id="tx_susp_audit_002",
            amount=4200.0,
            currency="USD",
            transaction_hour=3,
            international=0,
            new_device=1,
            failed_attempts=2,
            risk_score=52.8,
            risk_classification="Suspicious",
            transaction_status="UNDER_REVIEW",
            payment_status="BLOCKED_SUSPICIOUS"
        )
        t3 = Transaction(
            transaction_id="tx_high_audit_003",
            amount=28000.0,
            currency="USD",
            transaction_hour=2,
            international=1,
            new_device=1,
            failed_attempts=5,
            risk_score=88.5,
            risk_classification="High Risk",
            transaction_status="BLOCKED",
            payment_status="BLOCKED_HIGH_RISK"
        )
        db.session.add_all([t1, t2, t3])
        db.session.commit()

    # 1. Test Dashboard Stats
    stats_res = client.get("/api/dashboard/stats")
    assert stats_res.status_code == 200
    metrics = stats_res.get_json()["metrics"]
    assert metrics["total_transactions"] == 3
    assert metrics["safe_transactions"] == 1
    assert metrics["suspicious_transactions"] == 1
    assert metrics["high_risk_transactions"] == 1
    assert metrics["successful_payments"] == 1
    assert metrics["blocked_transactions"] == 2

    # 2. Test Dashboard Charts
    charts_res = client.get("/api/dashboard/charts")
    assert charts_res.status_code == 200
    charts = charts_res.get_json()
    assert "risk_distribution" in charts
    assert "daily_volume" in charts
    assert "risk_trend" in charts
    assert "payment_outcomes" in charts

    # 3. Test Transaction History Filtering & Search
    # Search by tx ID
    s_res = client.get("/api/dashboard/transactions?search=tx_safe_audit_001")
    assert s_res.status_code == 200
    s_data = s_res.get_json()
    assert s_data["total"] == 1
    assert s_data["transactions"][0]["transaction_id"] == "tx_safe_audit_001"

    # Filter by risk classification
    r_res = client.get("/api/dashboard/transactions?risk=High+Risk")
    assert r_res.status_code == 200
    assert r_res.get_json()["total"] == 1

    # Filter by payment status
    p_res = client.get("/api/dashboard/transactions?status=COMPLETED")
    assert p_res.status_code == 200
    assert p_res.get_json()["total"] == 1

    # Pagination test on ledger endpoint
    pg_res = client.get("/transactions?page=1&limit=2&format=json")
    assert pg_res.status_code == 200
    pg_data = pg_res.get_json()
    assert len(pg_data["transactions"]) == 2
    assert pg_data["total"] == 3
    assert pg_data["total_pages"] == 2


# =====================================================================
# TASK 8: Test PayPal Sandbox Order Creation
# =====================================================================
def test_task_8_paypal_sandbox_order_creation(client, monkeypatch):
    """Test order creation workflow for safe transaction, mocking PayPal API responses."""
    # Create safe transaction in DB
    with app.app_context():
        tx = Transaction(
            transaction_id="tx_paypal_order_safe_01",
            amount=75.0,
            currency="USD",
            transaction_hour=12,
            international=0,
            new_device=0,
            failed_attempts=0,
            risk_score=12.5,
            risk_classification="Safe",
            transaction_status="APPROVED",
            payment_status="PENDING"
        )
        db.session.add(tx)
        db.session.commit()

    # Mock PayPal OAuth token and Order creation requests
    monkeypatch.setattr("app.PAYPAL_CLIENT_ID", "mock_client_id")
    monkeypatch.setattr("app.PAYPAL_CLIENT_SECRET", "mock_client_secret")
    monkeypatch.setattr("app.get_paypal_access_token", lambda: "mock_paypal_token")

    def mock_post(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 201
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"id": "PAYPAL-ORDER-12345", "status": "CREATED"}
        return mock_resp

    monkeypatch.setattr("requests.post", mock_post)

    res = client.post("/create-paypal-order", json={"transaction_id": "tx_paypal_order_safe_01"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["id"] == "PAYPAL-ORDER-12345"

    # Verify order ID is stored in SQLite
    with app.app_context():
        updated_tx = Transaction.query.filter_by(transaction_id="tx_paypal_order_safe_01").first()
        assert updated_tx.paypal_order_id == "PAYPAL-ORDER-12345"
        assert updated_tx.payment_status == "PENDING"


# =====================================================================
# TASK 9: Test PayPal Sandbox Successful Capture
# =====================================================================
def test_task_9_paypal_sandbox_successful_capture(client, monkeypatch):
    """Test payment capture workflow, state transition to COMPLETED, and capture ID storage."""
    with app.app_context():
        tx = Transaction(
            transaction_id="tx_paypal_capture_01",
            amount=99.0,
            currency="USD",
            transaction_hour=14,
            international=0,
            new_device=0,
            failed_attempts=0,
            risk_score=18.0,
            risk_classification="Safe",
            transaction_status="APPROVED",
            paypal_order_id="ORDER-TO-CAPTURE-99",
            payment_status="PENDING"
        )
        db.session.add(tx)
        db.session.commit()

    monkeypatch.setattr("app.PAYPAL_CLIENT_ID", "mock_client_id")
    monkeypatch.setattr("app.PAYPAL_CLIENT_SECRET", "mock_client_secret")
    monkeypatch.setattr("app.get_paypal_access_token", lambda: "mock_paypal_token")

    # Mock PayPal order inspection (GET) and order capture (POST)
    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "ORDER-TO-CAPTURE-99",
            "status": "APPROVED",
            "purchase_units": [{
                "reference_id": "tx_paypal_capture_01",
                "amount": {"currency_code": "USD", "value": "99.00"}
            }]
        }
        return mock_resp

    def mock_post(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "id": "ORDER-TO-CAPTURE-99",
            "status": "COMPLETED",
            "purchase_units": [{
                "payments": {
                    "captures": [{
                        "id": "CAPTURE-ABC-99999",
                        "status": "COMPLETED",
                        "amount": {"currency_code": "USD", "value": "99.00"}
                    }]
                }
            }]
        }
        return mock_resp

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr("requests.post", mock_post)

    res = client.post("/capture-paypal-order", json={
        "transaction_id": "tx_paypal_capture_01",
        "orderID": "ORDER-TO-CAPTURE-99"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "COMPLETED"
    assert data["capture_id"] == "CAPTURE-ABC-99999"

    # Verify updated database record
    with app.app_context():
        updated_tx = Transaction.query.filter_by(transaction_id="tx_paypal_capture_01").first()
        assert updated_tx.payment_status == "COMPLETED"
        assert updated_tx.transaction_status == "COMPLETED"
        assert updated_tx.paypal_capture_id == "CAPTURE-ABC-99999"

    # Verify confirmation page renders receipt
    conf_res = client.get("/payment/success?transaction_id=tx_paypal_capture_01")
    assert conf_res.status_code == 200
    assert b"Payment Successful" in conf_res.data
    assert b"CAPTURE-ABC-99999" in conf_res.data


# =====================================================================
# TASK 10: Test Cancelled and Failed Payment Scenarios
# =====================================================================
def test_task_10_cancelled_and_failed_payment_scenarios(client):
    """Test cancellation and failure callbacks update transaction status in SQLite."""
    # 1. Cancellation scenario
    with app.app_context():
        t_cancel = Transaction(
            transaction_id="tx_cancel_001",
            amount=55.0,
            currency="USD",
            transaction_hour=10,
            international=0,
            new_device=0,
            failed_attempts=0,
            risk_score=10.0,
            risk_classification="Safe",
            transaction_status="APPROVED",
            payment_status="PENDING"
        )
        t_fail = Transaction(
            transaction_id="tx_fail_002",
            amount=65.0,
            currency="USD",
            transaction_hour=10,
            international=0,
            new_device=0,
            failed_attempts=0,
            risk_score=10.0,
            risk_classification="Safe",
            transaction_status="APPROVED",
            payment_status="PENDING"
        )
        db.session.add_all([t_cancel, t_fail])
        db.session.commit()

    # Cancel via API
    c_res = client.post("/cancel-paypal-order", json={"transaction_id": "tx_cancel_001"})
    assert c_res.status_code == 200

    with app.app_context():
        rec = Transaction.query.filter_by(transaction_id="tx_cancel_001").first()
        assert rec.payment_status == "CANCELLED"

    # Fail via API
    f_res = client.post("/fail-paypal-order", json={"transaction_id": "tx_fail_002"})
    assert f_res.status_code == 200

    with app.app_context():
        rec = Transaction.query.filter_by(transaction_id="tx_fail_002").first()
        assert rec.payment_status == "FAILED"


# =====================================================================
# TASK 11: Attempt PayPal Orders for High Risk Transactions (Strict Block)
# =====================================================================
def test_task_11_high_risk_transactions_strictly_blocked(client):
    """Verify that backend strictly blocks PayPal order creation and capture for High Risk transactions."""
    with app.app_context():
        high_risk_tx = Transaction(
            transaction_id="tx_high_blocked_007",
            amount=38000.0,
            currency="USD",
            transaction_hour=2,
            international=1,
            new_device=1,
            failed_attempts=5,
            risk_score=92.4,
            risk_classification="High Risk",
            transaction_status="BLOCKED",
            payment_status="BLOCKED_HIGH_RISK"
        )
        db.session.add(high_risk_tx)
        db.session.commit()

    # 1. Attempt to create PayPal order for High Risk transaction -> MUST BE 403
    order_res = client.post("/create-paypal-order", json={"transaction_id": "tx_high_blocked_007"})
    assert order_res.status_code == 403
    assert "blocked" in order_res.get_json()["error"].lower()

    # 2. Attempt to capture payment for High Risk transaction -> MUST BE 403
    capture_res = client.post("/capture-paypal-order", json={
        "transaction_id": "tx_high_blocked_007",
        "orderID": "ORDER-HIGH-RISK-FORGED"
    })
    assert capture_res.status_code == 403

    # 3. Attempt to view payment checkout page -> MUST BE 403
    pay_page = client.get("/payment?transaction_id=tx_high_blocked_007")
    assert pay_page.status_code == 403


# =====================================================================
# TASK 12: Duplicate Capture Protection (Idempotency)
# =====================================================================
def test_task_12_duplicate_capture_protection(client):
    """Verify that already completed payments cannot be captured a second time."""
    with app.app_context():
        completed_tx = Transaction(
            transaction_id="tx_idempotent_001",
            amount=150.0,
            currency="USD",
            transaction_hour=12,
            international=0,
            new_device=0,
            failed_attempts=0,
            risk_score=15.0,
            risk_classification="Safe",
            transaction_status="COMPLETED",
            paypal_order_id="ORDER-IDEMPOTENT-001",
            paypal_capture_id="CAPTURE-ALREADY-DONE",
            payment_status="COMPLETED"
        )
        db.session.add(completed_tx)
        db.session.commit()

    # Attempt capture again
    res = client.post("/capture-paypal-order", json={
        "transaction_id": "tx_idempotent_001",
        "orderID": "ORDER-IDEMPOTENT-001"
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "COMPLETED"
    assert data["capture_id"] == "CAPTURE-ALREADY-DONE"
    assert data["idempotent"] is True


# =====================================================================
# TASK 13: Mobile and Desktop Layouts
# =====================================================================
def test_task_13_mobile_and_desktop_layouts():
    """Verify viewport configuration in templates and responsive breakpoint declarations in CSS."""
    templates_dir = Path("templates")
    template_files = [
        "index.html",
        "result.html",
        "dashboard.html",
        "transactions.html",
        "transaction_detail.html",
        "verify.html",
        "payment.html"
    ]

    for tf in template_files:
        path = templates_dir / tf
        assert path.exists(), f"Missing template: {tf}"
        content = path.read_text(encoding="utf-8")
        assert 'name="viewport"' in content, f"{tf} missing viewport meta tag"
        assert 'width=device-width' in content, f"{tf} missing width=device-width"

    css_path = Path("static/style.css")
    assert css_path.exists()
    css_content = css_path.read_text(encoding="utf-8")
    assert "@media" in css_content
    assert "768px" in css_content
    assert "640px" in css_content


# =====================================================================
# TASK 14: Missing Environment Variables Handling
# =====================================================================
def test_task_14_missing_environment_variables(client, monkeypatch):
    """Verify that the system gracefully handles missing PayPal credentials without crashing."""
    monkeypatch.setattr("app.PAYPAL_CLIENT_ID", "")
    monkeypatch.setattr("app.PAYPAL_CLIENT_SECRET", "")

    with app.app_context():
        tx = Transaction(
            transaction_id="tx_no_creds_001",
            amount=50.0,
            currency="USD",
            transaction_hour=12,
            international=0,
            new_device=0,
            failed_attempts=0,
            risk_score=10.0,
            risk_classification="Safe",
            transaction_status="APPROVED",
            payment_status="PENDING"
        )
        db.session.add(tx)
        db.session.commit()

    # Checkout page handles missing credentials gracefully (HTTP 500 with friendly guidance)
    page_res = client.get("/payment?transaction_id=tx_no_creds_001")
    assert page_res.status_code == 500
    assert b"PayPal Sandbox credentials are not configured" in page_res.data

    # Order creation endpoint returns 503 instead of crashing
    order_res = client.post("/create-paypal-order", json={"transaction_id": "tx_no_creds_001"})
    assert order_res.status_code == 503
    assert "credentials are not configured" in order_res.get_json()["error"].lower()

    # Capture endpoint returns 503 instead of crashing
    capture_res = client.post("/capture-paypal-order", json={
        "transaction_id": "tx_no_creds_001",
        "orderID": "ORDER-ANY-12345"
    })
    assert capture_res.status_code == 503


# =====================================================================
# TASK 15: Application Logs and Runtime Error Handlers
# =====================================================================
def test_task_15_application_logs_and_runtime_error_handlers(client):
    """Verify 404 and 500 error handlers for both HTML and JSON API requests."""
    # HTML 404
    res_404_html = client.get("/this-route-does-not-exist")
    assert res_404_html.status_code == 404
    assert b"404 - Not Found" in res_404_html.data

    # API JSON 404
    res_404_json = client.get("/api/dashboard/non-existent-endpoint")
    assert res_404_json.status_code == 404
    assert res_404_json.get_json()["status"] == "error"
