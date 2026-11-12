"""
ScamShield Pay - Database Models & Persistence Layer
====================================================
Defines the SQLite / SQLAlchemy transaction schema for persisting evaluated
payment requests, risk scoring verdicts, and PayPal payment lifecycle events.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Transaction(db.Model):
    """
    Represents an analyzed payment transaction record.

    Stores behavioral input features, ML risk score, risk classification,
    and associated PayPal Sandbox order and capture lifecycle states.
    """

    __tablename__ = "transactions"

    # Database primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Unique URL-safe transaction tracking identifier
    transaction_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

    # Core transaction features
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(8), nullable=False, default="USD")
    transaction_hour = db.Column(db.Integer, nullable=False)
    international = db.Column(db.Integer, nullable=False, default=0)
    new_device = db.Column(db.Integer, nullable=False, default=0)
    failed_attempts = db.Column(db.Integer, nullable=False, default=0)

    # Risk evaluation results
    risk_score = db.Column(db.Float, nullable=False)
    risk_classification = db.Column(db.String(32), nullable=False)  # 'Safe', 'Suspicious', 'High Risk'

    # Transaction decision status ('APPROVED', 'UNDER_REVIEW', 'APPROVED_POST_MFA', 'BLOCKED', 'COMPLETED')
    transaction_status = db.Column(db.String(32), nullable=False)

    # Secondary Step-Up Verification (MFA)
    verification_status = db.Column(db.String(32), default="NONE", nullable=False)  # 'NONE', 'PENDING', 'VERIFIED', 'FAILED', 'EXPIRED'
    otp_code_hash = db.Column(db.String(128), nullable=True)  # SHA-256 hash of OTP code (never plain text)
    otp_expires_at = db.Column(db.DateTime, nullable=True)
    otp_attempts_left = db.Column(db.Integer, default=3, nullable=False)
    otp_requested_at = db.Column(db.DateTime, nullable=True)
    verification_notes = db.Column(db.String(255), nullable=True)

    # PayPal checkout tracking
    paypal_order_id = db.Column(db.String(64), nullable=True, index=True)
    paypal_capture_id = db.Column(db.String(64), nullable=True)

    # Payment status ('NOT_ATTEMPTED', 'PENDING', 'COMPLETED', 'FAILED', 'CANCELLED', 'BLOCKED')
    payment_status = db.Column(db.String(32), nullable=False, default="NOT_ATTEMPTED")

    # Timestamp (UTC)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self):
        """Serializes transaction model into a dictionary."""
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "amount": self.amount,
            "currency": self.currency,
            "transaction_hour": self.transaction_hour,
            "international": self.international,
            "new_device": self.new_device,
            "failed_attempts": self.failed_attempts,
            "risk_score": self.risk_score,
            "risk_classification": self.risk_classification,
            "transaction_status": self.transaction_status,
            "verification_status": self.verification_status,
            "otp_attempts_left": self.otp_attempts_left,
            "verification_notes": self.verification_notes,
            "paypal_order_id": self.paypal_order_id,
            "paypal_capture_id": self.paypal_capture_id,
            "payment_status": self.payment_status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return (
            f"<Transaction id={self.id} tx_id={self.transaction_id} "
            f"amount={self.amount} risk={self.risk_score}% "
            f"class={self.risk_classification} status={self.payment_status}>"
        )

