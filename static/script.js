/* ==========================================================================
   ScamShield Pay - Master Client-side Script
   Handles validation, scenario pre-fills, scanning animation, and UX feedback
   ========================================================================== */

document.addEventListener("DOMContentLoaded", function () {

    // 1. Transaction amount validation
    const amountInput = document.getElementById("amount") || document.querySelector('input[name="amount"]');

    if (amountInput) {
        amountInput.addEventListener("input", function () {
            const amount = Number(this.value);

            if (isNaN(amount) || amount <= 0 || amount > 10000000) {
                this.setCustomValidity("Please enter a valid amount between $0.01 and $10,000,000.");
            } else {
                this.setCustomValidity("");
            }
        });
    }

    // 2. Form submission with professional scanning animation
    const analyzerForm = document.getElementById("risk-analyzer-form") || document.querySelector(".analyzer-form");
    const scanModal = document.getElementById("scanning-modal");
    const scanStatusText = document.getElementById("scan-status-text");
    const scanStepHint = document.getElementById("scan-step-hint");

    if (analyzerForm) {
        analyzerForm.addEventListener("submit", function (event) {
            if (!this.checkValidity()) {
                event.preventDefault();
                this.reportValidity();
                return;
            }

            const button = document.getElementById("btn-submit-analyzer") || this.querySelector("button[type='submit']");
            if (button) {
                button.disabled = true;
                button.innerHTML = "<span>⏳</span> Evaluating Risk Signals...";
            }

            // Display scanning radar animation
            if (scanModal && scanStatusText && scanStepHint) {
                scanModal.style.display = "flex";
                
                // Phase 1
                scanStatusText.textContent = "Extracting Behavioral Fingerprints...";
                scanStepHint.textContent = "Analyzing hardware profile, temporal velocity, and transfer value.";

                // Phase 2
                setTimeout(() => {
                    if (scanStatusText) scanStatusText.textContent = "Running Random Forest ML Inference...";
                    if (scanStepHint) scanStepHint.textContent = "Evaluating non-linear feature split across 200 decision trees.";
                }, 400);

                // Phase 3
                setTimeout(() => {
                    if (scanStatusText) scanStatusText.textContent = "Applying Operational Security Policies...";
                    if (scanStepHint) scanStepHint.textContent = "Computing rule-based risk indicators and clearance status.";
                }, 850);
            }
        });
    }

    // 3. Restore button state when navigating back via browser history (bfcache)
    window.addEventListener("pageshow", function () {
        if (analyzerForm) {
            const button = document.getElementById("btn-submit-analyzer") || analyzerForm.querySelector("button[type='submit']");
            if (button) {
                button.disabled = false;
                button.innerHTML = "<span>🔍</span> Analyze Transaction Risk →";
            }
        }
        if (scanModal) {
            scanModal.style.display = "none";
        }
    });

    // 4. Smooth scroll for anchor navigation links
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener("click", function (event) {
            const targetId = this.getAttribute("href");
            if (targetId && targetId !== "#") {
                const target = document.querySelector(targetId);
                if (target) {
                    event.preventDefault();
                    target.scrollIntoView({
                        behavior: "smooth"
                    });
                }
            }
        });
    });

});

// 5. Global Scenario Pre-fill with Visual Pulse
window.fillScenario = function (type) {
    const amount = document.getElementById("amount");
    const hour = document.getElementById("transaction_hour");
    const intl = document.getElementById("international");
    const device = document.getElementById("new_device");
    const fails = document.getElementById("failed_attempts");

    if (!amount || !hour || !intl || !device || !fails) {
        return;
    }

    if (type === "safe") {
        amount.value = "45.00";
        hour.value = "14";
        intl.value = "0";
        device.value = "0";
        fails.value = "0";
    } else if (type === "suspicious") {
        amount.value = "12000.00";
        hour.value = "3";
        intl.value = "1";
        device.value = "1";
        fails.value = "2";
    } else if (type === "danger") {
        amount.value = "35000.00";
        hour.value = "2";
        intl.value = "1";
        device.value = "1";
        fails.value = "5";
    }

    // Dispatch input events
    [amount, hour, intl, device, fails].forEach(el => {
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));

        // Subtle visual highlight pulse
        el.style.transition = "border-color 0.3s, background 0.3s";
        el.style.borderColor = "#6366f1";
        el.style.background = "#131c30";
        setTimeout(() => {
            el.style.borderColor = "";
            el.style.background = "";
        }, 500);
    });

    // Smooth scroll down to analyzer if triggered from elsewhere
    const analyzer = document.getElementById("analyzer");
    if (analyzer && window.scrollY < analyzer.offsetTop - 150) {
        analyzer.scrollIntoView({ behavior: "smooth" });
    }
};
