document.addEventListener("DOMContentLoaded", function () {

    // Transaction amount validation
    const amountInput = document.querySelector('input[name="amount"]');

    if (amountInput) {
        amountInput.addEventListener("input", function () {
            const amount = Number(this.value);

            if (amount <= 0 || amount > 10000000) {
                this.setCustomValidity(
                    "Enter an amount between $0.01 and $10,000,000."
                );
            } else {
                this.setCustomValidity("");
            }
        });
    }

    // Prevent duplicate form submission
    const analyzerForm = document.querySelector(".analyzer-form");

    if (analyzerForm) {
        analyzerForm.addEventListener("submit", function (event) {
            if (!this.checkValidity()) {
                event.preventDefault();
                this.reportValidity();
                return;
            }

            const button = this.querySelector("button[type='submit']");
            if (button) {
                button.disabled = true;
                button.textContent = "Analyzing Transaction Risk...";
            }
        });
    }

    // Restore button state when navigating back via browser history (bfcache)
    window.addEventListener("pageshow", function (event) {
        if (analyzerForm) {
            const button = analyzerForm.querySelector("button[type='submit']");
            if (button) {
                button.disabled = false;
                button.textContent = "Analyze Transaction →";
            }
        }
    });

    // Smooth scroll for anchor navigation links
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

// Global Scenario Pre-fill for Testing & Demonstration
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
        amount.value = "15000.00";
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

    amount.dispatchEvent(new Event("input", { bubbles: true }));
};
