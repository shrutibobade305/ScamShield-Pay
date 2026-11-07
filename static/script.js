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
                button.textContent = "Analyzing Transaction...";
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
