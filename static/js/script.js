// Get the form and result elements.
const predictionForm = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-button");
const messageElement = document.getElementById("message");
const resultCard = document.getElementById("result-card");
const predictedPriceElement = document.getElementById("predicted-price");
const modelNameElement = document.getElementById("model-name");

// Submit the form and request a prediction from Flask.
predictionForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    messageElement.textContent = "";
    messageElement.className = "message";
    resultCard.hidden = true;

    // Collect all form values.
    const formData = new FormData(predictionForm);
    const carDetails = {};

    for (const [key, value] of formData.entries()) {
        const input = predictionForm.elements.namedItem(key);

        if (input.dataset.type === "number") {
            carDetails[key] = Number(value);
        } else {
            carDetails[key] = value;
        }
    }

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(carDetails)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed.");
        }

        // Display the predicted price returned by Flask.
        predictedPriceElement.textContent =
            new Intl.NumberFormat("en-US", {
                style: "currency",
                currency: result.currency || "USD",
                maximumFractionDigits: 2
            }).format(result.predicted_price);

        modelNameElement.textContent =
            "Model used: " + result.model;

        resultCard.hidden = false;
        messageElement.textContent = "Prediction completed successfully.";
        messageElement.className = "message";

    } catch (error) {
        messageElement.textContent = error.message;
        messageElement.className = "message error";

    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Price";
    }
});

// Clear old prediction results when the form is reset.
predictionForm.addEventListener("reset", function () {
    resultCard.hidden = true;
    messageElement.textContent = "";
    messageElement.className = "message";
});