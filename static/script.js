const API_URL = "/api";

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.glass-card').forEach(card => card.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(tabId + '-tab').classList.add('active');
}

// Prediction Logic
async function predict() {
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert inputs that look like numbers to numbers? 
    // No, let the backend 'enforce_types' handle that for robustness.
    // Just send everything as is (strings) from form.

    const payload = data;
    const model = document.getElementById('single-model-select').value;

    const resultDiv = document.getElementById('prediction-result');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<p>Analyzing flight data...</p>';
    resultDiv.className = '';

    try {
        const response = await fetch(`${API_URL}/predict-single`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: payload, model: model })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server Error");
        }

        const result = await response.json();

        resultDiv.innerHTML = `Prediction: ${result.label}`;
        resultDiv.className = result.prediction === 1 ? 'result-danger' : 'result-safe';

    } catch (error) {
        resultDiv.innerText = "Error: " + error.message;
        resultDiv.className = 'result-danger';
    }
}

// File Upload Handling
document.getElementById('file-upload').addEventListener('change', function (e) {
    if (e.target.files[0]) {
        document.getElementById('file-name').innerText = e.target.files[0].name;
    }
});

// Evaluation Logic
async function evaluate_models() {
    const fileInput = document.getElementById('file-upload');
    const model = document.getElementById('eval-model-select').value;
    const resultsDiv = document.getElementById('evaluation-results');

    if (!fileInput.files[0]) {
        alert("Please upload a CSV file first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', model);

    resultsDiv.innerHTML = '<p style="text-align:center; margin-top:20px;">Evaluating... This may take a moment.</p>';

    try {
        const response = await fetch(`${API_URL}/evaluate-models?model=${model}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server Error");
        }

        const data = await response.json();

        let html = '';
        data.results.forEach(res => {
            html += `
                <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
                    <h3 style="color: var(--primary); margin-bottom: 1rem;">${res.model}</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${(res.metrics.accuracy * 100).toFixed(1)}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(res.metrics.f1 * 100).toFixed(1)}%</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                         <div class="metric-card">
                            <div class="metric-value">${(res.metrics.recall * 100).toFixed(1)}%</div>
                            <div class="metric-label">Recall</div>
                        </div>
                         <div class="metric-card">
                            <div class="metric-value">${(res.metrics.precision * 100).toFixed(1)}%</div>
                            <div class="metric-label">Precision</div>
                        </div>
                    </div>
                </div>
            `;
        });

        resultsDiv.innerHTML = html;

    } catch (error) {
        resultsDiv.innerHTML = `<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger); color: var(--danger); padding: 1rem; border-radius: 12px; text-align: center;">
            <strong>Error:</strong> ${error.message}
        </div>`;
    }
}
