// Popup script for Phishing URL Guardian
document.addEventListener('DOMContentLoaded', async function () {
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');

    try {
        // Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab || !tab.url) {
            showError('Unable to get current tab information');
            return;
        }

        // Skip chrome:// and extension URLs
        if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
            showError('Cannot analyze Chrome internal pages');
            return;
        }

        // Request result from background script
        chrome.runtime.sendMessage({
            action: 'getResult',
            tabId: tab.id
        }, (result) => {
            loadingDiv.style.display = 'none';

            if (result) {
                showResult(result, tab.url);
            } else {
                // No result yet, try to trigger analysis
                triggerAnalysis(tab.url, tab.id);
            }
        });

    } catch (error) {
        console.error('Popup error:', error);
        showError('Extension error: ' + error.message);
    }

    // Add event listeners
    document.getElementById('refresh-btn').addEventListener('click', function () {
        location.reload();
    });

    document.getElementById('details-btn').addEventListener('click', function () {
        chrome.tabs.create({
            url: chrome.runtime.getURL('index.html')
        });
    });
});

function showResult(result, url) {
    const resultDiv = document.getElementById('result');
    const statusCard = document.getElementById('status-card');
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const statusDetail = document.getElementById('status-detail');
    const urlInfo = document.getElementById('url-info');
    const featuresDiv = document.getElementById('features-summary');
    const confidenceSection = document.getElementById('confidence-section');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');

    // Show result div
    resultDiv.style.display = 'block';

    // Set status based on prediction
    if (result.prediction_numeric === 1) {
        // Phishing detected
        statusCard.className = 'status-card status-danger';
        statusIcon.textContent = '⚠️';
        statusText.textContent = 'PHISHING DETECTED';
        statusDetail.textContent = 'This website may be dangerous';

        if (result.confidence) {
            confidenceSection.style.display = 'block';
            confidenceFill.className = 'confidence-fill confidence-danger';
            confidenceFill.style.width = (result.confidence * 100) + '%';
            confidenceText.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;
        }
    } else {
        // Safe
        statusCard.className = 'status-card status-safe';
        statusIcon.textContent = '✅';
        statusText.textContent = 'SAFE';
        statusDetail.textContent = 'This website appears to be legitimate';

        if (result.confidence) {
            confidenceSection.style.display = 'block';
            confidenceFill.className = 'confidence-fill confidence-safe';
            confidenceFill.style.width = (result.confidence * 100) + '%';
            confidenceText.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;
        }
    }

    // Show URL info
    urlInfo.textContent = `URL: ${url}`;

    // Show key features if phishing detected
    if (result.prediction_numeric === 1 && result.features) {
        showKeyFeatures(result.features);
    }

    // Add model info to status detail
    if (result.model_used) {
        statusDetail.textContent += ` (Model: ${result.model_used})`;
    }
}

function showKeyFeatures(features) {
    const featuresDiv = document.getElementById('features-summary');
    const featuresList = document.getElementById('features-list');

    const importantFeatures = {
        'NoHttps': 'Insecure Connection (HTTP)',
        'IpAddress': 'IP Address Used',
        'UrlLength': 'URL Length',
        'NumSensitiveWords': 'Suspicious Words',
        'EmbeddedBrandName': 'Fake Brand Name',
        'RandomString': 'Random Strings',
        'AtSymbol': 'Contains @ Symbol',
        'NumDots': 'Number of Dots',
        'SubdomainLevel': 'Subdomain Levels',
        'PathLevel': 'Path Depth',
        'NumDash': 'Dashes in URL',
        'ExtFavicon': 'External Favicon',
        'InsecureForms': 'Insecure Forms',
        'PctExtHyperlinks': 'External Links %'
    };

    let featuresHtml = '';
    let suspiciousCount = 0;

    for (const [key, label] of Object.entries(importantFeatures)) {
        if (features.hasOwnProperty(key)) {
            const value = features[key];
            if (value > 0) {
                suspiciousCount++;
                let displayValue = value;

                if (key.startsWith('Pct')) {
                    displayValue = (value * 100).toFixed(1) + '%';
                } else if (typeof value === 'number' && value !== 1) {
                    displayValue = value;
                } else if (value === 1) {
                    displayValue = '✓';
                }

                featuresHtml += `
                    <div class="feature-item">
                        <span>${label}</span>
                        <span class="feature-value">${displayValue}</span>
                    </div>
                `;
            }
        }
    }

    if (suspiciousCount > 0) {
        featuresList.innerHTML = featuresHtml;
        featuresDiv.style.display = 'block';
    }
}

function showError(message) {
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const errorMessage = document.getElementById('error-message');

    loadingDiv.style.display = 'none';
    errorDiv.style.display = 'block';
    errorMessage.textContent = message;
}

function triggerAnalysis(url, tabId) {
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    resultDiv.style.display = 'none';
    loadingDiv.style.display = 'block';

    setTimeout(() => {
        chrome.runtime.sendMessage({
            action: 'getResult',
            tabId: tabId
        }, (result) => {
            loadingDiv.style.display = 'none';

            if (result) {
                showResult(result, url);
            } else {
                showError('Analysis in progress. Please wait...');
            }
        });
    }, 2000);
}
