// Popup script for Phishing URL Guardian
document.addEventListener('DOMContentLoaded', async function() {
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
        if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || 
            tab.url.startsWith('moz-extension://') || tab.url.startsWith('about:')) {
            showError('Cannot analyze browser internal pages');
            return;
        }
        
        // Request result from background script first
        chrome.runtime.sendMessage({ 
            action: 'getResult', 
            tabId: tab.id 
        }, (result) => {
            if (chrome.runtime.lastError) {
                console.error('Runtime error:', chrome.runtime.lastError);
                triggerAnalysis(tab.url, tab.id);
                return;
            }
            
            loadingDiv.style.display = 'none';
            
            if (result && result.url) {
                showResult(result, tab.url);
            } else {
                // No result yet, trigger analysis
                triggerAnalysis(tab.url, tab.id);
            }
        });
        
    } catch (error) {
        console.error('Popup error:', error);
        showError('Extension error: ' + error.message);
    }
    
    // Add event listeners
    document.getElementById('refresh-btn')?.addEventListener('click', function() {
        location.reload();
    });
    
    // document.getElementById('details-btn')?.addEventListener('click', function() {
    //     chrome.tabs.create({ 
    //         url: chrome.runtime.getURL('http://localhost:5000/') 
    //     });
    // });
    
    document.getElementById('option-btn')?.addEventListener('click', function() {
        chrome.tabs.create({ 
            url: chrome.runtime.getURL('options.html') 
        });
    });

    document.getElementById('details-btn')?.addEventListener('click', function() {
        window.open("http://localhost:5000/", "_blank");
    });
    
    document.getElementById('report-btn')?.addEventListener('click', function() {
        // Add reporting functionality here
        alert('Report functionality would be implemented here');
    });
});

function showResult(result, currentUrl) {
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
    const actionButtons = document.getElementById('action-buttons');
    
    // Show result div
    resultDiv.style.display = 'block';
    
    // Display URL info
    if (urlInfo) {
        urlInfo.innerHTML = `<strong>URL:</strong> ${truncateUrl(currentUrl, 50)}`;
        urlInfo.title = currentUrl; // Full URL on hover
    }
    
    // Set status based on prediction
    if (result.prediction_numeric === 1) {
        // Phishing detected
        statusCard.className = 'status-card status-danger';
        statusIcon.textContent = '⚠️';
        statusText.textContent = 'PHISHING DETECTED';
        statusDetail.textContent = 'This website may be dangerous and could steal your information';
        
        // Show confidence if available
        if (result.confidence && confidenceSection) {
            confidenceSection.style.display = 'block';
            confidenceFill.className = 'confidence-fill confidence-danger';
            confidenceFill.style.width = (result.confidence * 100) + '%';
            confidenceText.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;
        }
        
        // Show warning features
        if (result.features) {
            showSuspiciousFeatures(result.features);
        }
        
        // Show action buttons for phishing
        if (actionButtons) {
            actionButtons.innerHTML = `
                <button class="btn btn-danger" onclick="closeTab()">
                    🔒 Block & Close Tab
                </button>
                <button class="btn btn-warning" onclick="continueAnyway()">
                    ⚠️ Continue Anyway
                </button>
            `;
            actionButtons.style.display = 'block';
        }
        
    } else {
        // Safe/Legitimate
        statusCard.className = 'status-card status-safe';
        statusIcon.textContent = '✅';
        statusText.textContent = 'WEBSITE IS SAFE';
        statusDetail.textContent = 'This website appears to be legitimate and secure';
        
        // Show confidence if available
        if (result.confidence && confidenceSection) {
            confidenceSection.style.display = 'block';
            confidenceFill.className = 'confidence-fill confidence-safe';
            confidenceFill.style.width = (result.confidence * 100) + '%';
            confidenceText.textContent = `${(result.confidence * 100).toFixed(1)}% confidence`;
        }
        
        // Show positive security features
        if (result.features) {
            showSecurityFeatures(result.features);
        }
        
        // Hide action buttons for safe sites
        if (actionButtons) {
            actionButtons.style.display = 'none';
        }
    }
    
    // Add model info to status detail
    if (result.model_used && statusDetail) {
        const modelInfo = document.createElement('div');
        modelInfo.className = 'model-info';
        modelInfo.innerHTML = `<small>Analyzed by: ${result.model_used}</small>`;
        statusDetail.appendChild(modelInfo);
    }
    
    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.className = 'timestamp';
    timestamp.innerHTML = `<small>Last checked: ${new Date().toLocaleTimeString()}</small>`;
    resultDiv.appendChild(timestamp);
}

function showSuspiciousFeatures(features) {
    const featuresDiv = document.getElementById('features-summary');
    const featuresList = document.getElementById('features-list');
    
    if (!featuresDiv || !featuresList) return;
    
    // Define important suspicious features with explanations
    const suspiciousFeatures = {
        'NoHttps': {
            label: 'Insecure Connection (HTTP)',
            explanation: 'Uses HTTP instead of secure HTTPS protocol',
            icon: '🔓'
        },
        'IpAddress': {
            label: 'IP Address Used',
            explanation: 'Uses IP address instead of domain name to hide identity',
            icon: '🔢'
        },
        'AtSymbol': {
            label: 'Contains @ Symbol',
            explanation: '@ symbol often used to disguise the real destination',
            icon: '📧'
        },
        'RandomString': {
            label: 'Random Character Sequences',
            explanation: 'Domain contains random strings to avoid detection',
            icon: '🎲'
        },
        'EmbeddedBrandName': {
            label: 'Fake Brand Name',
            explanation: 'Legitimate brand name embedded to deceive users',
            icon: '🏷️'
        },
        'NumSensitiveWords': {
            label: 'Suspicious Keywords',
            explanation: 'Contains words like "secure", "login", "bank" to build false trust',
            icon: '🔍'
        },
        'DomainInSubdomains': {
            label: 'Domain in Subdomain',
            explanation: 'Real domain name appears in subdomain to confuse users',
            icon: '🌐'
        },
        'DomainInPaths': {
            label: 'Domain in URL Path',
            explanation: 'Real domain name appears in URL path for deception',
            icon: '📁'
        },
        'HttpsInHostname': {
            label: '"HTTPS" in Domain Name',
            explanation: 'Word "https" in domain name to appear secure',
            icon: '🔐'
        },
        'ExtFavicon': {
            label: 'External Favicon',
            explanation: 'Website icon loaded from different domain',
            icon: '🖼️'
        },
        'InsecureForms': {
            label: 'Insecure Forms',
            explanation: 'Login forms submit data without encryption',
            icon: '📝'
        },
        'ExtFormAction': {
            label: 'External Form Submission',
            explanation: 'Forms send data to different domains',
            icon: '📤'
        },
        'PctExtHyperlinks': {
            label: 'Many External Links',
            explanation: 'High percentage of links pointing to other websites',
            icon: '🔗'
        },
        'IframeOrFrame': {
            label: 'Hidden Frames',
            explanation: 'Uses hidden frames that could load malicious content',
            icon: '🖼️'
        },
        'MissingTitle': {
            label: 'Missing Page Title',
            explanation: 'Webpage lacks proper title (common in quickly-made phishing sites)',
            icon: '📄'
        }
    };
    
    // Length and structure warnings
    const structuralWarnings = [];
    
    if (features.UrlLength > 100) {
        structuralWarnings.push({
            label: 'Very Long URL',
            explanation: `URL is ${features.UrlLength} characters long (suspicious)`,
            icon: '📏',
            value: features.UrlLength
        });
    }
    
    if (features.SubdomainLevel > 3) {
        structuralWarnings.push({
            label: 'Too Many Subdomains',
            explanation: `${features.SubdomainLevel} subdomain levels (often used to confuse)`,
            icon: '🌐',
            value: features.SubdomainLevel
        });
    }
    
    if (features.NumDots > 5) {
        structuralWarnings.push({
            label: 'Excessive Dots',
            explanation: `${features.NumDots} dots in URL (abnormal structure)`,
            icon: '⚫',
            value: features.NumDots
        });
    }
    
    if (features.NumDash > 5) {
        structuralWarnings.push({
            label: 'Too Many Dashes',
            explanation: `${features.NumDash} dashes in URL (suspicious pattern)`,
            icon: '➖',
            value: features.NumDash
        });
    }
    
    let featuresHtml = '';
    let suspiciousCount = 0;
    
    // Check for suspicious features
    for (const [key, config] of Object.entries(suspiciousFeatures)) {
        if (features.hasOwnProperty(key) && features[key] > 0) {
            suspiciousCount++;
            let displayValue = '';
            
            if (key.startsWith('Pct') && typeof features[key] === 'number') {
                displayValue = ` (${(features[key] * 100).toFixed(1)}%)`;
            } else if (typeof features[key] === 'number' && features[key] > 1) {
                displayValue = ` (${features[key]})`;
            }
            
            featuresHtml += `
                <div class="feature-item suspicious">
                    <div class="feature-header">
                        <span class="feature-icon">${config.icon}</span>
                        <span class="feature-label">${config.label}${displayValue}</span>
                        <span class="feature-status danger">⚠️</span>
                    </div>
                    <div class="feature-explanation">${config.explanation}</div>
                </div>
            `;
        }
    }
    
    // Add structural warnings
    structuralWarnings.forEach(warning => {
        suspiciousCount++;
        featuresHtml += `
            <div class="feature-item suspicious">
                <div class="feature-header">
                    <span class="feature-icon">${warning.icon}</span>
                    <span class="feature-label">${warning.label}</span>
                    <span class="feature-status danger">⚠️</span>
                </div>
                <div class="feature-explanation">${warning.explanation}</div>
            </div>
        `;
    });
    
    if (suspiciousCount > 0) {
        featuresList.innerHTML = `
            <div class="features-header">
                <h3>Suspicious Features Detected (${suspiciousCount})</h3>
                <p>These features indicate this might be a phishing website:</p>
            </div>
            ${featuresHtml}
        `;
        featuresDiv.style.display = 'block';
    }
}

function showSecurityFeatures(features) {
    const featuresDiv = document.getElementById('features-summary');
    const featuresList = document.getElementById('features-list');
    
    if (!featuresDiv || !featuresList) return;
    
    let securityFeatures = '';
    let safeCount = 0;
    
    // Check for positive security indicators
    if (features.NoHttps === 0) {
        safeCount++;
        securityFeatures += `
            <div class="feature-item safe">
                <div class="feature-header">
                    <span class="feature-icon"></span>
                    <span class="feature-label">Secure HTTPS Connection</span>
                    <span class="feature-status safe"></span>
                </div>
                <div class="feature-explanation">Website uses encrypted HTTPS protocol</div>
            </div>
        `;
    }
    
    if (features.IpAddress === 0) {
        safeCount++;
        securityFeatures += `
            <div class="feature-item safe">
                <div class="feature-header">
                    <span class="feature-icon"></span>
                    <span class="feature-label">Proper Domain Name</span>
                    <span class="feature-status safe"></span>
                </div>
                <div class="feature-explanation">Uses legitimate domain name instead of IP address</div>
            </div>
        `;
    }
    
    if (features.UrlLength < 50) {
        safeCount++;
        securityFeatures += `
            <div class="feature-item safe">
                <div class="feature-header">
                    <span class="feature-icon"></span>
                    <span class="feature-label">Normal URL Length</span>
                    <span class="feature-status safe"></span>
                </div>
                <div class="feature-explanation">URL length is normal (${features.UrlLength} characters)</div>
            </div>
        `;
    }
    
    if (features.NumSensitiveWords === 0) {
        safeCount++;
        securityFeatures += `
            <div class="feature-item safe">
                <div class="feature-header">
                    <span class="feature-icon"></span>
                    <span class="feature-label">No Suspicious Keywords</span>
                    <span class="feature-status safe"></span>
                </div>
                <div class="feature-explanation">Doesn't contain common phishing keywords</div>
            </div>
        `;
    }
    
    if (safeCount > 0) {
        featuresList.innerHTML = `
            <div class="features-header">
                <h3>Security Features (${safeCount})</h3>
                <p>These features indicate the website is likely legitimate:</p>
            </div>
            ${securityFeatures}
        `;
        featuresDiv.style.display = 'block';
    }
}

function showError(message) {
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const errorMessage = document.getElementById('error-message');
    
    if (loadingDiv) loadingDiv.style.display = 'none';
    if (errorDiv) errorDiv.style.display = 'block';
    if (errorMessage) errorMessage.textContent = message;
}

function triggerAnalysis(url, tabId) {
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const loadingText = document.getElementById('loading-text');
    
    // Show loading state
    if (resultDiv) resultDiv.style.display = 'none';
    if (loadingDiv) loadingDiv.style.display = 'block';
    if (loadingText) loadingText.textContent = 'Analyzing URL for phishing threats...';
    
    // Request analysis from background script
    chrome.runtime.sendMessage({
        action: 'analyzeUrl',
        url: url,
        tabId: tabId
    }, (response) => {
        if (chrome.runtime.lastError) {
            console.error('Analysis request error:', chrome.runtime.lastError);
            showError('Failed to analyze URL. Please try again.');
            return;
        }
        
        // Wait a bit for analysis to complete
        setTimeout(() => {
            chrome.runtime.sendMessage({ 
                action: 'getResult', 
                tabId: tabId 
            }, (result) => {
                if (loadingDiv) loadingDiv.style.display = 'none';
                
                if (result && result.url) {
                    showResult(result, url);
                } else {
                    showError('Analysis taking longer than expected. Please refresh or try again.');
                }
            });
        }, 3000);
    });
}

// Utility functions
function truncateUrl(url, maxLength) {
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength - 3) + '...';
}

// Global functions for button actions
window.closeTab = function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (tabs[0]) {
            chrome.tabs.remove(tabs[0].id);
        }
    });
};

window.continueAnyway = function() {
    // Close popup - user chooses to continue
    window.close();
};

window.reportPhishing = function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        if (tabs[0]) {
            // Open reporting interface or send report
            chrome.tabs.create({
                url: `https://safebrowsing.google.com/safebrowsing/report_phish/?url=${encodeURIComponent(tabs[0].url)}`
            });
        }
    });
};