// Options page script for Phishing Guardian
document.addEventListener('DOMContentLoaded', function() {
    loadSettings();
    loadStatistics();
    
    // Add event listeners
    document.getElementById('save-btn').addEventListener('click', saveSettings);
    document.getElementById('reset-btn').addEventListener('click', resetSettings);
    document.getElementById('test-btn').addEventListener('click', testApiConnection);
    document.getElementById('clear-stats-btn').addEventListener('click', clearStatistics);
});

// Default settings
const defaultSettings = {
    apiUrl: 'http://localhost:5000',
    apiTimeout: 10,
    showNotifications: true,
    showWarningOverlay: true,
    autoBlockPhishing: false,
    warningSensitivity: 'medium',
    monitorAllUrls: true,
    cacheResults: true,
    cacheDuration: 5,
    preferredModel: 'auto',
    whitelistDomains: []
};

// Load settings from storage
function loadSettings() {
    chrome.storage.sync.get(defaultSettings, function(settings) {
        document.getElementById('api-url').value = settings.apiUrl;
        document.getElementById('api-timeout').value = settings.apiTimeout;
        document.getElementById('show-notifications').checked = settings.showNotifications;
        document.getElementById('show-warning-overlay').checked = settings.showWarningOverlay;
        document.getElementById('auto-block-phishing').checked = settings.autoBlockPhishing;
        document.getElementById('warning-sensitivity').value = settings.warningSensitivity;
        document.getElementById('monitor-all-urls').checked = settings.monitorAllUrls;
        document.getElementById('cache-results').checked = settings.cacheResults;
        document.getElementById('cache-duration').value = settings.cacheDuration;
        document.getElementById('preferred-model').value = settings.preferredModel;
        document.getElementById('whitelist-domains').value = settings.whitelistDomains.join('\n');
    });
}

// Save settings to storage
function saveSettings() {
    const settings = {
        apiUrl: document.getElementById('api-url').value.trim(),
        apiTimeout: parseInt(document.getElementById('api-timeout').value),
        showNotifications: document.getElementById('show-notifications').checked,
        showWarningOverlay: document.getElementById('show-warning-overlay').checked,
        autoBlockPhishing: document.getElementById('auto-block-phishing').checked,
        warningSensitivity: document.getElementById('warning-sensitivity').value,
        monitorAllUrls: document.getElementById('monitor-all-urls').checked,
        cacheResults: document.getElementById('cache-results').checked,
        cacheDuration: parseInt(document.getElementById('cache-duration').value),
        preferredModel: document.getElementById('preferred-model').value,
        whitelistDomains: document.getElementById('whitelist-domains').value
            .split('\n')
            .map(domain => domain.trim())
            .filter(domain => domain.length > 0)
    };
    
    // Validate settings
    if (!settings.apiUrl || !isValidUrl(settings.apiUrl)) {
        showStatus('Please enter a valid API URL', 'error');
        return;
    }
    
    if (settings.apiTimeout < 1 || settings.apiTimeout > 30) {
        showStatus('API timeout must be between 1 and 30 seconds', 'error');
        return;
    }
    
    if (settings.cacheDuration < 1 || settings.cacheDuration > 60) {
        showStatus('Cache duration must be between 1 and 60 minutes', 'error');
        return;
    }
    
    // Save settings
    chrome.storage.sync.set(settings, function() {
        if (chrome.runtime.lastError) {
            showStatus('Error saving settings: ' + chrome.runtime.lastError.message, 'error');
        } else {
            showStatus('Settings saved successfully!', 'success');
            
            // Notify background script of settings change
            chrome.runtime.sendMessage({
                action: 'settingsUpdated',
                settings: settings
            });
        }
    });
}

// Reset settings to defaults
function resetSettings() {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
        chrome.storage.sync.set(defaultSettings, function() {
            loadSettings();
            showStatus('Settings reset to defaults', 'success');
        });
    }
}

// Test API connection
async function testApiConnection() {
    const apiUrl = document.getElementById('api-url').value.trim();
    const testUrl = document.getElementById('test-url').value.trim();
    const testBtn = document.getElementById('test-btn');
    const testResult = document.getElementById('test-result');
    
    if (!apiUrl || !testUrl) {
        showTestResult('Please enter both API URL and test URL', 'error');
        return;
    }
    
    testBtn.disabled = true;
    testBtn.textContent = 'Testing...';
    testResult.style.display = 'none';
    
    try {
        const response = await fetch(`${apiUrl}/api/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: testUrl }),
            timeout: 10000
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            showTestResult(`API Error: ${result.error}`, 'error');
        } else {
            showTestResult(
                `✅ Connection successful!\n` +
                `Prediction: ${result.prediction}\n` +
                `Confidence: ${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A'}\n` +
                `Model: ${result.model_used}`,
                'success'
            );
        }
        
    } catch (error) {
        showTestResult(`Connection failed: ${error.message}`, 'error');
    } finally {
        testBtn.disabled = false;
        testBtn.textContent = 'Test Connection';
    }
}

// Load and display statistics
function loadStatistics() {
    chrome.storage.local.get(['totalChecks', 'phishingDetected', 'safeSites'], function(stats) {
        document.getElementById('total-checks').textContent = stats.totalChecks || 0;
        document.getElementById('phishing-detected').textContent = stats.phishingDetected || 0;
        document.getElementById('safe-sites').textContent = stats.safeSites || 0;
    });
}

// Clear statistics
function clearStatistics() {
    if (confirm('Are you sure you want to clear all statistics?')) {
        chrome.storage.local.remove(['totalChecks', 'phishingDetected', 'safeSites'], function() {
            loadStatistics();
            showStatus('Statistics cleared', 'success');
        });
    }
}

// Show status message
function showStatus(message, type) {
    const statusDiv = document.getElementById('status-message');
    statusDiv.textContent = message;
    statusDiv.className = `status-message status-${type}`;
    statusDiv.style.display = 'block';
    
    setTimeout(() => {
        statusDiv.style.display = 'none';
    }, 3000);
}

// Show test result
function showTestResult(message, type) {
    const testResult = document.getElementById('test-result');
    testResult.textContent = message;
    testResult.className = `test-result test-${type}`;
    testResult.style.display = 'block';
}

// Validate URL
function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}