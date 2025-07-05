// Background script for Phishing URL Guardian
const API_BASE_URL = 'http://localhost:5000'; // Change this to your Flask server URL
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes cache

// Cache for URL analysis results
let urlCache = new Map();

// Listen for tab updates (URL changes)
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && tab.url) {
    // Skip chrome:// and extension URLs
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
      return;
    }
    
    console.log('Checking URL:', tab.url);
    await checkURL(tab.url, tabId);
  }
});

// Listen for navigation events
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId === 0) { // Main frame only
    console.log('Navigation to:', details.url);
    await checkURL(details.url, details.tabId);
  }
});

// Function to check URL against Flask API
async function checkURL(url, tabId) {
  try {
    // Check cache first
    const cacheKey = url;
    const cachedResult = urlCache.get(cacheKey);
    
    if (cachedResult && (Date.now() - cachedResult.timestamp) < CACHE_DURATION) {
      console.log('Using cached result for:', url);
      await handleResult(cachedResult.result, tabId);
      return;
    }
    
    // Call Flask API
    const response = await fetch(`${API_BASE_URL}/api/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url }),
    });
    
    if (!response.ok) {
      console.error('API request failed:', response.status);
      return;
    }
    
    const result = await response.json();
    
    // Cache the result
    urlCache.set(cacheKey, {
      result: result,
      timestamp: Date.now()
    });
    
    // Clean old cache entries
    if (urlCache.size > 100) {
      const oldestKey = urlCache.keys().next().value;
      urlCache.delete(oldestKey);
    }
    
    await handleResult(result, tabId);
    
  } catch (error) {
    console.error('Error checking URL:', error);
  }
}

// Handle the analysis result
async function handleResult(result, tabId) {
  if (result.error) {
    console.error('API Error:', result.error);
    return;
  }
  
  // Update extension icon based on result
  const iconPath = result.prediction_numeric === 1 ? 'icons/icon_danger.png' : 'icons/icon_safe.png';
  
  try {
    await chrome.action.setIcon({
      tabId: tabId,
      path: {
        16: iconPath,
        48: iconPath,
        128: iconPath
      }
    });
  } catch (error) {
    console.error('Error setting icon:', error);
  }
  
  // Store result for popup
  await chrome.storage.local.set({
    [`result_${tabId}`]: result
  });
  
  // If phishing detected, show warning
  if (result.prediction_numeric === 1) {
    await showPhishingWarning(tabId, result);
  }
}

// Show phishing warning
async function showPhishingWarning(tabId, result) {
  try {
    // Create notification
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon48.png',
      title: '⚠️ Phishing Warning',
      message: `Potential phishing site detected: ${result.url}`,
      priority: 2
    });
    
    // Inject warning overlay
    await chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: injectWarningOverlay,
      args: [result]
    });
    
  } catch (error) {
    console.error('Error showing warning:', error);
  }
}

// Function to inject warning overlay (will be executed in content script context)
function injectWarningOverlay(result) {
  // Remove existing warning if any
  const existingWarning = document.getElementById('phishing-warning-overlay');
  if (existingWarning) {
    existingWarning.remove();
  }
  
  // Create warning overlay
  const overlay = document.createElement('div');
  overlay.id = 'phishing-warning-overlay';
  overlay.innerHTML = `
    <div class="phishing-warning-modal">
      <div class="warning-header">
        <h2>⚠️ Phishing Warning</h2>
        <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">&times;</button>
      </div>
      <div class="warning-content">
        <p><strong>This website may be a phishing site!</strong></p>
        <p><strong>URL:</strong> ${result.url}</p>
        <p><strong>Confidence:</strong> ${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : 'N/A'}</p>
        <p><strong>Model:</strong> ${result.model_used}</p>
        
        <div class="features-section">
          <h3>Suspicious Features Detected:</h3>
          <ul class="features-list">
            ${generateFeaturesList(result.features)}
          </ul>
        </div>
        
        <div class="warning-actions">
          <button class="btn-danger" onclick="history.back()">← Go Back</button>
          <button class="btn-warning" onclick="this.parentElement.parentElement.parentElement.parentElement.remove()">Continue Anyway</button>
        </div>
      </div>
    </div>
  `;
  
  // Add CSS styles
  const style = document.createElement('style');
  style.textContent = `
    #phishing-warning-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      z-index: 999999;
      display: flex;
      justify-content: center;
      align-items: center;
      font-family: Arial, sans-serif;
    }
    
    .phishing-warning-modal {
      background: white;
      border-radius: 10px;
      max-width: 600px;
      width: 90%;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .warning-header {
      background: #dc3545;
      color: white;
      padding: 20px;
      border-radius: 10px 10px 0 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .warning-header h2 {
      margin: 0;
      font-size: 24px;
    }
    
    .close-btn {
      background: none;
      border: none;
      color: white;
      font-size: 30px;
      cursor: pointer;
      padding: 0;
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .close-btn:hover {
      background: rgba(255, 255, 255, 0.2);
      border-radius: 50%;
    }
    
    .warning-content {
      padding: 20px;
    }
    
    .warning-content p {
      margin: 10px 0;
      font-size: 16px;
    }
    
    .features-section {
      margin: 20px 0;
      padding: 15px;
      background: #f8f9fa;
      border-radius: 5px;
    }
    
    .features-section h3 {
      margin: 0 0 10px 0;
      color: #dc3545;
    }
    
    .features-list {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    
    .features-list li {
      padding: 5px 0;
      border-bottom: 1px solid #dee2e6;
    }
    
    .features-list li:last-child {
      border-bottom: none;
    }
    
    .warning-actions {
      margin-top: 20px;
      display: flex;
      gap: 10px;
      justify-content: center;
    }
    
    .btn-danger {
      background: #dc3545;
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 16px;
      font-weight: bold;
    }
    
    .btn-danger:hover {
      background: #c82333;
    }
    
    .btn-warning {
      background: #ffc107;
      color: #000;
      border: none;
      padding: 12px 24px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 16px;
      font-weight: bold;
    }
    
    .btn-warning:hover {
      background: #e0a800;
    }
  `;
  
  document.head.appendChild(style);
  document.body.appendChild(overlay);
}

// Generate features list for display
function generateFeaturesList(features) {
  const suspiciousFeatures = [];
  
  // Define feature descriptions
  const featureDescriptions = {
    'NoHttps': 'Website uses HTTP instead of HTTPS (insecure)',
    'IpAddress': 'URL uses IP address instead of domain name',
    'UrlLength': 'URL is unusually long',
    'NumDash': 'Excessive dashes in URL',
    'AtSymbol': 'Contains @ symbol (often used to hide real domain)',
    'TildeSymbol': 'Contains ~ symbol (suspicious pattern)',
    'NumSensitiveWords': 'Contains suspicious words (login, secure, bank, etc.)',
    'EmbeddedBrandName': 'Fake brand name embedded in domain',
    'RandomString': 'Domain contains random character sequences',
    'DomainInSubdomains': 'Domain name appears in subdomain (deceptive)',
    'DomainInPaths': 'Domain name appears in URL path (deceptive)',
    'HttpsInHostname': 'Word "https" appears in hostname (deceptive)',
    'ExtFavicon': 'Favicon loaded from external domain',
    'InsecureForms': 'Forms submit data insecurely',
    'ExtFormAction': 'Forms submit to external domains',
    'PctExtHyperlinks': 'High percentage of external links',
    'PctExtResourceUrls': 'Many resources loaded from external domains',
    'MissingTitle': 'Webpage has missing or suspicious title',
    'IframeOrFrame': 'Uses hidden frames (potential clickjacking)'
  };
  
  // Check for suspicious features
  for (const [feature, value] of Object.entries(features)) {
    if (featureDescriptions[feature] && value > 0) {
      let description = featureDescriptions[feature];
      if (typeof value === 'number' && value > 1) {
        description += ` (${value})`;
      }
      suspiciousFeatures.push(`<li>${description}</li>`);
    }
  }
  
  // Add some general suspicious patterns
  if (features.UrlLength > 100) {
    suspiciousFeatures.push(`<li>Very long URL (${features.UrlLength} characters)</li>`);
  }
  
  if (features.SubdomainLevel > 3) {
    suspiciousFeatures.push(`<li>Too many subdomain levels (${features.SubdomainLevel})</li>`);
  }
  
  if (features.NumDots > 5) {
    suspiciousFeatures.push(`<li>Too many dots in URL (${features.NumDots})</li>`);
  }
  
  return suspiciousFeatures.length > 0 ? suspiciousFeatures.join('') : '<li>General suspicious patterns detected</li>';
}

// Message listener for popup communication
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getResult') {
    chrome.storage.local.get(`result_${request.tabId}`, (data) => {
      sendResponse(data[`result_${request.tabId}`] || null);
    });
    return true; // Keep message channel open for async response
  }
});

// Clean up storage when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.local.remove(`result_${tabId}`);
});