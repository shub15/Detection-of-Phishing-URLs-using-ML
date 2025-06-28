// Content script for additional URL monitoring
(function() {
  'use strict';
  
  // Monitor hash changes and dynamic navigation
  let lastUrl = location.href;
  
  function checkUrlChange() {
    const currentUrl = location.href;
    if (currentUrl !== lastUrl) {
      lastUrl = currentUrl;
      // Notify background script of URL change
      chrome.runtime.sendMessage({
        action: 'urlChanged',
        url: currentUrl,
        tabId: chrome.runtime.id
      }).catch(() => {
        // Ignore errors if background script is not ready
      });
    }
  }
  
  // Monitor for URL changes (for SPAs)
  const observer = new MutationObserver(checkUrlChange);
  observer.observe(document, { subtree: true, childList: true });
  
  // Listen for popstate events (back/forward navigation)
  window.addEventListener('popstate', checkUrlChange);
  
  // Listen for hash changes
  window.addEventListener('hashchange', checkUrlChange);
  
  // Override history methods to catch programmatic navigation
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;
  
  history.pushState = function() {
    originalPushState.apply(history, arguments);
    setTimeout(checkUrlChange, 0);
  };
  
  history.replaceState = function() {
    originalReplaceState.apply(history, arguments);
    setTimeout(checkUrlChange, 0);
  };
  
  // Monitor form submissions to potentially dangerous domains
  document.addEventListener('submit', function(event) {
    const form = event.target;
    if (form.tagName === 'FORM') {
      const action = form.action || window.location.href;
      
      // Check if form is submitting to external domain
      try {
        const formUrl = new URL(action, window.location.href);
        const currentUrl = new URL(window.location.href);
        
        if (formUrl.hostname !== currentUrl.hostname) {
          console.log('Form submitting to external domain:', action);
          // Could add additional warnings here
        }
      } catch (error) {
        console.error('Error analyzing form submission:', error);
      }
    }
  });
  
  // Monitor for suspicious JavaScript patterns
  const originalWindowOpen = window.open;
  window.open = function(url, name, features) {
    console.log('window.open called with:', url);
    return originalWindowOpen.call(this, url, name, features);
  };
  
  // Monitor for suspicious redirects
  let redirectCount = 0;
  const originalLocationReplace = location.replace;
  const originalLocationAssign = location.assign;
  
  location.replace = function(url) {
    redirectCount++;
    if (redirectCount > 3) {
      console.warn('Multiple redirects detected - potential phishing behavior');
    }
    return originalLocationReplace.call(this, url);
  };
  
  location.assign = function(url) {
    redirectCount++;
    if (redirectCount > 3) {
      console.warn('Multiple redirects detected - potential phishing behavior');
    }
    return originalLocationAssign.call(this, url);
  };
  
})();