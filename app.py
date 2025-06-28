# Flask Web Application for Phishing URL Detection
from flask import Flask, render_template, request, jsonify
import os
import sys
import joblib
import numpy as np
from urllib.parse import urlparse
import tldextract
import re
import requests
from bs4 import BeautifulSoup
import socket

# Import the PhishingURLDetector class (assuming it's in the same directory)
# For this demo, we'll include a simplified version here

class PhishingURLDetectorFlask:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.best_model_name = None
        
    def extract_features(self, url):
        """Extract features from URL for phishing detection"""
        features = {}
        
        try:
            parsed_url = urlparse(url)
            extracted = tldextract.extract(url)
            
            # Basic URL features
            features['url_length'] = len(url)
            features['domain_length'] = len(parsed_url.netloc)
            features['path_length'] = len(parsed_url.path)
            features['query_length'] = len(parsed_url.query)
            
            # Domain-based features
            features['subdomain_count'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
            features['has_ip'] = 1 if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', parsed_url.netloc) else 0
            
            # Suspicious characters
            features['dash_count'] = url.count('-')
            features['dot_count'] = url.count('.')
            features['underscore_count'] = url.count('_')
            features['question_count'] = url.count('?')
            features['equal_count'] = url.count('=')
            features['and_count'] = url.count('&')
            features['at_count'] = url.count('@')
            
            # Protocol features
            features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
            features['has_port'] = 1 if parsed_url.port else 0
            
            # Suspicious patterns
            features['has_suspicious_words'] = 1 if any(word in url.lower() for word in 
                                                     ['secure', 'account', 'webscr', 'login', 'signin', 
                                                      'update', 'verify', 'confirm', 'click', 'bank']) else 0
            
            # URL structure
            features['path_depth'] = len([x for x in parsed_url.path.split('/') if x])
            features['digits_in_domain'] = len(re.findall(r'\d', parsed_url.netloc))
            features['letters_in_domain'] = len(re.findall(r'[a-zA-Z]', parsed_url.netloc))
            
            # Entropy (measure of randomness)
            def calculate_entropy(s):
                if not s:
                    return 0
                entropy = 0
                for x in set(s):
                    p_x = s.count(x) / len(s)
                    entropy += -p_x * np.log2(p_x)
                return entropy
            
            features['domain_entropy'] = calculate_entropy(parsed_url.netloc)
            features['path_entropy'] = calculate_entropy(parsed_url.path)
            
        except Exception as e:
            print(f"Error extracting features from {url}: {e}")
            # Return default features if parsing fails
            features = {f'feature_{i}': 0 for i in range(17)}
            
        return features

    def extract_comprehensive_features(self, url):
        """Extract comprehensive features from URL for phishing detection"""
        features = {}
        
        try:
            parsed_url = urlparse(url)
            extracted = tldextract.extract(url)
            
            # Get webpage content for HTML-based features (with error handling)
            html_content = None
            soup = None
            try:
                response = requests.get(url, timeout=5, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
                html_content = response.text
                soup = BeautifulSoup(html_content, 'html.parser')
            except:
                pass
            
            # 1. NumDots - Number of dots in URL
            features['NumDots'] = url.count('.')
            
            # 2. SubdomainLevel - Number of subdomain levels
            features['SubdomainLevel'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
            
            # 3. PathLevel - Number of path levels
            path_parts = [x for x in parsed_url.path.split('/') if x]
            features['PathLevel'] = len(path_parts)
            
            # 4. UrlLength - Total URL length
            features['UrlLength'] = len(url)
            
            # 5. NumDash - Number of dashes in URL
            features['NumDash'] = url.count('-')
            
            # 6. NumDashInHostname - Number of dashes in hostname
            features['NumDashInHostname'] = parsed_url.netloc.count('-')
            
            # 7. AtSymbol - Presence of @ symbol
            features['AtSymbol'] = 1 if '@' in url else 0
            
            # 8. TildeSymbol - Presence of ~ symbol
            features['TildeSymbol'] = 1 if '~' in url else 0
            
            # 9. NumUnderscore - Number of underscores
            features['NumUnderscore'] = url.count('_')
            
            # 10. NumPercent - Number of % symbols (URL encoding)
            features['NumPercent'] = url.count('%')
            
            # 11. NumQueryComponents - Number of query parameters
            query_params = parsed_url.query.split('&') if parsed_url.query else []
            features['NumQueryComponents'] = len([q for q in query_params if q])
            
            # 12. NumAmpersand - Number of & symbols
            features['NumAmpersand'] = url.count('&')
            
            # 13. NumHash - Number of # symbols
            features['NumHash'] = url.count('#')
            
            # 14. NumNumericChars - Number of numeric characters
            features['NumNumericChars'] = len(re.findall(r'\d', url))
            
            # 15. NoHttps - Not using HTTPS (1 if HTTP, 0 if HTTPS)
            features['NoHttps'] = 1 if parsed_url.scheme != 'https' else 0
            
            # 16. RandomString - Check for random strings (high entropy in domain)
            def calculate_entropy(s):
                if not s or len(s) < 2:
                    return 0
                entropy = 0
                for x in set(s):
                    p_x = s.count(x) / len(s)
                    entropy += -p_x * np.log2(p_x)
                return entropy
            
            domain_entropy = calculate_entropy(extracted.domain)
            features['RandomString'] = 1 if domain_entropy > 3.5 else 0
            
            # 17. IpAddress - Using IP address instead of domain
            ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
            features['IpAddress'] = 1 if re.match(ip_pattern, parsed_url.netloc.split(':')[0]) else 0
            
            # 18. DomainInSubdomains - Domain name appears in subdomains
            main_domain = extracted.domain.lower()
            subdomains = extracted.subdomain.lower() if extracted.subdomain else ""
            features['DomainInSubdomains'] = 1 if main_domain in subdomains else 0
            
            # 19. DomainInPaths - Domain name appears in path
            features['DomainInPaths'] = 1 if main_domain in parsed_url.path.lower() else 0
            
            # 20. HttpsInHostname - "https" appears in hostname
            features['HttpsInHostname'] = 1 if 'https' in parsed_url.netloc.lower() else 0
            
            # 21. HostnameLength - Length of hostname
            features['HostnameLength'] = len(parsed_url.netloc)
            
            # 22. PathLength - Length of path
            features['PathLength'] = len(parsed_url.path)
            
            # 23. QueryLength - Length of query string
            features['QueryLength'] = len(parsed_url.query)
            
            # 24. DoubleSlashInPath - Double slash in path
            features['DoubleSlashInPath'] = 1 if '//' in parsed_url.path else 0
            
            # 25. NumSensitiveWords - Number of sensitive/phishing words
            sensitive_words = ['secure', 'account', 'webscr', 'login', 'signin', 'bank', 'update', 
                            'verify', 'confirm', 'paypal', 'ebay', 'amazon', 'microsoft', 'apple']
            features['NumSensitiveWords'] = sum(1 for word in sensitive_words if word in url.lower())
            
            # 26. EmbeddedBrandName - Brand names embedded in domain
            brand_names = ['paypal', 'ebay', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'twitter']
            features['EmbeddedBrandName'] = 1 if any(brand in parsed_url.netloc.lower() for brand in brand_names) else 0
            
            # HTML-based features (require webpage content)
            if soup:
                # 27. PctExtHyperlinks - Percentage of external hyperlinks
                all_links = soup.find_all('a', href=True)
                if all_links:
                    ext_links = [link for link in all_links if self._is_external_link(link['href'], parsed_url.netloc)]
                    features['PctExtHyperlinks'] = len(ext_links) / len(all_links)
                else:
                    features['PctExtHyperlinks'] = 0
                
                # 28. PctExtResourceUrls - Percentage of external resources
                resources = soup.find_all(['img', 'script', 'link'], src=True) + soup.find_all(['link'], href=True)
                if resources:
                    ext_resources = [res for res in resources if self._is_external_resource(res, parsed_url.netloc)]
                    features['PctExtResourceUrls'] = len(ext_resources) / len(resources)
                else:
                    features['PctExtResourceUrls'] = 0
                
                # 29. ExtFavicon - External favicon
                favicon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
                if favicon and favicon.get('href'):
                    features['ExtFavicon'] = 1 if self._is_external_link(favicon['href'], parsed_url.netloc) else 0
                else:
                    features['ExtFavicon'] = 0
                
                # 30. InsecureForms - Forms not using HTTPS
                forms = soup.find_all('form')
                insecure_forms = 0
                for form in forms:
                    action = form.get('action', '')
                    if action.startswith('http://') or (not action.startswith('https://') and parsed_url.scheme == 'http'):
                        insecure_forms += 1
                features['InsecureForms'] = 1 if insecure_forms > 0 else 0
                
                # 31. RelativeFormAction - Forms with relative actions
                relative_forms = sum(1 for form in forms if not form.get('action', '').startswith(('http://', 'https://')))
                features['RelativeFormAction'] = 1 if relative_forms > 0 else 0
                
                # 32. ExtFormAction - Forms with external actions
                ext_form_actions = 0
                for form in forms:
                    action = form.get('action', '')
                    if action and self._is_external_link(action, parsed_url.netloc):
                        ext_form_actions += 1
                features['ExtFormAction'] = 1 if ext_form_actions > 0 else 0
                
                # 33. AbnormalFormAction - Abnormal form actions
                abnormal_forms = 0
                for form in forms:
                    action = form.get('action', '')
                    if action in ['', '#', 'about:blank'] or 'javascript:' in action:
                        abnormal_forms += 1
                features['AbnormalFormAction'] = 1 if abnormal_forms > 0 else 0
                
                # 34. PctNullSelfRedirectHyperlinks - Percentage of null/self-redirect links
                null_links = [link for link in all_links if link['href'] in ['#', '', 'javascript:void(0)']]
                features['PctNullSelfRedirectHyperlinks'] = len(null_links) / len(all_links) if all_links else 0
                
                # 35. FrequentDomainNameMismatch - Domain name mismatches
                features['FrequentDomainNameMismatch'] = self._check_domain_mismatch(soup, parsed_url.netloc)
                
                # 36-40. JavaScript-based features (simplified detection)
                script_content = ' '.join([script.get_text() for script in soup.find_all('script')])
                features['FakeLinkInStatusBar'] = 1 if 'status' in script_content.lower() else 0
                features['RightClickDisabled'] = 1 if 'contextmenu' in script_content.lower() else 0
                features['PopUpWindow'] = 1 if 'window.open' in script_content else 0
                
                # 41. SubmitInfoToEmail - Forms submitting to email
                email_forms = sum(1 for form in forms if 'mailto:' in form.get('action', ''))
                features['SubmitInfoToEmail'] = 1 if email_forms > 0 else 0
                
                # 42. IframeOrFrame - Presence of iframes/frames
                iframes = soup.find_all(['iframe', 'frame'])
                features['IframeOrFrame'] = 1 if len(iframes) > 0 else 0
                
                # 43. MissingTitle - Missing or suspicious title
                title = soup.find('title')
                features['MissingTitle'] = 1 if not title or len(title.get_text().strip()) < 1 else 0
                
                # 44. ImagesOnlyInForm - Forms containing only images
                image_only_forms = 0
                for form in forms:
                    inputs = form.find_all(['input', 'textarea', 'select'])
                    text_inputs = [inp for inp in inputs if inp.get('type') not in ['image', 'submit', 'button']]
                    if len(text_inputs) == 0 and form.find_all('img'):
                        image_only_forms += 1
                features['ImagesOnlyInForm'] = 1 if image_only_forms > 0 else 0
                
            else:
                # Default values when HTML content is not available
                for feature_name in ['PctExtHyperlinks', 'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
                                'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
                                'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
                                'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
                                'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle', 'ImagesOnlyInForm']:
                    features[feature_name] = 0
            
            # RT (Real-time) features - simplified versions
            features['SubdomainLevelRT'] = features['SubdomainLevel']
            features['UrlLengthRT'] = features['UrlLength']
            features['PctExtResourceUrlsRT'] = features['PctExtResourceUrls']
            features['AbnormalExtFormActionR'] = features['AbnormalFormAction']
            features['ExtMetaScriptLinkRT'] = self._check_external_meta_script_links(soup) if soup else 0
            features['PctExtNullSelfRedirectHyperlinksRT'] = features['PctNullSelfRedirectHyperlinks']
            
        except Exception as e:
            print(f"Error extracting features from {url}: {e}")
            # Return default features if parsing fails
            feature_names = ['NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
                            'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
                            'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
                            'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
                            'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
                            'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
                            'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
                            'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
                            'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
                            'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
                            'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
                            'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
                            'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
                            'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT',
                            'PctExtNullSelfRedirectHyperlinksRT']
            features = {name: 0 for name in feature_names}
        
        return features

    def _is_external_link(self, href, current_domain):
        """Check if a link is external to the current domain"""
        if not href or href.startswith(('#', 'javascript:', 'mailto:')):
            return False
        
        if href.startswith(('http://', 'https://')):
            link_domain = urlparse(href).netloc
            return link_domain != current_domain
        
        return False

    def _is_external_resource(self, resource, current_domain):
        """Check if a resource is external to the current domain"""
        src = resource.get('src') or resource.get('href')
        return self._is_external_link(src, current_domain) if src else False

    def _check_domain_mismatch(self, soup, current_domain):
        """Check for frequent domain name mismatches"""
        # Look for domain names in text content that don't match current domain
        text_content = soup.get_text().lower()
        common_domains = ['paypal.com', 'ebay.com', 'amazon.com', 'google.com', 'microsoft.com']
        
        mismatches = 0
        for domain in common_domains:
            if domain in text_content and domain not in current_domain.lower():
                mismatches += 1
        
        return 1 if mismatches > 2 else 0

    def _check_external_meta_script_links(self, soup):
        """Check for external meta/script/link tags"""
        if not soup:
            return 0
        
        external_count = 0
        for tag in soup.find_all(['meta', 'script', 'link']):
            src = tag.get('src') or tag.get('href')
            if src and src.startswith(('http://', 'https://')):
                external_count += 1
        
        return 1 if external_count > 5 else 0
    
    def load_model(self, filename='phishing_detector.pkl'):
        """Load trained models and scaler"""
        try:
            model_data = joblib.load(filename)
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.best_model_name = model_data['best_model_name']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, url, model_name=None):
        """Predict if a URL is phishing or legitimate"""
        if not self.models:
            return {"error": "No trained models available"}
        
        # Use best model if none specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        try:
            # Extract features
            # features = self.extract_features(url)
            features = self.extract_comprehensive_features(url)
            feature_vector = np.array([list(features.values())]).reshape(1, -1)
            
            # Scale features if needed
            if model_name in ['SVM', 'Logistic Regression']:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(feature_vector)[0]
            
            # Get prediction probability if available
            confidence = None
            try:
                if hasattr(model, 'predict_proba'):
                    if model_name in ['SVM', 'Logistic Regression']:
                        probabilities = model.predict_proba(feature_vector)[0]
                    else:
                        probabilities = model.predict_proba(np.array([list(features.values())]).reshape(1, -1))[0]
                    confidence = max(probabilities)
            except:
                pass
            
            return {
                'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
                'prediction_numeric': int(prediction),
                'confidence': float(confidence) if confidence else None,
                'model_used': model_name,
                'features': features,
                'url': url
            }
        
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize detector
detector = PhishingURLDetectorFlask()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_phishing():
    """API endpoint for phishing detection"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        model_name = data.get('model', None)
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        result = detector.predict(url, model_name)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['GET', 'POST'])
def detect_form():
    """Form-based detection page"""
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        model_name = request.form.get('model', None)
        
        if not url:
            return render_template('detect.html', error='URL is required')
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        result = detector.predict(url, model_name)
        
        return render_template('detect.html', result=result, url=url)
    
    return render_template('detect.html')

@app.route('/api/models')
def get_models():
    """Get available models"""
    return jsonify({
        'models': list(detector.models.keys()) if detector.models else [],
        'best_model': detector.best_model_name
    })

@app.route('/batch')
def batch_detection():
    """Batch detection page"""
    return render_template('batch.html')

@app.route('/api/batch_detect', methods=['POST'])
def batch_detect():
    """API endpoint for batch detection"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        model_name = data.get('model', None)
        
        if not urls:
            return jsonify({'error': 'URLs list is required'}), 400
        
        results = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
                
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            result = detector.predict(url, model_name)
            results.append(result)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to load pre-trained model
    # model_loaded = detector.load_model('phishing_detector.pkl')
    # model_loaded = detector.load_model('phishing_detector_url_dataset.pkl')
    model_loaded = detector.load_model('phishing_detector_feature_dataset.pkl')
    
    if not model_loaded:
        print("Warning: No pre-trained model found. Please run the training script first.")
        print("The web app will start but predictions won't work until a model is trained.")
    
    print("Starting Phishing URL Detection Web App...")
    print("Available endpoints:")
    print("- / : Main page")
    print("- /detect : Form-based detection")
    print("- /batch : Batch detection")
    print("- /api/detect : API endpoint for single URL")
    print("- /api/batch_detect : API endpoint for multiple URLs")
    print("- /api/models : Get available models")
    
    app.run(debug=True, host='0.0.0.0', port=5000)