# Flask Web Application for Phishing URL Detection
from flask import Flask, render_template, request, jsonify
import os
import sys
import joblib
import numpy as np
from urllib.parse import urlparse
import tldextract
import re

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
            features = self.extract_features(url)
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
    model_loaded = detector.load_model('phishing_detector.pkl')
    
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