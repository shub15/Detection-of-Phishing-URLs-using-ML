# GPU-Accelerated Phishing URL Detection using Machine Learning
import pandas as pd
import numpy as np
import re
import urllib.parse
from urllib.parse import urlparse
import tldextract
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated libraries
try:
    import cudf  # GPU-accelerated pandas
    import cupy as cp  # GPU-accelerated numpy
    import cuml  # GPU-accelerated scikit-learn
    from cuml.model_selection import train_test_split
    from cuml.ensemble import RandomForestClassifier
    from cuml.svm import SVC
    from cuml.linear_model import LogisticRegression
    from cuml.metrics import accuracy_score, precision_score, recall_score, f1_score
    from cuml.preprocessing import StandardScaler
    from cuml.impute import SimpleImputer
    GPU_AVAILABLE = True
    print("GPU libraries loaded successfully!")
except ImportError as e:
    print(f"GPU libraries not available: {e}")
    print("Falling back to CPU libraries...")
    # Fallback to CPU libraries
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    GPU_AVAILABLE = False

# Additional GPU libraries for deep learning (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = torch.cuda.is_available()
    if PYTORCH_AVAILABLE:
        print(f"PyTorch GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch available but no GPU detected")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available")

import joblib

class GPUPhishingURLDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler() if not GPU_AVAILABLE else cuml.preprocessing.StandardScaler()
        self.feature_names = []
        self.imputer = SimpleImputer(strategy='mean') if not GPU_AVAILABLE else cuml.impute.SimpleImputer(strategy='mean')
        self.gpu_available = GPU_AVAILABLE
        self.device = torch.device('cuda' if PYTORCH_AVAILABLE else 'cpu')
        
        print(f"GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        print(f"PyTorch Device: {self.device}")
        
    def parse_csv_file(self, file_path):
        """
        Parse a CSV file containing URL and Label columns with robust error handling.
        Uses GPU-accelerated pandas (cuDF) if available.
        """
        try:
            # Read the CSV file - use cuDF if available for GPU acceleration
            if self.gpu_available:
                try:
                    df = cudf.read_csv(file_path)
                    print("Using GPU-accelerated CSV reading (cuDF)")
                except:
                    df = pd.read_csv(file_path)
                    print("Fallback to CPU CSV reading")
            else:
                df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if 'URL' not in df.columns or 'Label' not in df.columns:
                raise ValueError("CSV file must contain 'URL' and 'Label' columns")
            
            print(f"Initial dataset size: {len(df)} rows")
            
            # Remove rows with missing URLs or Labels
            df = df.dropna(subset=['URL', 'Label'])
            print(f"After removing missing values: {len(df)} rows")
            
            # Convert to pandas if using cuDF for easier processing
            if self.gpu_available and hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            
            # Process URLs and Labels
            processed_data = []
            skipped_count = 0
            
            for idx, row in df.iterrows():
                try:
                    # Clean and validate URL
                    cleaned_url = self.clean_url(str(row['URL']))
                    if not self.is_valid_url(cleaned_url):
                        skipped_count += 1
                        continue
                    
                    # Process label
                    label = self.process_label(str(row['Label']))
                    if label is None:
                        skipped_count += 1
                        continue
                    
                    processed_data.append((cleaned_url, label))
                    
                except Exception as e:
                    skipped_count += 1
                    continue
            
            print(f"Successfully processed: {len(processed_data)} URLs")
            print(f"Skipped due to errors: {skipped_count} URLs")
            
            if not processed_data:
                print("No valid data found!")
                return [], []
            
            # Separate URLs and labels
            urls, labels = zip(*processed_data)
            return list(urls), list(labels)
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
            return [], []
        except Exception as e:
            print(f"Error processing CSV file: {str(e)}")
            return [], []

    def clean_url(self, url):
        """Clean and standardize URL format with robust error handling."""
        try:
            url = url.strip()
            
            if not url or url.lower() in ['nan', 'null', '']:
                return None
            
            # Remove encoding artifacts and special characters
            url = ''.join(char for char in url if ord(char) < 128 and char.isprintable())
            
            # Fix common URL issues
            url = url.replace(' ', '')
            url = re.sub(r'https?://https?://', 'https://', url)
            
            # Add protocol if missing
            if not re.match(r'^https?://', url, re.IGNORECASE):
                url = 'https://' + url
            
            # Basic URL validation
            url_pattern = re.compile(
                r'^https?://'
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
                r'localhost|'
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
                r'(?::\d+)?'
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(url):
                return None
                
            return url
            
        except Exception:
            return None

    def is_valid_url(self, url):
        """Validate if URL can be processed without errors."""
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
            
            if not parsed.netloc:
                return False
                
            if len(url) > 2048:
                return False
                
            return True
            
        except Exception:
            return False

    def process_label(self, label):
        """Convert label to numeric format."""
        try:
            label_str = str(label).lower().strip()
            
            if label_str in ['good', '0', 'legitimate', 'benign']:
                return 0
            elif label_str in ['bad', '1', 'phishing', 'malicious']:
                return 1
            else:
                return None
                
        except Exception:
            return None

    def handle_feature_nan(self, X):
        """Handle NaN values in feature matrix using GPU if available."""
        try:
            # Convert to appropriate array type
            if self.gpu_available:
                if not isinstance(X, cp.ndarray):
                    X = cp.array(X)
                nan_count = cp.isnan(X).sum()
            else:
                X = np.array(X)
                nan_count = np.isnan(X).sum()
            
            if nan_count > 0:
                print(f"Found {nan_count} NaN values in features. Imputing with mean values...")
                X = self.imputer.fit_transform(X)
                
                # Double-check for remaining NaN values
                if self.gpu_available:
                    remaining_nan = cp.isnan(X).sum()
                    if remaining_nan > 0:
                        print(f"Warning: {remaining_nan} NaN values still remain. Replacing with 0...")
                        X = cp.nan_to_num(X, nan=0.0)
                else:
                    remaining_nan = np.isnan(X).sum()
                    if remaining_nan > 0:
                        print(f"Warning: {remaining_nan} NaN values still remain. Replacing with 0...")
                        X = np.nan_to_num(X, nan=0.0)
            
            return X
            
        except Exception as e:
            print(f"Error handling NaN values: {str(e)}")
            # Fallback
            if self.gpu_available:
                return cp.nan_to_num(X, nan=0.0)
            else:
                return np.nan_to_num(X, nan=0.0)

    def extract_features_safely(self, urls):
        """Extract features from URLs with error handling and GPU acceleration."""
        features = []
        valid_indices = []
        
        print("Extracting features with error handling...")
        
        for i, url in enumerate(urls):
            try:
                feature_vector = self.extract_url_features(url)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    valid_indices.append(i)
                    
            except Exception as e:
                print(f"Skipping URL {i}: {str(e)[:100]}...")
                continue
        
        if not features:
            raise ValueError("No features could be extracted from any URLs")
        
        # Convert to appropriate array type and handle NaN values
        if self.gpu_available:
            X = cp.array(features)
        else:
            X = np.array(features)
            
        X = self.handle_feature_nan(X)
        
        print(f"Successfully extracted features from {len(features)} URLs")
        return X, valid_indices

    def extract_features(self, url):
        """Extract comprehensive features from URL for phishing detection"""
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
            suspicious_words = ['secure', 'account', 'webscr', 'login', 'signin', 
                              'update', 'verify', 'confirm', 'click', 'bank']
            features['has_suspicious_words'] = 1 if any(word in url.lower() for word in suspicious_words) else 0
            
            # URL structure
            features['path_depth'] = len([x for x in parsed_url.path.split('/') if x])
            features['digits_in_domain'] = len(re.findall(r'\d', parsed_url.netloc))
            features['letters_in_domain'] = len(re.findall(r'[a-zA-Z]', parsed_url.netloc))
            
            # Entropy calculation (measure of randomness)
            def calculate_entropy(s):
                if not s:
                    return 0
                entropy = 0
                for x in set(s):
                    p_x = s.count(x) / len(s)
                    if self.gpu_available:
                        entropy += -p_x * cp.log2(p_x)
                    else:
                        entropy += -p_x * np.log2(p_x)
                return float(entropy)
            
            features['domain_entropy'] = calculate_entropy(parsed_url.netloc)
            features['path_entropy'] = calculate_entropy(parsed_url.path)
            
            # Additional security features
            features['has_shortening'] = 1 if any(short in url for short in ['bit.ly', 'tinyurl', 't.co', 'goo.gl']) else 0
            features['special_char_count'] = len(re.findall(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?]', url))
            features['uppercase_count'] = sum(1 for c in url if c.isupper())
            features['lowercase_count'] = sum(1 for c in url if c.islower())
            
        except Exception as e:
            print(f"Error extracting features from {url}: {e}")
            # Return default features if parsing fails
            features = {f'feature_{i}': 0 for i in range(22)}
            
        return features

    def extract_url_features(self, url):
        """Extract features and return as list for compatibility."""
        feature_dict = self.extract_features(url)
        return list(feature_dict.values()) if feature_dict else None
    
    def prepare_data(self, urls, labels):
        """Prepare feature matrix from URLs with GPU acceleration."""
        features_list = []
        
        print("Preparing data with GPU acceleration..." if self.gpu_available else "Preparing data...")
        
        for url in urls:
            features = self.extract_features(url)
            features_list.append(features)
        
        # Convert to appropriate DataFrame type
        if self.gpu_available:
            try:
                df = cudf.DataFrame(features_list)
                print("Using GPU-accelerated DataFrame (cuDF)")
            except:
                df = pd.DataFrame(features_list)
                print("Fallback to CPU DataFrame")
        else:
            df = pd.DataFrame(features_list)
        
        self.feature_names = df.columns.tolist()
        
        # Convert to appropriate array type
        if self.gpu_available and hasattr(df, 'values'):
            X = df.values
            if not isinstance(X, cp.ndarray):
                X = cp.array(X)
            y = cp.array(labels)
        else:
            X = df.values
            y = np.array(labels)
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models with GPU acceleration"""
        print(f"Training on {'GPU' if self.gpu_available else 'CPU'}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models with GPU acceleration
        if self.gpu_available:
            models = {
                'Random Forest': cuml.ensemble.RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': cuml.svm.SVC(kernel='rbf', random_state=42),
                'Logistic Regression': cuml.linear_model.LogisticRegression(random_state=42, max_iter=1000)
            }
        else:
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
        
        results = {}
        
        print("Training and evaluating models...")
        print("=" * 50)
        
        for name, model in models.items():
            print(f"\nTraining {name} on {'GPU' if self.gpu_available else 'CPU'}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Convert GPU arrays to CPU for metric calculations if needed
            if self.gpu_available:
                y_test_cpu = y_test.get() if hasattr(y_test, 'get') else y_test
                y_pred_cpu = y_pred.get() if hasattr(y_pred, 'get') else y_pred
            else:
                y_test_cpu = y_test
                y_pred_cpu = y_pred
            
            # Calculate metrics
            try:
                if self.gpu_available:
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                else:
                    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
                    precision = precision_score(y_test_cpu, y_pred_cpu)
                    recall = recall_score(y_test_cpu, y_pred_cpu)
                    f1 = f1_score(y_test_cpu, y_pred_cpu)
            except:
                # Fallback to sklearn metrics
                from sklearn.metrics import accuracy_score as sk_accuracy
                from sklearn.metrics import precision_score as sk_precision
                from sklearn.metrics import recall_score as sk_recall
                from sklearn.metrics import f1_score as sk_f1
                
                accuracy = sk_accuracy(y_test_cpu, y_pred_cpu)
                precision = sk_precision(y_test_cpu, y_pred_cpu)
                recall = sk_recall(y_test_cpu, y_pred_cpu)
                f1 = sk_f1(y_test_cpu, y_pred_cpu)
            
            results[name] = {
                'model': model,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        
        print(f"\nBest Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
        
        # Store models
        self.models = {name: result['model'] for name, result in results.items()}
        self.best_model_name = best_model_name
        
        return results
    
    def predict(self, url, model_name=None):
        """Predict if a URL is phishing or legitimate using GPU acceleration"""
        if not self.models:
            raise ValueError("No trained models available. Please train models first.")
        
        # Use best model if none specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        # Extract features
        features = self.extract_features(url)
        
        if self.gpu_available:
            feature_vector = cp.array([list(features.values())]).reshape(1, -1)
        else:
            feature_vector = np.array([list(features.values())]).reshape(1, -1)
        
        # Scale features if needed
        if model_name in ['SVM', 'Logistic Regression']:
            feature_vector = self.scaler.transform(feature_vector)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(feature_vector)[0]
        
        # Convert GPU result to CPU if needed
        if self.gpu_available and hasattr(prediction, 'get'):
            prediction = prediction.get()
        
        # Get prediction probability if available
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_vector)[0]
                if self.gpu_available and hasattr(probabilities, 'get'):
                    probabilities = probabilities.get()
                confidence = float(max(probabilities))
            else:
                confidence = None
        except:
            confidence = None
        
        return {
            'prediction': 'Phishing' if int(prediction) == 1 else 'Legitimate',
            'confidence': confidence,
            'model_used': model_name,
            'features': features,
            'gpu_used': self.gpu_available
        }
    
    def save_model(self, filename='gpu_phishing_detector.pkl'):
        """Save the trained models and scaler"""
        # Convert GPU models to CPU for saving
        cpu_models = {}
        for name, model in self.models.items():
            if self.gpu_available and hasattr(model, 'get_params'):
                # For cuML models, we might need special handling
                cpu_models[name] = model
            else:
                cpu_models[name] = model
        
        model_data = {
            'models': cpu_models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name,
            'gpu_trained': self.gpu_available
        }
        
        joblib.dump(model_data, filename)
        print(f"Models saved to {filename}")
    
    def load_model(self, filename='gpu_phishing_detector.pkl'):
        """Load trained models and scaler"""
        model_data = joblib.load(filename)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        
        gpu_trained = model_data.get('gpu_trained', False)
        print(f"Models loaded from {filename} (Originally trained on {'GPU' if gpu_trained else 'CPU'})")

def main():
    """Main function to demonstrate the GPU-accelerated phishing detection system"""
    print("GPU-Accelerated Phishing URL Detection System")
    print("=" * 50)
    
    # Check GPU availability
    if GPU_AVAILABLE:
        print("🚀 GPU acceleration enabled with RAPIDS cuML/cuDF")
    else:
        print("⚠️  GPU libraries not available, using CPU fallback")
    
    if PYTORCH_AVAILABLE:
        print(f"🔥 PyTorch GPU available: {torch.cuda.get_device_name(0)}")
    
    # Initialize detector
    detector = GPUPhishingURLDetector()
    
    # Use your dataset file path
    file_path = "../Datasets/phishing_urls/phishing_site_urls.csv"  # Update this path
    
    print(f"\nLoading dataset from: {file_path}")
    urls, labels = detector.parse_csv_file(file_path)
    
    if not urls or not labels:
        print("No valid data found. Creating sample dataset...")
        # Create sample data if file not found
        legitimate_urls = [
            "https://www.google.com/search?q=example",
            "https://github.com/user/repo",
            "https://stackoverflow.com/questions/123",
            "https://www.amazon.com/products",
            "https://en.wikipedia.org/wiki/Article"
        ] * 100
        
        phishing_urls = [
            "http://secure-paypal-update.com/signin",
            "https://amazon-security-check.net/login",
            "http://192.168.1.1/bank-login",
            "https://facebook-security.tk/verify",
            "http://gmail-account-verify.ml/confirm"
        ] * 100
        
        urls = legitimate_urls + phishing_urls
        labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)
    
    print(f"\nDataset loaded: {len(urls)} URLs")
    print(f"Legitimate URLs: {sum(1 for label in labels if label == 0)}")
    print(f"Phishing URLs: {sum(1 for label in labels if label == 1)}")
    
    # Prepare data with GPU acceleration
    print("\nExtracting features with GPU acceleration...")
    X, y = detector.prepare_data(urls, labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Using {'GPU arrays (CuPy)' if detector.gpu_available else 'CPU arrays (NumPy)'}")
    
    # Train models with GPU acceleration
    results = detector.train_models(X, y)
    
    # Test with sample URLs
    print("\n" + "=" * 50)
    print("Testing with sample URLs:")
    print("=" * 50)
    
    test_urls = [
        "https://www.google.com/search?q=machine+learning",
        "http://secure-paypal-update.com/signin?user=123",
        "https://github.com/user/repository",
        "http://192.168.1.1/bank-login?token=abc123",
        "https://www.amazon.com/products/electronics"
    ]
    
    for url in test_urls:
        result = detector.predict(url)
        print(f"\nURL: {url[:60]}...")
        print(f"Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.4f}")
        print(f"Model: {result['model_used']}")
        print(f"GPU Used: {result['gpu_used']}")
    
    # Save model
    detector.save_model('gpu_phishing_detector.pkl')
    
    print(f"\n🎉 Training completed using {'GPU' if detector.gpu_available else 'CPU'} acceleration!")
    
    return detector

if __name__ == "__main__":
    detector = main()