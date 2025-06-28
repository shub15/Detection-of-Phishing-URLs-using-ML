# Phishing URL Detection using Machine Learning
import pandas as pd
import numpy as np
import re
import urllib.parse
from urllib.parse import urlparse
import tldextract
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class PhishingURLDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.imputer = SimpleImputer(strategy='mean')
        
    def parse_csv_file(self, file_path):
        """
        Parse a CSV file containing URL and Label columns with robust error handling.
        
        Parameters:
        file_path (str): Path to the CSV file
        
        Returns:
        tuple: (urls, labels) where urls is a list of processed URLs and 
               labels is a list of numeric labels (0 for 'good', 1 for 'bad')
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if 'URL' not in df.columns or 'Label' not in df.columns:
                raise ValueError("CSV file must contain 'URL' and 'Label' columns")
            
            print(f"Initial dataset size: {len(df)} rows")
            
            # Remove rows with missing URLs or Labels
            df = df.dropna(subset=['URL', 'Label'])
            print(f"After removing missing values: {len(df)} rows")
            
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
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty")
            return [], []
        except Exception as e:
            print(f"Error processing CSV file: {str(e)}")
            return [], []

    def clean_url(self, url):
        """
        Clean and standardize URL format with robust error handling.
        
        Parameters:
        url (str): Raw URL string
        
        Returns:
        str: Cleaned URL or None if invalid
        """
        try:
            # Remove leading/trailing whitespace
            url = url.strip()
            
            # Check for empty or invalid URLs
            if not url or url.lower() in ['nan', 'null', '']:
                return None
            
            # Remove obvious encoding artifacts and special characters
            # Keep only printable ASCII characters for URL processing
            url = ''.join(char for char in url if ord(char) < 128 and char.isprintable())
            
            # Fix common URL issues
            url = url.replace(' ', '')  # Remove spaces
            url = re.sub(r'https?://https?://', 'https://', url)  # Fix double protocols
            
            # Add protocol if missing
            if not re.match(r'^https?://', url, re.IGNORECASE):
                url = 'https://' + url
            
            # Basic URL validation using regex
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(url):
                return None
                
            return url
            
        except Exception:
            return None

    def is_valid_url(self, url):
        """
        Validate if URL can be processed without errors.
        
        Parameters:
        url (str): URL to validate
        
        Returns:
        bool: True if URL is valid for processing
        """
        if not url:
            return False
            
        try:
            # Try to parse the URL
            parsed = urlparse(url)
            
            # Check if essential components exist
            if not parsed.netloc:
                return False
                
            # Check for reasonable length (avoid extremely long URLs)
            if len(url) > 2048:
                return False
                
            return True
            
        except Exception:
            return False

    def process_label(self, label):
        """
        Convert label to numeric format.
        
        Parameters:
        label (str): Label string
        
        Returns:
        int: 0 for 'good', 1 for 'bad', None for invalid
        """
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
        """
        Handle NaN values in feature matrix.
        
        Parameters:
        X (array-like): Feature matrix that may contain NaN values
        
        Returns:
        array: Feature matrix with NaN values handled
        """
        try:
            # Convert to numpy array if not already
            X = np.array(X)
            
            # Check for NaN values
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                print(f"Found {nan_count} NaN values in features. Imputing with mean values...")
                
                # Fit and transform the data
                X = self.imputer.fit_transform(X)
                
                # Double-check for any remaining NaN values
                remaining_nan = np.isnan(X).sum()
                if remaining_nan > 0:
                    print(f"Warning: {remaining_nan} NaN values still remain. Replacing with 0...")
                    X = np.nan_to_num(X, nan=0.0)
            
            return X
            
        except Exception as e:
            print(f"Error handling NaN values: {str(e)}")
            # Fallback: replace all NaN with 0
            return np.nan_to_num(X, nan=0.0)

    def extract_features_safely(self, urls):
        """
        Extract features from URLs with error handling.
        
        Parameters:
        urls (list): List of URLs
        
        Returns:
        array: Feature matrix
        """
        features = []
        valid_indices = []
        
        print("Extracting features with error handling...")
        
        for i, url in enumerate(urls):
            try:
                # Your existing feature extraction method here
                # Replace this with your actual feature extraction logic
                feature_vector = self.extract_features(url)
                
                if feature_vector is not None:
                    features.append(feature_vector)
                    valid_indices.append(i)
                    
            except Exception as e:
                print(f"Skipping URL {i}: {str(e)[:100]}...")
                continue
        
        if not features:
            raise ValueError("No features could be extracted from any URLs")
        
        # Convert to numpy array and handle NaN values
        X = np.array(features)
        X = self.handle_feature_nan(X)
        
        print(f"Successfully extracted features from {len(features)} URLs")
        return X, valid_indices

    def extract_url_features(self, url):
        """
        Placeholder for your actual feature extraction method.
        Replace this with your existing feature extraction logic.
        
        Parameters:
        url (str): URL to extract features from
        
        Returns:
        list: Feature vector
        """
        # This is a placeholder - replace with your actual feature extraction
        # For now, returning dummy features to avoid errors
        try:
            parsed = urlparse(url)
            
            # Example basic features - replace with your actual features
            features = [
                len(url),                          # URL length
                len(parsed.netloc),                # Domain length
                url.count('.'),                    # Number of dots
                url.count('-'),                    # Number of hyphens
                url.count('_'),                    # Number of underscores
                url.count('/'),                    # Number of slashes
                url.count('?'),                    # Number of question marks
                url.count('='),                    # Number of equals
                url.count('&'),                    # Number of ampersands
                1 if 'https' in url else 0,       # HTTPS usage
                # Add your other 28 features here...
            ]
            
            # Pad to 38 features if needed (replace with your actual 38 features)
            while len(features) < 38:
                features.append(0)
            
            return features[:38]  # Ensure exactly 38 features
            
        except Exception:
            return None
        
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
    
    def create_sample_dataset(self, size=1000):
        """Create a sample dataset for demonstration"""
        # Legitimate URLs (examples)
        legitimate_urls = [
            "https://www.google.com",
            "https://github.com/user/repo",
            "https://stackoverflow.com/questions",
            "https://www.amazon.com/products",
            "https://en.wikipedia.org/wiki/Article",
            "https://www.youtube.com/watch",
            "https://www.facebook.com",
            "https://twitter.com/user",
            "https://www.linkedin.com/in/profile",
            "https://www.microsoft.com/products"
        ]
        
        # Phishing URLs (examples - these are simulated patterns)
        phishing_urls = [
            "http://secure-paypal-update.com/signin",
            "https://amazon-security-check.net/login",
            "http://192.168.1.1/bank-login",
            "https://facebook-security.tk/verify",
            "http://gmail-account-verify.ml/confirm",
            "https://paypal-resolution-center.ga/update",
            "http://apple-id-locked.cf/unlock",
            "https://microsoft-security-alert.tk/verify",
            "http://bank-of-america-alert.ml/login",
            "https://amazon-prime-renewal.ga/payment"
        ]
        
        # Generate variations
        urls = []
        labels = []
        
        # Add legitimate URLs with variations
        for _ in range(size // 2):
            base_url = np.random.choice(legitimate_urls)
            # Add some random parameters or paths
            if np.random.random() > 0.5:
                base_url += f"/page{np.random.randint(1, 100)}"
            if np.random.random() > 0.7:
                base_url += f"?id={np.random.randint(1, 1000)}"
            urls.append(base_url)
            labels.append(0)  # 0 = legitimate
        
        # Add phishing URLs with variations
        for _ in range(size // 2):
            base_url = np.random.choice(phishing_urls)
            # Add suspicious elements
            if np.random.random() > 0.5:
                base_url += f"?user={np.random.randint(1, 1000)}&token=abc123"
            urls.append(base_url)
            labels.append(1)  # 1 = phishing
        
        return urls, labels
    
    def prepare_data(self, urls, labels):
        """Prepare feature matrix from URLs"""
        features_list = []
        
        for url in urls:
            features = self.extract_features(url)
            features_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()
        
        return df.values, np.array(labels)
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        print("Training and evaluating models...")
        print("=" * 50)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
        
        # Store models
        self.models = {name: result['model'] for name, result in results.items()}
        self.best_model_name = best_model_name
        
        return results
    
    def predict(self, url, model_name=None):
        """Predict if a URL is phishing or legitimate"""
        if not self.models:
            raise ValueError("No trained models available. Please train models first.")
        
        # Use best model if none specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
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
        try:
            if hasattr(model, 'predict_proba'):
                if model_name in ['SVM', 'Logistic Regression']:
                    probabilities = model.predict_proba(feature_vector)[0]
                else:
                    probabilities = model.predict_proba(np.array([list(features.values())]).reshape(1, -1))[0]
                confidence = max(probabilities)
            else:
                confidence = None
        except:
            confidence = None
        
        return {
            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
            'confidence': confidence,
            'model_used': model_name,
            'features': features
        }
    
    def save_model(self, filename='phishing_detector.pkl'):
        """Save the trained models and scaler"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name
        }
        joblib.dump(model_data, filename)
        print(f"Models saved to {filename}")
    
    def load_model(self, filename='phishing_detector.pkl'):
        """Load trained models and scaler"""
        model_data = joblib.load(filename)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        print(f"Models loaded from {filename}")

def main():
    """Main function to demonstrate the phishing detection system"""
    print("Phishing URL Detection System")
    print("=" * 40)
    
    # Initialize detector
    detector = PhishingURLDetector()
    
    # # Create sample dataset
    # print("\nCreating sample dataset...")
    # urls, labels = detector.create_sample_dataset(1000)
    
    file_path = "../Datasets/phishing_urls/phishing_site_urls_Copy.csv"
    
    urls, labels = detector.parse_csv_file(file_path)
    
    if urls and labels:
        print(f"\nSample data:")
        for i in range(min(5, len(urls))):
            print(f"{i+1}. URL: {urls[i][:80]}..., Label: {labels[i]}")
        
        # Extract features safely
        try:
            X, valid_indices = detector.extract_features_safely(urls)
            y = np.array(labels)[valid_indices]  # Match labels to valid URLs
            
            print(f"\nFinal dataset: {X.shape[0]} samples with {X.shape[1]} features")
            print(f"Labels distribution: {np.bincount(y)}")
            
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
    
    # Prepare data
    print("Extracting features...")
    X, y = detector.prepare_data(urls, labels)
    
    print(f"Dataset created: {len(urls)} URLs with {X.shape[1]} features each")
    print(f"Legitimate URLs: {sum(1 for label in labels if label == 0)}")
    print(f"Phishing URLs: {sum(1 for label in labels if label == 1)}")
    
    # Train models
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
        print(f"\nURL: {url}")
        print(f"Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']:.4f}")
        print(f"Model: {result['model_used']}")
    
    # Save model
    detector.save_model()
    
    return detector

if __name__ == "__main__":
    detector = main()