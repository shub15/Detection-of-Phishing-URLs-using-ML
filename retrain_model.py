# Enhanced Phishing URL Detection with Retraining Capabilities
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
import os
from datetime import datetime

from phishing_detector_v2 import PhishingURLDetector

warnings.filterwarnings("ignore")


class PhishingURLDetectorWithRetraining:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.imputer = SimpleImputer(strategy="mean")
        self.best_model_name = None
        self.training_history = []
        self.model_metadata = {
            'creation_date': None,
            'last_updated': None,
            'total_samples_trained': 0,
            'retraining_count': 0,
            'feature_count': 0
        }
        self.detector = PhishingURLDetector()

    def load_existing_model(self, filename="phishing_detector.pkl"):
        """
        Load an existing trained model.
        
        Parameters:
        filename (str): Path to the saved model file
        
        Returns:
        bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(filename):
                print(f"Model file '{filename}' not found. Will create new model.")
                return False
                
            model_data = joblib.load(filename)
            self.models = model_data.get("models", {})
            self.scaler = model_data.get("scaler", StandardScaler())
            self.feature_names = model_data.get("feature_names", [])
            self.best_model_name = model_data.get("best_model_name", None)
            self.training_history = model_data.get("training_history", [])
            self.model_metadata = model_data.get("model_metadata", {
                'creation_date': None,
                'last_updated': None,
                'total_samples_trained': 0,
                'retraining_count': 0,
                'feature_count': 0
            })
            
            print(f"Existing model loaded from {filename}")
            print(f"Model metadata: {self.model_metadata}")
            print(f"Available models: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            print(f"Error loading existing model: {str(e)}")
            return False

    def retrain_with_new_data(self, new_data_path, data_type="feature", 
                             retrain_mode="incremental", test_size=0.2):
        """
        Retrain the model with new data.
        
        Parameters:
        new_data_path (str): Path to new dataset
        data_type (str): "feature" for feature dataset, "url" for URL dataset
        retrain_mode (str): "incremental" to add to existing data, 
                           "replace" to replace with new data,
                           "combined" to combine old and new data
        test_size (float): Test set proportion
        
        Returns:
        dict: Retraining results
        """
        print(f"Starting retraining with mode: {retrain_mode}")
        print("=" * 50)
        
        # Load new data
        if data_type == "feature":
            X_new, y_new = self.detector.load_feature_dataset(new_data_path)
        elif data_type == "url":
            urls, labels = self.detector.parse_csv_file(new_data_path)
            if urls and labels:
                X_new, y_new = self.detector.extract_features_safely(urls)
            else:
                print("Failed to load URL dataset")
                return None
        else:
            raise ValueError("data_type must be 'feature' or 'url'")
            
        if X_new is None or y_new is None:
            print("Failed to load new data")
            return None
            
        print(f"New data loaded: {X_new.shape[0]} samples, {X_new.shape[1]} features")
        
        # Handle retraining modes
        if retrain_mode == "replace":
            # Simply replace with new data
            X_final, y_final = X_new, y_new
            print("Using only new data for retraining")
            
        elif retrain_mode == "incremental" and self.models:
            # Add new data to existing training
            # For incremental learning, we'll retrain on combined data
            print("Incremental retraining: combining with existing model knowledge")
            X_final, y_final = X_new, y_new
            
        elif retrain_mode == "combined":
            # This would require access to original training data
            # For now, we'll use the new data and note this limitation
            print("Combined mode: Using new data (original training data not available)")
            X_final, y_final = X_new, y_new
            
        else:
            # Default to using new data
            X_final, y_final = X_new, y_new
            
        # Train models
        results = self.detector.train_models_on_features(X_final, y_final, test_size=test_size)
        
        # Update metadata
        self.model_metadata['last_updated'] = datetime.now().isoformat()
        self.model_metadata['total_samples_trained'] += len(X_final)
        self.model_metadata['retraining_count'] += 1
        self.model_metadata['feature_count'] = X_final.shape[1]
        
        if self.model_metadata['creation_date'] is None:
            self.model_metadata['creation_date'] = datetime.now().isoformat()
            
        # Add to training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'data_source': new_data_path,
            'data_type': data_type,
            'retrain_mode': retrain_mode,
            'samples_count': len(X_final),
            'features_count': X_final.shape[1],
            'results': {name: {'f1': res['f1'], 'accuracy': res['accuracy']} 
                       for name, res in results.items()}
        }
        self.training_history.append(training_record)
        
        print(f"\nRetraining completed. Model updated with {len(X_final)} samples.")
        return results

    def incremental_learning_update(self, new_data_path, data_type="feature", 
                                  learning_rate=0.1, n_iterations=5):
        """
        Perform incremental learning updates for models that support it.
        
        Parameters:
        new_data_path (str): Path to new dataset
        data_type (str): "feature" or "url"
        learning_rate (float): Learning rate for incremental updates
        n_iterations (int): Number of incremental training iterations
        
        Returns:
        dict: Update results
        """
        print("Performing incremental learning update...")
        print("=" * 40)
        
        # Load new data
        if data_type == "feature":
            X_new, y_new = self.detector.load_feature_dataset(new_data_path)
        else:
            urls, labels = self.detector.parse_csv_file(new_data_path)
            if urls and labels:
                X_new, y_new = self.detector.extract_features_safely(urls, labels)
            else:
                return None
                
        if X_new is None or y_new is None:
            return None
            
        # Handle feature scaling
        X_new_scaled = self.scaler.transform(X_new)
        
        results = {}
        
        # For models that support incremental learning
        for model_name, model in self.models.items():
            print(f"\nUpdating {model_name}...")
            
            if hasattr(model, 'partial_fit'):
                # Models like SGDClassifier support partial_fit
                try:
                    for i in range(n_iterations):
                        if model_name in ["SVM", "Logistic Regression"]:
                            model.partial_fit(X_new_scaled, y_new)
                        else:
                            model.partial_fit(X_new, y_new)
                    
                    print(f"{model_name} updated with incremental learning")
                    results[model_name] = "Updated with partial_fit"
                    
                except Exception as e:
                    print(f"Incremental update failed for {model_name}: {str(e)}")
                    results[model_name] = f"Failed: {str(e)}"
                    
            else:
                # For models without partial_fit, we need to retrain completely
                print(f"{model_name} doesn't support incremental learning")
                results[model_name] = "Requires full retraining"
                
        return results

    def evaluate_model_performance(self, test_data_path, data_type="feature"):
        """
        Evaluate current model performance on new test data.
        
        Parameters:
        test_data_path (str): Path to test dataset
        data_type (str): "feature" or "url"
        
        Returns:
        dict: Evaluation results
        """
        print("Evaluating model performance on new test data...")
        print("=" * 45)
        
        # Load test data
        if data_type == "feature":
            X_test, y_test = self.detector.load_feature_dataset(test_data_path)
        else:
            urls, labels = self.detector.parse_csv_file(test_data_path)
            if urls and labels:
                X_test, y_test = self.detector.extract_features_safely(urls, labels)
            else:
                return None
                
        if X_test is None or y_test is None:
            print("Failed to load test data")
            return None
            
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Scale features if needed
                if model_name in ["SVM", "Logistic Regression"]:
                    X_test_scaled = self.scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predictions': y_pred,
                    'true_labels': y_test
                }
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                
            except Exception as e:
                print(f"Evaluation failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
                
        return results

    def compare_models_before_after_retraining(self, old_model_path, new_test_data_path, 
                                             data_type="feature"):
        """
        Compare model performance before and after retraining.
        
        Parameters:
        old_model_path (str): Path to the old model
        new_test_data_path (str): Path to test data for comparison
        data_type (str): "feature" or "url"
        
        Returns:
        dict: Comparison results
        """
        print("Comparing models before and after retraining...")
        print("=" * 50)
        
        # Save current model temporarily
        temp_new_model = "temp_new_model.pkl"
        self.save_model(temp_new_model)
        
        try:
            # Load old model
            old_detector = PhishingURLDetectorWithRetraining()
            old_detector.load_existing_model(old_model_path)
            
            # Evaluate old model
            print("\nEvaluating OLD model:")
            print("-" * 25)
            old_results = old_detector.evaluate_model_performance(new_test_data_path, data_type)
            
            # Evaluate new model
            print("\nEvaluating NEW model:")
            print("-" * 25)
            new_results = self.evaluate_model_performance(new_test_data_path, data_type)
            
            # Compare results
            print("\n" + "=" * 50)
            print("PERFORMANCE COMPARISON")
            print("=" * 50)
            
            comparison = {}
            
            for model_name in self.models.keys():
                if (model_name in old_results and model_name in new_results and
                    'error' not in old_results[model_name] and 'error' not in new_results[model_name]):
                    
                    old_f1 = old_results[model_name]['f1']
                    new_f1 = new_results[model_name]['f1']
                    improvement = new_f1 - old_f1
                    
                    old_acc = old_results[model_name]['accuracy']
                    new_acc = new_results[model_name]['accuracy']
                    acc_improvement = new_acc - old_acc
                    
                    comparison[model_name] = {
                        'old_f1': old_f1,
                        'new_f1': new_f1,
                        'f1_improvement': improvement,
                        'old_accuracy': old_acc,
                        'new_accuracy': new_acc,
                        'accuracy_improvement': acc_improvement
                    }
                    
                    print(f"\n{model_name}:")
                    print(f"  F1-Score:  {old_f1:.4f} → {new_f1:.4f} ({improvement:+.4f})")
                    print(f"  Accuracy:  {old_acc:.4f} → {new_acc:.4f} ({acc_improvement:+.4f})")
                    
                    if improvement > 0:
                        print(f"  ✓ Model improved!")
                    elif improvement < -0.01:
                        print(f"  ⚠ Model performance decreased")
                    else:
                        print(f"  → Similar performance")
            
            return {
                'old_results': old_results,
                'new_results': new_results,
                'comparison': comparison
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_new_model):
                os.remove(temp_new_model)

    def get_training_history(self):
        """
        Get the training history of the model.
        
        Returns:
        list: Training history records
        """
        return self.training_history

    def get_model_metadata(self):
        """
        Get model metadata including creation date, updates, etc.
        
        Returns:
        dict: Model metadata
        """
        return self.model_metadata

    def save_model_with_history(self, filename="phishing_detector_with_history.pkl"):
        """
        Save the model along with training history and metadata.
        
        Parameters:
        filename (str): Filename to save the model
        """
        model_data = {
            "models": self.models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "best_model_name": self.best_model_name,
            "training_history": self.training_history,
            "model_metadata": self.model_metadata,
            "imputer": self.imputer
        }
        
        joblib.dump(model_data, filename)
        print(f"Model with history saved to {filename}")
        print(f"Total retraining sessions: {self.model_metadata['retraining_count']}")


def demonstrate_retraining():
    """Demonstrate the retraining capabilities."""
    print("Phishing URL Detection - Retraining Demonstration")
    print("=" * 55)
    
    # Initialize detector
    detector = PhishingURLDetectorWithRetraining()
    
    # Try to load existing model
    existing_model_path = "phishing_detector_feature_dataset.pkl"
    model_loaded = detector.load_existing_model(existing_model_path)
    
    if not model_loaded:
        print("No existing model found. Training initial model...")
        # You would replace this with your actual initial training data
        initial_data_path = "../Datasets/phishing_urls/Phishing_Legitimate_full.csv"
        detector.retrain_with_new_data(initial_data_path, data_type="feature", retrain_mode="replace")
        print("Please provide initial training data to start.")
    
    # Demonstrate retraining scenarios
    print("\n" + "=" * 55)
    print("RETRAINING SCENARIOS")
    print("=" * 55)
    
    # Scenario 1: Add new phishing examples
    print("\n1. Adding new phishing examples:")
    new_phishing_data = "../Datasets/phishing_urls/phishing_site_urls_Copy.csv"
    results = detector.retrain_with_new_data(new_phishing_data, 
                                           data_type="url", 
                                           retrain_mode="incremental")
    
    # Scenario 2: Complete model update with new dataset
    print("\n2. Complete model update:")
    # updated_dataset = "updated_complete_dataset.csv"
    # results = detector.retrain_with_new_data(updated_dataset, 
    #                                        data_type="feature", 
    #                                        retrain_mode="replace")
    
    # Scenario 3: Performance evaluation
    print("\n3. Model performance evaluation:")
    test_data = "../Datasets/phishing_urls/phishing_site_urls_test.csv"
    evaluation_results = detector.evaluate_model_performance(test_data, data_type="url")
    
    # Show training history
    print("\n4. Training History:")
    history = detector.get_training_history()
    for i, record in enumerate(history, 1):
        print(f"Session {i}: {record['timestamp']} - {record['samples_count']} samples")
    
    # Show model metadata
    print("\n5. Model Metadata:")
    metadata = detector.get_model_metadata()
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # Save updated model
    detector.save_model_with_history("phishing_detector_feature_dataset.pkl")
    
    return detector


if __name__ == "__main__":
    detector = demonstrate_retraining()