from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
import numpy as np

class Classifier:
    def __init__(self, contamination=0.1, random_state=42):
        self.ocsvm = OneClassSVM(kernel='rbf', nu=contamination)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=random_state)

    def train(self, features):
        features_array = np.array(features)
        print(f"Training classifier with {len(features)} samples")
        print(f"Feature array shape: {features_array.shape}")
        
        self.ocsvm.fit(features_array)
        
        # Create synthetic negative samples for RandomForest
        synthetic_negatives = features_array + np.random.normal(0, 0.1, features_array.shape)
        X = np.vstack([features_array, synthetic_negatives])
        y = np.hstack([np.ones(len(features)), np.zeros(len(features))])
        self.rf.fit(X, y)

    def predict(self, features):
        features_array = np.array(features)
        ocsvm_decision = self.ocsvm.decision_function(features_array)
        rf_proba = self.rf.predict_proba(features_array)[:, 1]
        
        print(f"One-Class SVM decision: {ocsvm_decision[0]:.4f}")
        print(f"Random Forest probability: {rf_proba[0]:.4f}")
        
        # Combine decisions (you may need to adjust these thresholds)
        combined_decision = (ocsvm_decision > -0.1) & (rf_proba > 0.5)
        return combined_decision