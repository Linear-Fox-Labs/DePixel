from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import numpy as np

class Classifier:
    def __init__(self, contamination=0.1, random_state=42):
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
        self.one_class_svm = OneClassSVM(kernel='rbf', nu=0.1)

    def train(self, features):
        features_array = np.array(features)
        print(f"Training classifiers with {len(features)} samples")
        print(f"Feature array shape: {features_array.shape}")
        print(f"Feature mean: {np.mean(features_array):.4f}")
        print(f"Feature std: {np.std(features_array):.4f}")
        self.isolation_forest.fit(features_array)
        self.one_class_svm.fit(features_array)

    def predict(self, features):
        features_array = np.array(features)
        if_prediction = self.isolation_forest.predict(features_array)
        if_decision = self.isolation_forest.decision_function(features_array)
        svm_prediction = self.one_class_svm.predict(features_array)
        svm_decision = self.one_class_svm.decision_function(features_array)
        
        print(f"Isolation Forest decision: {if_decision[0]:.4f}")
        print(f"One-Class SVM decision: {svm_decision[0]:.4f}")
        
        # Combine predictions (1 if both predict 1, -1 otherwise)
        combined_prediction = np.where((if_prediction == 1) & (svm_prediction == 1), 1, -1)
        
        return combined_prediction