# src/classical_ml/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_logistic_regression():
    return LogisticRegression(max_iter=1000)

def get_svm():
    return SVC(kernel='linear', C=1.0)

def get_random_forest(n_estimators=100):
    return RandomForestClassifier(n_estimators=n_estimators)
