from sklearn.tree import DecisionTreeClassifier
import math

class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def fit(self, X, y, **kwargs):
        self.max_features=None
        return super().fit(X, y)
