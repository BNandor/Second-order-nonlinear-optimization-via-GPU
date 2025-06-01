from sklearn.ensemble import RandomForestClassifier
import math

class CustomRandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, **kwargs):
        self.max_features=None
        return super().fit(X, y)
