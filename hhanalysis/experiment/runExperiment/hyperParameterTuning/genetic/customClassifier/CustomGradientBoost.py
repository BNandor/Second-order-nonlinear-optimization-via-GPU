from sklearn.ensemble import GradientBoostingClassifier
import math

class CustomGradientBoostClassifier(GradientBoostingClassifier):
    def fit(self, X, y, **kwargs):
        if self.criterion == 'squared_error':
            self.criterion='mse'
        self.warm_start=False
        return super().fit(X, y)
