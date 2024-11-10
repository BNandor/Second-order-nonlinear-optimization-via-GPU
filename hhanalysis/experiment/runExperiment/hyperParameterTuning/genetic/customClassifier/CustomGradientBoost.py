from sklearn.ensemble import GradientBoostingClassifier
import math

class CustomGradientBoostClassifier(GradientBoostingClassifier):
    def fit(self, X, y, **kwargs):
        # If using percentage and want to see actual number of features used
        # rounded_kwargs = kwargs.copy()
        # for param, value in kwargs.items():
        #     if isinstance(value, float):
        #         rounded_kwargs[param] = int(math.floor(value))
        #         # print(f"Using {actual_features} features (rounded down from {self.max_features * n_features})")
        if self.criterion == 'squared_error':
            self.criterion='mse'
        self.warm_start=False
        # print(f'MAX_FEATURES : n_features{self.max_features} - {self.n_features}')
        return super().fit(X, y)
