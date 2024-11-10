from sklearn.ensemble import RandomForestClassifier
import math

class CustomRandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, **kwargs):
        # If using percentage and want to see actual number of features used
        # rounded_kwargs = kwargs.copy()
        # for param, value in kwargs.items():
        #     if isinstance(value, float):
        #         rounded_kwargs[param] = int(math.floor(value))
        #         # print(f"Using {actual_features} features (rounded down from {self.max_features * n_features})")
        self.max_features=None
        # print(f'MAX_FEATURES : n_features{self.max_features} - {self.n_features}')
        return super().fit(X, y)
