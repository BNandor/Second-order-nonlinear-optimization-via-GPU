from sklearn import svm
import math

class CustomSVC(svm.SVC):
    def fit(self, X, y, **kwargs):
        self.max_iter=1000
        return super().fit(X, y)
