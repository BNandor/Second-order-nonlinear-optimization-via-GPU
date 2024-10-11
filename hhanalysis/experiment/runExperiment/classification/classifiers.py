from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

def getClassifiers():
      return {
            'SVM':lambda :svm.SVC,
            'RF':lambda: RandomForestClassifier
      }