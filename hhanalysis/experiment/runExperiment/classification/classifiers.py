from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def getClassifiers():
      return {
            'SVM':lambda :svm.SVC,
            'RF':lambda: RandomForestClassifier,
            'GBoost': lambda: GradientBoostingClassifier,
            'KNN': lambda: KNeighborsClassifier,
            'DecisionTree': lambda: DecisionTreeClassifier
      }