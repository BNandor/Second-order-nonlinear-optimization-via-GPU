from sklearn import datasets
import pandas as pd
import numpy as np
import os

class Dataset:
    def __init__(self, X,Y):
        self.data = X
        self.target = Y

def read_banknotes(file_path):
    # Read the CSV file using pandas, specifying that the first row is a header
    df = pd.read_csv(file_path, header=0).sample(frac=1.0)
    
    # Separate features (X) and classes (Y)
    X = df.iloc[:, :-1].values  # All columns except the last one
    Y = df.iloc[:, -1].values   # Only the last column
    
    return Dataset(X[0:1000], Y[0:1000])


def getDatasets():
    # digits = datasets.load_digits()
    # iris = datasets.load_iris()
    # # diabetes = datasets.load_diabetes()
    # breastCancer = datasets.load_breast_cancer()
    # wine = datasets.load_wine()
    # boston = load_boston()
    # covtype = fetch_covtype()
    return {
        'Digits': datasets.load_digits,
        # (iris.data, iris.target, 'Iris'),
        # 'BreastCancer': datasets.load_breast_cancer,
        # 'Wine':datasets.load_wine,
        'Banknotes': lambda :read_banknotes(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/BankNote_Authentication.csv')
        # (boston.data, boston.target, 'Boston'),
        # (covtype.data[:100], covtype.target[:100], 'Covtype')
    }