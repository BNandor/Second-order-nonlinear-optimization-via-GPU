from sklearn import datasets
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd
import numpy as np
import os

class Dataset:
    def __init__(self, X,Y):
        self.data = X
        self.target = Y

def read_from_csv(file_path):
    # Read the CSV file using pandas, specifying that the first row is a header
    df = pd.read_csv(file_path, header=0).sample(frac=1.0)
    
    # Separate features (X) and classes (Y)
    X = df.iloc[:, :-1].values  # All columns except the last one
    Y = df.iloc[:, -1].values   # Only the last column
    
    return Dataset(X[0:1000], Y[0:1000])

def readCervicalCancer():
    df=pd.read_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/cervical_cancer.csv')
    df=df.replace('?',np.nan)
    df=df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
    df = df.apply(pd.to_numeric)
    df =  df.fillna(df.mean())
    object_data=df.select_dtypes(include=['object'])
    for i in object_data:
        df[i]=df[i].astype(float)
    num_data=df.select_dtypes(exclude=['object'])
    df=pd.concat([num_data,object_data],axis=1)
    X = df.drop(['Biopsy'], axis=1)
    y=df['Biopsy']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return Dataset(X, y.to_numpy().ravel())

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
        'Iris': datasets.load_iris,
        'BreastCancer': datasets.load_breast_cancer,
        'Wine':datasets.load_wine,
        'Banknotes': lambda :read_from_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/BankNote_Authentication.csv'),
        'AuditRisk': lambda :read_from_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/audit_risk.csv'),
        'CervicalCancer': readCervicalCancer,
        # (boston.data, boston.target, 'Boston'),
        # (covtype.data[:100], covtype.target[:100], 'Covtype')
    }