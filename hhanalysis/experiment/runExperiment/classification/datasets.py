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

def read_from_csv(file_path,scale=False):
    # Read the CSV file using pandas, specifying that the first row is a header
    df = pd.read_csv(file_path, header=0).sample(frac=1.0)
    
    # Separate features (X) and classes (Y)
    X = df.iloc[:, :-1].values  # All columns except the last one
    Y = df.iloc[:, -1].values   # Only the last column
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    # return X,Y
    return Dataset(X[0:1000], Y[0:1000])

def read_and_map_csv(file_path,sep):
    # Read the CSV file using pandas, specifying that the first row is a header
    df = pd.read_csv(file_path,sep=sep, header=0).sample(frac=1.0)
    df_filled = df.fillna(0)
    categorical_columns = []

    # Identify categorical columns (object/string type)
    for col in df_filled.columns:
        if df_filled[col].dtype == 'object':
            categorical_columns.append(col)
            # print(f"\nCategorical column '{col}' unique values: {df_filled[col].unique()}")
    label_encoders = {}
    df_encoded = df_filled.copy()

    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        # print(f"\nEncoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Separate features (X) and classes (Y)
    X = df_encoded.iloc[:, :-1].values  # All columns except the last one
    Y = df_encoded.iloc[:, -1].values   # Only the last column
    
    return Dataset(X, Y)
    # return X,Y

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
        'GallStone': lambda :read_from_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/gallstone.csv'),
        'HigherEducation': lambda :read_from_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/education.csv'),
        'MouseProtein':lambda :read_and_map_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/mouse_protein.csv',sep=";"),
        'WholesaleCustomer': lambda :read_from_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/wholesale_customer.csv'),
        'HeartFailure': lambda :read_from_csv(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/heart_failure.csv',scale=True)
        # (boston.data, boston.target, 'Boston'),
        # (covtype.data[:100], covtype.target[:100], 'Covtype')
    }

# X,Y=getDatasets()["HeartFailure"]()
# print(X[0])