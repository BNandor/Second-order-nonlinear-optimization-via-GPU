from commonAnalysis import *
from dict_hash import sha256
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.model_selection import train_test_split
from commonANN import *
import numpy as np

def recordToChainRegressionExperiment(record):
    experiment={}
    experiment["problemPath"]=record['experiment']["problems"][1]
    experiment["modelSize"]=record['experiment']["modelSize"]
    experiment["baselevelIterations"]=record['experiment']['baselevelIterations']
    experiment["trialStepCount"]=record['experiment']['trialStepCount']
    experiment["HH-SA-temp"]=record['experiment']['HH-SA-temp']
    experiment["HH-SA-alpha"]=record['experiment']['HH-SA-alpha']
    return experiment

def getSimplexParams(simplex,metric):
    bestParameters=metric["bestParameters"]
    return [(simplex+"-"+node,bestParameters[simplex][node]["value"]) for node in bestParameters[simplex]] 
def interestingBestParameters():
    return ["OptimizerChainPerturbatorSimplex",
    "OptimizerChainRefinerSimplex",
    "PerturbatorDESimplex",
    "PerturbatorGASimplex",
    "RefinerGDSimplex",
    "RefinerLBFGSSimplex",
    "RefinerLBFGSOperatorParams","RefinerGDOperatorParams"]
def getChainFeatures(metric):
    features ={
        "hashSHA256":metric["experimentHashSha256"]
    }
    paramsList=[getSimplexParams(bestParameter,metric) for bestParameter in interestingBestParameters()]
    for (param,value) in [paramValue for sublist in paramsList for paramValue in sublist]:
        features[param]=value
    return features

def dontEnrichNoFilter(recordsWithMetrics,aggregations,experimentColumns):
    return aggregations

SA_EXPERIMENT_RECORDS_PATH="../../logs/records.json"
def getChainTrainingData():
    np.set_printoptions(precision=4)
    chainView=createTestGroupView(
                                    recordsPath=SA_EXPERIMENT_RECORDS_PATH,
                                    getMetricsAndId=(getChainFeatures,"hashSHA256"),
                                    mapRecordToExperiment=recordToChainRegressionExperiment,
                                    explanatoryColumns=set(),
                                    responseColumns=set(),
                                    metricsAggregation={},
                                    enrichAndFilter=dontEnrichNoFilter)
    chainView=chainView.reset_index(drop=True)                                
    chainView=chainView.drop(['problemPath'],axis=1)                                    
    print(chainView.to_html())
    return chainView

def regress():
    np.set_printoptions(precision=2,suppress=True)
    chainData=getChainTrainingData()
    chainData=chainData[["modelSize","baselevelIterations","HH-SA-temp","HH-SA-alpha",
                    "OptimizerChainPerturbatorSimplex-refiner","OptimizerChainRefinerSimplex-refiner","RefinerGDOperatorParams-GD_FEVALS","RefinerLBFGSOperatorParams-LBFGS_FEVALS"]]
    train,test=train_test_split(chainData, test_size=0.1)
    # train=chainData#train_test_split(chainData, test_size=0)
    # test=pd.DataFrame({"modelSize":500,"baselevelIterations":[100],"HH-SA-temp":[10000],"HH-SA-alpha":[5],
    #                    "OptimizerChainPerturbatorSimplex-refiner":[0],"OptimizerChainRefinerSimplex-refiner":[0],"RefinerGDOperatorParams-GD_FEVALS":[0],"RefinerLBFGSOperatorParams-LBFGS_FEVALS":[0]})
    featureColumns=["modelSize","baselevelIterations","HH-SA-temp","HH-SA-alpha"]
    targetColumns=set(train.columns).difference(set(featureColumns))

    model,x_test_scaled,y_test=trainANN(train,test,targetColumns,featureColumns)
    predictions=model.predict(x_test_scaled)
    
    print(f"Real: {y_test.values[0:5]}")
    print(f"Predicted: {predictions[0:5]}")
    # print(predictions)

regress()