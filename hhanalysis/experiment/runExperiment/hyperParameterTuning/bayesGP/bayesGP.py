import sys
import os

sys.path.insert(0, '../')
sys.path.insert(0, '../..')

import numpy as np
import json
import subprocess
import itertools
import os
import copy
from timeit import default_timer as timer
import pandas as pd

from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize

from runExperiment.commonRun import *
from runExperiment.classification.classifiers import getClassifiers
from runExperiment.classification.datasets import getDatasets
import ipynb.fs.full.runExperiment.hyperParameterTuning.pyNMHH.classification.classify as classify


def pyNMHHHyperParametersToGP(paramConfig):
    gpParamconfig=[]
    for (key,value) in paramConfig.items():
            if isinstance(value[0], int) and not isinstance(value[0], bool):
                gpParamconfig.append(Integer(value[0],value[1],name=key))
            elif isinstance(value[0], float) and not isinstance(value[0], bool):
                gpParamconfig.append(Real(value[0],value[1],name=key))
            else:
                converted=[]
                for category in value:
                    if isinstance(category,np.str_):
                          converted.append(str(category))
                    else:
                        converted.append(category)
                gpParamconfig.append(Categorical(converted,name=key))
    return gpParamconfig

def unflatten(flatParams,paramConfig):
    unflattened=dict()
    for (flatValue,(key,value)) in zip(flatParams,paramConfig.items()):
            if isinstance(value[0], int):
                unflattened[key]=int(flatValue)
            elif isinstance(value[0], float):
                unflattened[key]=float(flatValue)
            else:
                unflattened[key]=flatValue
    return unflattened
def castToRightString(params):
    # First, let's convert any numpy string to Python string
    for key, value in params.items():
        if isinstance(value, np.str_):
            params[key] = str(value)
        elif isinstance(value, np.generic):  # This covers other numpy types like np.int64, np.float64, etc.
            params[key] = value.item()
    return params

def bayesGPTuning(config):      
    gpParams=pyNMHHHyperParametersToGP((config['hyperParameters']))
    @use_named_args(gpParams)
    def objective(**params):
        clf = config['classifier'](**castToRightString(params))
        scores = cross_val_score(clf, config['X'], config['Y'],cv=config['crossValidations'],scoring='accuracy',n_jobs=-1)
        return -np.mean(scores)
    res_gp = gp_minimize(objective, gpParams, n_calls=config['bayesGPIterations'], random_state=0,verbose=True,n_jobs=-1)
    print("Accuracy:%.4f" % -res_gp.fun)
    print(res_gp.x)
    return {
                "bestAccuracy":-res_gp.fun,
                'solution':json.loads(json.dumps(unflatten(res_gp.x,config['hyperParameters']))),
            }

def bayesGPTunings(config):
    solutions=[]
    experimentTimes=[]
    for i in range(config['classificationTuningCount']):
        print(f'            >>>Running  iteration {i+1}/{config["classificationTuningCount"]}\n')
        start = timer()
        solutions.append(bayesGPTuning(config))
        end = timer()
        elapsed=end-start
        experimentTimes.append(elapsed)
        etaSeconds=(config['classificationTuningCount']-(i+1))*np.mean(experimentTimes)
        print(f'            >>>Ran  {i+1}/{config["classificationTuningCount"]} iterations elapsed seconds {elapsed} eta: {etaSeconds} seconds')
    return solutions

def toPersist(config,experiment,solutions):
        persistedData=copy.copy(config)
        persistedData['hyperParameters']=experiment['classifier']['hyperparameters']
        persistedData['classifierModel']=experiment['classifier']['model'] 
        persistedData['solutions']=json.loads(json.dumps(solutions))
        persistedData['totalFunctionEvaluations']=config['bayesGPIterations']
        persistedData['solver']=experiment['solver']
        persistedData.pop('X')
        persistedData.pop('classifier')
        persistedData.pop('Y')
        return persistedData

def bayesGPClassificationExperiment(experiment,recordsPath,experimentId):
            print(f"        >>>Running experiment {experimentId}")
            start = timer()
            dataset=getDatasets()[experiment['problems']['name']]()
            config={
                        'X':dataset.data,
                        'Y':dataset.target,
                        'hyperParameters':experiment['classifier']['hyperparameters'],
                        'crossValidations':3,
                        'bayesGPIterations': experiment['solutionConfigs']['iterations'],
                        'classifier':getClassifiers()[experiment['classifier']['model']](),
                        'classificationTuningCount':experiment['classificationTuningCount'],
                        'classifierName':experiment['classifier']['name'],
                        'datasetName':experiment['problems']['name']
                    }
            solutions=bayesGPTunings(config)
            end = timer()
            metadata={"elapsedTimeSec":end-start}            

            recordExperiment(toPersist(config,experiment,solutions),experimentId,recordsPath,metadata)
            return metadata
