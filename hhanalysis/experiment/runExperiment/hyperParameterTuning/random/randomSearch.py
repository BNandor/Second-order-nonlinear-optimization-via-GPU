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
import pandas as pd
from timeit import default_timer as timer

from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score,RandomizedSearchCV

from runExperiment.commonRun import *
from runExperiment.classification.classifiers import getClassifiers
from runExperiment.classification.datasets import getDatasets
import ipynb.fs.full.runExperiment.hyperParameterTuning.pyNMHH.classification.classify as classify

def pyNMHHHyperParametersToSP(paramConfig):
    spParamConfig={}
    for (key,value) in paramConfig.items():
            if isinstance(value[0], int) and not isinstance(value[0], bool):
                spParamConfig[key]=sp_randint(value[0],value[1])
            elif isinstance(value[0], float) and not isinstance(value[0], bool):
                spParamConfig[key]=uniform(loc=value[0],scale=value[1]-value[0])
            else:
                converted=[]
                for category in value:
                    if isinstance(category,np.str_):
                          converted.append(str(category))
                    else:
                        converted.append(category)
                spParamConfig[key]=converted
    return spParamConfig

def randomSearch(config):
    rf_params = pyNMHHHyperParametersToSP((config['hyperParameters']))
    
    # if config['classifierName'] == "SVM":
    #     rf_params['max_iter']=sp_randint(1000,1001)

    clf = config['classifier'](random_state=0)
    Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=config['randomSearchIterations'],cv=config['crossValidations'],scoring='accuracy',verbose=2,n_jobs=-1)
    Random.fit(config['X'], config['Y'])
    return {"bestAccuracy":Random.best_score_,'solution':json.loads(json.dumps(Random.best_params_))}

def randomSearchTunings(config):
    solutions=[]
    experimentTimes=[]
    for i in range(config['classificationTuningCount']):
        print(f'            >>>Running  iteration {i+1}/{config["classificationTuningCount"]}\n')
        start = timer()
        solutions.append(randomSearch(config))
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
        persistedData['totalFunctionEvaluations']=config['randomSearchIterations']
        persistedData['solver']=experiment['solver']
        persistedData.pop('X')
        persistedData.pop('classifier')
        persistedData.pop('Y')
        return persistedData

def randomSearchClassificationExperiment(experiment,recordsPath,experimentId):
            print(f"        >>>Running experiment {experimentId}")
            start = timer()
            dataset=getDatasets()[experiment['problems']['name']]()
            config={
                        'X':dataset.data,
                        'Y':dataset.target,
                        'hyperParameters':experiment['classifier']['hyperparameters'],
                        'crossValidations':3,
                        'randomSearchIterations': experiment['solutionConfigs']['iterations'],
                        'classifier':getClassifiers()[experiment['classifier']['model']](),
                        'classificationTuningCount':experiment['classificationTuningCount'],
                        'classifierName':experiment['classifier']['name'],
                        'datasetName':experiment['problems']['name']
                    }
            solutions=randomSearchTunings(config)
            end = timer()
            metadata={"elapsedTimeSec":end-start}            

            recordExperiment(toPersist(config,experiment,solutions),experimentId,recordsPath,metadata)
            return metadata
