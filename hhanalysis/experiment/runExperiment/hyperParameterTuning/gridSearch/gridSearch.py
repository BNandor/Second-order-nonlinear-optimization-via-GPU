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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from runExperiment.commonRun import *
from runExperiment.classification.classifiers import getClassifiers
from runExperiment.classification.datasets import getDatasets
import ipynb.fs.full.runExperiment.hyperParameterTuning.pyNMHH.classification.classify as classify

def pyNMHHHyperParametersToRange(paramConfig,iterations):
    rangeParams=0
    listParams=1
    for (key,value) in paramConfig.items():
        if (isinstance(value[0], float) or isinstance(value[0], int) )and not isinstance(value[0], bool):
            rangeParams=rangeParams+1
        else:
            listParams=listParams*len(value)

    rangeIterations=(iterations/listParams)**(1/rangeParams)
    spParamConfig={}
    for (key,value) in paramConfig.items():
            if isinstance(value[0], int) and not isinstance(value[0], bool):
                spParamConfig[key]=[ int(v) for v in range(value[0],value[1],int((value[1]-value[0])/rangeIterations))]
            elif isinstance(value[0], float) and not isinstance(value[0], bool):
                spParamConfig[key]=np.random.uniform(value[0],value[1],int(rangeIterations))
            else:
                converted=[]
                for category in value:
                    if isinstance(category,np.str_):
                          converted.append(str(category))
                    else:
                        converted.append(category)
                spParamConfig[key]=converted
    return spParamConfig

def gridSearch(config):
    clf_params = pyNMHHHyperParametersToRange((config['hyperParameters']),config['iterations'])
    if config['classifierName'] == "SVM":
        clf_params['max_iter']=[1000]
    clf =config['classifier'](random_state=0)
    grid = GridSearchCV(clf, clf_params, cv=config['crossValidations'], scoring='accuracy',n_jobs=-1,verbose=3)
    grid.fit(config['X'], config['Y'])
    return {"bestAccuracy":grid.best_score_,'solution':json.loads(json.dumps(grid.best_params_))}

def gridSearchTunings(config):
    solutions=[]
    experimentTimes=[]
    for i in range(config['classificationTuningCount']):
        print(f'            >>>Running  iteration {i+1}/{config["classificationTuningCount"]}\n')
        start = timer()
        solutions.append(gridSearch(config))
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
        persistedData['totalFunctionEvaluations']=config['iterations']
        persistedData['solver']=experiment['solver']
        persistedData.pop('X')
        persistedData.pop('classifier')
        persistedData.pop('Y')
        return persistedData

def gridSearchClassificationExperiment(experiment,recordsPath,experimentId):
            print(f"        >>>Running experiment {experimentId}")
            start = timer()
            dataset=getDatasets()[experiment['problems']['name']]()
            config={
                        'X':dataset.data,
                        'Y':dataset.target,
                        'hyperParameters':experiment['classifier']['hyperparameters'],
                        'crossValidations':3,
                        'classifier':getClassifiers()[experiment['classifier']['model']](),
                        'iterations':experiment['solutionConfigs']['iterations'],
                        'classificationTuningCount':experiment['classificationTuningCount'],
                        'classifierName':experiment['classifier']['name'],
                        'datasetName':experiment['problems']['name']
                    }
            solutions=gridSearchTunings(config)
            end = timer()
            metadata={"elapsedTimeSec":end-start}            

            recordExperiment(toPersist(config,experiment,solutions),experimentId,recordsPath,metadata)
            return metadata
