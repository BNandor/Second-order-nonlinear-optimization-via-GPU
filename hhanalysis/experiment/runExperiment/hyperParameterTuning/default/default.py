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
from sklearn.model_selection import cross_val_score

from runExperiment.commonRun import *
from runExperiment.classification.classifiers import getClassifiers
from runExperiment.classification.datasets import getDatasets
import ipynb.fs.full.runExperiment.hyperParameterTuning.pyNMHH.classification.classify as classify

def default(config):
    clf =config['classifier'](random_state=0)
    clf.fit(config['X'], config['Y'])
    scores = cross_val_score(clf, config['X'], config['Y'], cv=config['crossValidations'],scoring='accuracy')
    return {"bestAccuracy":scores.mean(),'solution':{}}

def defaultTunings(config):
    solutions=[]
    experimentTimes=[]
    for i in range(config['classificationTuningCount']):
        print(f'            >>>Running  iteration {i+1}/{config["classificationTuningCount"]}\n')
        start = timer()
        solutions.append(default(config))
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

def defaultClassificationExperiment(experiment,recordsPath,experimentId):
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
            solutions=defaultTunings(config)
            end = timer()
            metadata={"elapsedTimeSec":end-start}            

            recordExperiment(toPersist(config,experiment,solutions),experimentId,recordsPath,metadata)
            return metadata
