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
# from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn import svm

from runExperiment.hyperParameterTuning.genetic.customClassifier.CustomRF import *
from runExperiment.hyperParameterTuning.genetic.customClassifier.CustomGradientBoost import *
from runExperiment.hyperParameterTuning.genetic.customClassifier.CustomDT import *
from runExperiment.hyperParameterTuning.genetic.customClassifier.CustomSVC import *

from runExperiment.commonRun import *
from runExperiment.classification.classifiers import getClassifiers
from runExperiment.classification.datasets import getDatasets
import ipynb.fs.full.runExperiment.hyperParameterTuning.pyNMHH.classification.classify as classify

def pyNMHHHyperParametersToRange(paramConfig):
    spParamConfig={}
    for (key,value) in paramConfig.items():
            if isinstance(value[0], int) and not isinstance(value[0], bool):
                spParamConfig[key]=[ int(v) for v in range(value[0],value[1])]
            elif isinstance(value[0], float) and not isinstance(value[0], bool):
                spParamConfig[key]=np.random.uniform(value[0],value[1],1000)
            else:
                converted=[]
                for category in value:
                    if isinstance(category,np.str_):
                          converted.append(str(category))
                    else:
                        converted.append(category)
                spParamConfig[key]=converted
    return spParamConfig

def geneTypes(paramConfig):
    gene_type=[]
    for (key,value) in paramConfig.items():
            if isinstance(value[0], float) and not isinstance(value[0], bool):
                gene_type.append(2)
            else:
                gene_type.append(1)
    return gene_type

def handleCustomClassifiers(clf):
    if isinstance(clf,RandomForestClassifier):
        return CustomRandomForestClassifier()
    if isinstance(clf,GradientBoostingClassifier):
        return CustomGradientBoostClassifier()
    if isinstance(clf,DecisionTreeClassifier):
        return CustomDecisionTreeClassifier()
    if isinstance(clf,svm.SVC):
        return CustomSVC()
    return clf

def geneticSearch(config):
    clf_params = pyNMHHHyperParametersToRange((config['hyperParameters']))
    clf = handleCustomClassifiers(config['classifier'](random_state=0))
    # ga1 = EvolutionaryAlgorithmSearchCV(estimator=clf,
    #                                params=clf_params,
    #                                gene_type=geneTypes(config['hyperParameters']),
    #                                scoring="accuracy",
    #                                cv=config['crossValidations'],
    #                                population_size=config['populationSize'],
    #                                gene_mutation_prob=0.10,
    #                                gene_crossover_prob=0.5,
    #                                tournament_size=3,
    #                                generations_number=config['geneticSearchGenerations'],
    #                                verbose=3,
    #                                n_jobs=10)
    # ga1.fit(config['X'], config['Y'])
    # return {"bestAccuracy":ga1.best_score_,'solution':json.loads(json.dumps(ga1.best_params_))}
    print("genetic hyperparameter tuner requires scikit-learn 0.24.2")
    exit(1)
    return {"bestAccuracy":0.0,'solution':{}}

def geneticSearchTunings(config):
    solutions=[]
    experimentTimes=[]
    for i in range(config['classificationTuningCount']):
        print(f'            >>>Running  iteration {i+1}/{config["classificationTuningCount"]}\n')
        start = timer()
        solutions.append(geneticSearch(config))
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
        persistedData['totalFunctionEvaluations']=config['geneticSearchGenerations']*config['populationSize']
        persistedData['solver']=experiment['solver']
        persistedData.pop('X')
        persistedData.pop('classifier')
        persistedData.pop('Y')
        return persistedData

def geneticSearchClassificationExperiment(experiment,recordsPath,experimentId):
            print(f"        >>>Running experiment {experimentId}")
            start = timer()
            dataset=getDatasets()[experiment['problems']['name']]()
            config={
                        'X':dataset.data,
                        'Y':dataset.target,
                        'hyperParameters':experiment['classifier']['hyperparameters'],
                        'crossValidations':3,
                        'geneticSearchGenerations': experiment['solutionConfigs']['iterations'],
                        'populationSize': experiment['solutionConfigs']['populationSize'],
                        'classifier':getClassifiers()[experiment['classifier']['model']](),
                        'classificationTuningCount':experiment['classificationTuningCount'],
                        'classifierName':experiment['classifier']['name'],
                        'datasetName':experiment['problems']['name']
                    }
            solutions=geneticSearchTunings(config)
            end = timer()
            metadata={"elapsedTimeSec":end-start}            

            recordExperiment(toPersist(config,experiment,solutions),experimentId,recordsPath,metadata)
            return metadata
