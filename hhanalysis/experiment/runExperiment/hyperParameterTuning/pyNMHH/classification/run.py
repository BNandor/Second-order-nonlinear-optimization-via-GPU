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

from runExperiment.classification.classifiers import getClassifiers
from runExperiment.classification.datasets import getDatasets
from runExperiment.commonRun import recordExperiment

import ipynb.fs.full.runExperiment.hyperParameterTuning.pyNMHH.classification.classify as classify

backslash="\\"
dquote='"'

def toPersist(config,experiment,solutions):
        persistedData=copy.copy(config)
        persistedData['hyperParameters']=experiment['classifier']['hyperparameters']
        persistedData['classifierModel']=experiment['classifier']['model'] 
        persistedData['totalFunctionEvaluations']=config['baselevelIterations']*config['pyNMHHSteps']
        persistedData['solutions']=json.loads(json.dumps(solutions))
        persistedData['solver']=experiment['solver']
        persistedData.pop('X')
        persistedData.pop('classifier')
        persistedData.pop('flatHyperParameters')
        persistedData.pop('Y')
        return persistedData

def pyNMHHClassificationExperiment(experiment,recordsPath,experimentId):
            print(f"        >>>Running experiment {experimentId}")
            start = timer()
            dataset=getDatasets()[experiment['problems']['name']]()
            config={
                        'X':dataset.data,
                        'Y':dataset.target,
                        'flatHyperParameters':classify.FlatHyperParameters(experiment['classifier']['hyperparameters']),
                        'populationSize':experiment['solutionConfigs']['populationSize'],
                        'baselevelIterations':experiment['solutionConfigs']['baselevelIterations'],
                        'crossValidations':3,
                        'pyNMHHSteps': experiment['solutionConfigs']['pyNMHHSteps'],
                        'classifier':getClassifiers()[experiment['classifier']['model']](),
                        'classificationTuningCount':experiment['classificationTuningCount'],
                        'classifierName':experiment['classifier']['name'],
                        'datasetName':experiment['problems']['name'],
                        'trainingFraction':0.75,
                        'baseLevelConfig':experiment['baseLevelConfigs'],
                        'SA_temp0':1.0,
                        'SA_coolingRate':0.995
                    }
            solutions=classify.runClassificationTuningExperiment(config)
            end = timer()
            metadata={"elapsedTimeSec":end-start}            

            recordExperiment(toPersist(config,experiment,solutions),experimentId,recordsPath,metadata)
            return metadata
