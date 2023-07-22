import re
import os
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
import scipy.stats as stats
import json
from common import *
import json
from importData import *

problemNameRecordsMapping={
    "qing.json":"PROBLEM_QING",
    "rastrigin.json":"PROBLEM_RASTRIGIN",
    "rosenbrock.json":"PROBLEM_ROSENBROCK",
    "schwefel223.json": "PROBLEM_SCHWEFEL223",
    "styblinskitang.json": "PROBLEM_STYBLINSKITANG",
    "trid.json":"PROBLEM_TRID"
}

def selectBasedOnRecordsMapping(df,recordPath):
    return pd.DataFrame(df[df['problemName'] == problemNameRecordsMapping[recordPath]])
    
def recordToExperiment(record):
    experiment=record['experiment']
    experiment["problemName"]=experiment["problems"][0]
    experiment["minMedIQR"]=record['metadata']['med_iqr']
    experiment["steps"]=json.dumps(record['metadata']['trials'])
    experiment["trials"]=json.dumps(record['metadata']['trials'])
    experiment["baseLevelEvals"]=experiment["baselevelIterations"]
    experiment["baseLevel-xDim"]=experiment["modelSize"]
    experiment["trialStepCount"]=experiment["hhsteps"]
    experiment["trialCount"]=experiment["hhsteps"]
    experiment["hyperLevel-id"]=experiment["optimizer"]
    (minAvg,minStd,samples)=minTrialAverage(record['metadata']['trials'])
    experiment["minAvg"]=minAvg,
    experiment["minStd"]=minStd
    experiment["samples"]=json.dumps(samples)
    experiment['elapsedTimeSec']=record['metadata']["elapsedTimeSec"]
    experiment.pop("problems")
    experiment.pop("hhsteps")
    experiment.pop("optimizer")
    return experiment

def expandTrials(df):
    df['trials']=df['trials'].map(lambda trials: json.loads(trials))
    return df