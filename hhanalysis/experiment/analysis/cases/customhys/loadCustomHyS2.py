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


from importData import *

problemNameMapping={
    "QING":"PROBLEM_QING",
    "RASTRIGIN":"PROBLEM_RASTRIGIN",
    "ROSENBROCK":"PROBLEM_ROSENBROCK",
    "SCHWEFEL223": "PROBLEM_SCHWEFEL223",
    "STYBLINSKITANG": "PROBLEM_STYBLINSKITANG",
    "TRID":"PROBLEM_TRID",
    "PROBLEM_MICHALEWICZ":"PROBLEM_MICHALEWICZ",
    "PROBLEM_DIXONPRICE":"PROBLEM_DIXONPRICE",
    "PROBLEM_LEVY":"PROBLEM_LEVY",
    "PROBLEM_SCHWEFEL":"PROBLEM_SCHWEFEL",
    "PROBLEM_SUMSQUARES":"PROBLEM_SUMSQUARES",
    "PROBLEM_SPHERE":"PROBLEM_SPHERE"
}

problemNameRecordsMapping={
    "qing.json":"QING",
    "rastrigin.json":"RASTRIGIN",
    "rosenbrock.json":"ROSENBROCK",
    "schwefel223.json": "SCHWEFEL223",
    "styblinskitang.json": "STYBLINSKITANG",
    "trid.json":"TRID",
    "michalewicz.json":"PROBLEM_MICHALEWICZ",
    "dixonprice.json":"PROBLEM_DIXONPRICE",
    "levy.json":"PROBLEM_LEVY",
    "schwefel.json":"PROBLEM_SCHWEFEL",
    "sumsquares.json":"PROBLEM_SUMSQUARES",
    "sphere.json":"PROBLEM_SPHERE"
}

def selectBasedOnRecordsMapping(df,recordPath):
    return pd.DataFrame(df[df['baseLevel-problemId'] == problemNameRecordsMapping[recordPath]])
    
def customhys2RecordToExperiment(record):
    experiment={}
    experiment["problemName"]=problemNameMapping[record["baseLevel-problemId"]]
    experiment["minMedIQR"]=min(list(map(lambda r: r['med_+_iqr'],record['trials'])))
    experiment["steps"]=json.dumps(list(map(lambda r: r['med_+_iqr'],record['trials'])))
    experiment["trialStepCount"]=record["hhsteps"]
    experiment["baselevelIterations"]=record["baselevelIterations"]
    experiment["populationSize"]=record["populationSize"]
    experiment['modelSize']=record['baseLevel-xDim']
    (minAvg,minStd,samples)=minTrialAverage(record['trials'])
    experiment["minAvg"]=minAvg,
    experiment["minStd"]=minStd
    experiment["samples"]=json.dumps(samples)
    return experiment

def loadCustomHysDF():
    logs = open(CUSTOMHYS2_RESULTS_PATH)
    df=pd.DataFrame(list(json.load(logs)['experiments'].values()))
    df['baseLevelEvals']=df['baselevelIterations']
    df['trialCount']=df['hhsteps']
    return df
    
    