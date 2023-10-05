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
    "trid.json":"PROBLEM_TRID",
    "michalewicz.json":"PROBLEM_MICHALEWICZ",
    "dixonprice.json":"PROBLEM_DIXONPRICE",
    "levy.json":"PROBLEM_LEVY",
    "schwefel.json":"PROBLEM_SCHWEFEL",
    "sumsquares.json":"PROBLEM_SUMSQUARES",
    "sphere.json":"PROBLEM_SPHERE"
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


def mergeExperiments(datapath):
    # Initialize an empty list to store the merged experiments
    merged_experiments = {}

    # Loop through each file in the directory
    for filename in os.listdir(datapath):
        if filename!='records_concat.json' and '.json' in filename:
            # Read the content of the JSON file
            with open(os.path.join(datapath, filename), 'r') as file:
                data = json.load(file)
                experiments = data.get("experiments", {})
                # Append the experiments to the merged list
                if experiments != {}:
                    merged_experiments.update(experiments)

    # Write the merged experiments to records.json
    with open(f"{datapath}/records_concat.json", 'w') as output_file:
        json.dump({"experiments": merged_experiments}, output_file, indent=4)


def expandTrials(df):
    df['trials']=df['trials'].map(lambda trials: json.loads(trials))
    return df