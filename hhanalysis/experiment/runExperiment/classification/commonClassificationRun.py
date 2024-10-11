import numpy as np
import json
import subprocess
import itertools
import os
import copy
from timeit import default_timer as timer
from analysis.common import *
import pandas as pd
from runExperiment.commonRun import *
from runExperiment.hyperParameterTuning.pyNMHH.classification.run import *
from runExperiment.hyperParameterTuning.bayesGP.bayesGP import *

backslash="\\"
dquote='"'



def runClassificationExperimentVariations(experimentVariations,experimentIdMapper,experimenter,recordsPath):
    remainingExperimentsToRun=sum([not experimented(experimentIdMapper(mapExperimentListToDict(experiment)),recordsPath) for experiment in experimentVariations])
    print(f">Total tuning experiments: {len(experimentVariations)}")
    print(f">Remaining tuning experiments: {remainingExperimentsToRun}")
    runningId=0
    experimentTimes=[]
    for experiment in experimentVariations:
        experimentDict=mapExperimentListToDict(experiment=experiment)
        experimentId=experimentIdMapper(experimentDict)
        if not experimented(experimentId,recordsPath):
            print(f"    >>Running tuning experiments{runningId+1}/{remainingExperimentsToRun}: {experiment}")
            metadata=experimenter(experiment=experimentDict,recordsPath=recordsPath,experimentId=experimentId)
            experimentTimes.append(metadata["elapsedTimeSec"])
            etaSeconds=(remainingExperimentsToRun-(runningId+1))*np.mean(experimentTimes)
            print(f'    >>Ran {runningId+1}/{remainingExperimentsToRun}: {experiment} for: {metadata["elapsedTimeSec"]} ETA: {etaSeconds} seconds ')
            runningId+=1
        else:
            print(f"Skipping {experiment}")

def runClassificationExperiments(logsPathFromRoot,root,config,solver):
    for classifier in config['classifiers']:
        print(f'>Running experiments for {classifier}')
        logspath=f"{logsPathFromRoot}/{config['solver']}/classification/{classifier['name']}/{config['name']}"
        recordspath=f"{root}/{logspath}/records.json"
        params={}
        params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
        params["solutionConfigs"]=zipWithProperty(config['solutionConfigs'],"solutionConfigs")
        params["classificationTuningCount"]=zipWithProperty([config['classificationTuningCount']],"classificationTuningCount")
        params["classifier"]=zipWithProperty([classifier],"classifier")
        params["baseLevelConfigs"]=zipWithProperty(config['baseLevelConfigs'],"baseLevelConfigs")
        params["solver"]=zipWithProperty([config['solver']],"solver")
        
        variations=list(itertools.product(*list(params.values())))
        runClassificationExperimentVariations(variations,lambda exp:hashOfExperiment(exp),solver,recordspath)