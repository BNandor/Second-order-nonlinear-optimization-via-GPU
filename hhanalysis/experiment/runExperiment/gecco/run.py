import sys
import os

sys.path.insert(0, '../')
import numpy as np
import json
import subprocess
import itertools
import os
import copy
from timeit import default_timer as timer
from analysis.common import *
from commonRun import *
import pandas as pd

backslash="\\"
dquote='"'
DEFAULT_THREAD_COUNT=128

# need this, in order to save the samples
def runNMHH2(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/SA-NMHH/GA_DE_GD_LBFGS"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def testScalability(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/scalabilityTests"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([128,256,512],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([10],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["PERTURB","REFINE"],"hyperLevelMethod")
    
    
    variations=list(itertools.product(*list(params.values())))
    # runExperimentVariations(variations,lambda exp:f"{hashOfExperiment(exp)}-threads1",SCALABILITY_EXPERIMENT_RECORDS_PATH,1)
    runExperimentVariations(variations,lambda exp:f"{hashOfExperiment(exp)}-threads{64}",recordspath,64)
    # runExperimentVariations(variations,lambda exp:f"{hashOfExperiment(exp)}-threads{DEFAULT_THREAD_COUNT}",SCALABILITY_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)
    runExperimentVariations(variations,lambda exp:f"{hashOfExperiment(exp)}-threads{256}",recordspath,256)

def runTemperatureAnalysis(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100,5000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([50,100,200],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([1000,10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([5,50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runRandomHHControlGroupExperiments(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/randomHH"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM"],"hyperLevelMethod")
    
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

# runAllExperiments()
# testScalability()
# runTemperatureAnalysis()
# runRandomHHControlGroupExperiments()