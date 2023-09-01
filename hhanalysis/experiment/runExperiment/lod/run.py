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

def runSAPerturbExperiments(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/SAPerturb"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA_PERTURB"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runSAPerturbGWOExperiments(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/SAPerturb/GWO"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json")],"problems")
      
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA_PERTURB"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runSAPerturbMultiOperatorExperiments(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/SAPerturb/MultiOperators"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json")],"problems")
      
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO,DE2,GA2{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA_PERTURB"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runGAExperiments(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/GA"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json")],"problems")
      
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["GA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runDEExperiments(logsPathFromRoot,root):
    logspath=f"{logsPathFromRoot}/DE"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json")],"problems")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["DE"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)