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
from runExperiment.commonRun import *
import pandas as pd

backslash="\\"
dquote='"'
DEFAULT_THREAD_COUNT=128


def runRandomHHControlGroupExperiments(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/randomHH/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM"],"hyperLevelMethod")
    
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runNMHH2(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-NMHH/GA_DE_GD_LBFGS/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runCMAESExperiments(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/CMA-ES/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runSA_CMAESExperiments(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-CMA-ES-NMHH/GA_DE_GD_LBFGS/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runMADSExperiments(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/MADS-NMHH/GA_DE_GD_LBFGS/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

# def runSASNLPExperiment(logsPathFromRoot,root,config):
#     logspath=f"{logsPathFromRoot}/CMA-ES/{config['name']}"
#     recordspath=f"{root}/{logspath}/records.json"
#     n=27
#     params={}
#     params["problems"]=zipWithProperty([("PROBLEM_SNLP","hhanalysis/logs/SNLP/SA/snlp.json")],"problems")
#     params["baselevelIterations"]=zipWithProperty([50000],"baselevelIterations")
#     params["populationSize"]=zipWithProperty([20],"populationSize")
#     params["modelSize"]=zipWithProperty([n*2],"modelSize")
#     params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
#     params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
#     params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
#     params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
#     constants1=len(pd.read_csv("../data/snlp/exported.snlp",header=None,delimiter=' ')[0])
#     constants2=len(pd.read_csv("../data/snlp/exported.snlpa",header=None,delimiter=' ')[0])
#     pathFlags=[f"-DPROBLEM_PATH={backslash}{dquote}hhanalysis/data/snlp/exported.snlp{backslash}{dquote} \
#                  -DPROBLEM_ANCHOR_PATH={backslash}{dquote}hhanalysis/data/snlp/exported.snlpa{backslash}{dquote} \
#                  -DPROBLEM_INPUT_POPULATION_PATH={backslash}{dquote}hhanalysis/data/snlp/random227-400-20.pop{backslash}{dquote} \
#                  -DRESIDUAL_CONSTANTS_COUNT_1={constants1} \
#                  -DRESIDUAL_CONSTANTS_COUNT_2={constants2}"]
#     params["additionalFlags"]=zipWithProperty(pathFlags,"additionalFlags")
#     params["hyperLevelMethod"]=zipWithProperty(["SA"],"hyperLevelMethod")
#     variations=list(itertools.product(*list(params.values())))
#     runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runSA_NMHH_GA_DE_GD_LBFGS_GWO(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-NMHH/GWO/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}

    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runMADS_NMHH_GA_DE_GD_LBFGS_GWO(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/MADS-NMHH/GA_DE_GD_LBFGS_GWO/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"

    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runSA_MADS_NMHH_GA_DE_GD_LBFGS(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-MADS-NMHH/GA_DE_GD_LBFGS/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)
    
def runbigSA_MADS_NMHH_GA_DE_GD_LBFGS(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-MADS-NMHH/GA_DE_GD_LBFGS/bigSA/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runbigSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO/bigSA/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66 -DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runCMAES_GA_DE_GD_LBFGS_GWOExperiments(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/CMA-ES/GWO/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runSA_CMAES_ES_GA_DE_GD_LBFGS_GWO_Experiments(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-CMA-ES-NMHH/GWO/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-CMA-ES-NMHH/GA_DE_GD_LBFGS/bigSA/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments_GWO(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-CMA-ES-NMHH/GWO/bigSA/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66 -DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)

def runClusterinProblems(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-CMA-ES-NMHH/GWO/bigSA/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    # datapath=f"hhanalysis/experiment/dataset/clustering/CMC/cmc.txt"
    # datapath=f"hhanalysis/experiment/dataset/clustering/glass/glass.txt"
    datapath=f"hhanalysis/experiment/dataset/clustering/iris/iris.txt"
    # datapath=f"hhanalysis/experiment/dataset/clustering/wine/wine.txt"
    # datapath=f"hhanalysis/experiment/dataset/test/test.txt"
    with open(f"{root}/{datapath}", 'r') as dataset:
            numbers = dataset.read().split()
            samples=int(numbers[0])
            dimension=int(numbers[1])
            clusters=int(numbers[2])

    params={}
    params["problems"]=zipWithProperty([("PROBLEM_CLUSTERING",f"{logspath}/clustering.json")],"problems")
    params["baselevelIterations"]=zipWithProperty([1000],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty([clusters*dimension],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([20],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66 \
                -DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote} \
                -DPROBLEM_PATH={backslash}{dquote}{datapath}{backslash}{dquote} \
                -DPROBLEM_CLUSTERING_POINT_DIM={dimension} \
                -DPROBLEM_CLUSTERING_K={clusters}"]
    # flags=[f"-DPROBLEM_PATH={backslash}{dquote}{datapath}{backslash}{dquote} "]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT,False)

def runSPRTTestNMHH(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SA-NMHH/GA_DE_GD_LBFGS/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"   -DSAMPLING={backslash}{dquote}SPRT-T-test{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT,False)