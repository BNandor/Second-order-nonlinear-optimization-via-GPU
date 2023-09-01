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
ROOT="../../"
LOGS_ROOT="../logs"

EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/records.json"
SA_GA_DE_GD_LBFGS_RECORDS_PATH=f"{LOGS_ROOT}/SA-NMHH/GA_DE_GD_LBFGS/records.json"
SANMHH_MANY_HYPERSTEPS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-NMHH/manyHyperSteps/records.json"
SANMHH_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-NMHH/GWO/records.json"
RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/randomHH/records.json"
SAPERTURB_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SAPerturb/records.json"
SAPERTURBGWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SAPerturb/GWO/records.json"
SAPERTURBMULTIOPERATORS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SAPerturb/MultiOperators/records.json"
GA_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/GA/records.json"
DE_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/DE/records.json"
RANDOM_GA_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/RANDOM-GA/records.json"
RANDOM_DE_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/RANDOM-DE/records.json"
SAREFINE_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SARefine/records.json"
GD_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/GD/records.json"
LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/LBFGS/records.json"
RANDOM_GD_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/RANDOM-GD/records.json"
RANDOM_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/RANDOM-LBFGS/records.json"
CMA_ES_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/CMA-ES/records.json"
CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/CMA-ES/GWO/records.json"
SA_CMA_ES_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/records.json"
SA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/GWO/records.json"
BIGSA_CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/GA_DE_GD_LBFGS/bigSA/records.json"
BIGSA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-CMA-ES-NMHH/GWO/bigSA/records.json"
MADS_NMHH_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/MADS-NMHH/GA_DE_GD_LBFGS/records.json"
MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/MADS-NMHH/GA_DE_GD_LBFGS_GWO/records.json"
SA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO/records.json"
SA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS/records.json"
BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS/bigSA/records.json"
BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO/bigSA/records.json"
SASNLP_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/SNLP/SA/records.json"
SCALABILITY_EXPERIMENT_RECORDS_PATH=f"{LOGS_ROOT}/scalabilityTests/records.json"

DEFAULT_THREAD_COUNT=128

def runAllExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100,5000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

# need this, in order to save the samples
def runNMHH2():
    logspath="hhanalysis/logs/SA-NMHH/GA_DE_GD_LBFGS"
    recordspath=f"{logspath.replace('hhanalysis/logs',LOGS_ROOT)}/records.json"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
            ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
              ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SA_GA_DE_GD_LBFGS_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def testScalability():
    params={}
    params["problems"]=zipWithProperty([("PROBLEM_ROSENBROCK","hhanalysis/logs/scalabilityTests/rosenbrock.json")],"problems")
    
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
    runExperimentVariations(variations,lambda exp:f"{hashOfExperiment(exp)}-threads{64}",SCALABILITY_EXPERIMENT_RECORDS_PATH,64)
    # runExperimentVariations(variations,lambda exp:f"{hashOfExperiment(exp)}-threads{DEFAULT_THREAD_COUNT}",SCALABILITY_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)
    runExperimentVariations(variations,lambda exp:f"{hashOfExperiment(exp)}-threads{256}",SCALABILITY_EXPERIMENT_RECORDS_PATH,256)

def runTemperatureAnalysis():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/rosenbrock.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100,5000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([50,100,200],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([1000,10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([5,50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runRandomHHControlGroupExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/randomHH/rosenbrock.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/randomHH/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/randomHH/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM"],"hyperLevelMethod")
    
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSAPerturbExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SAPerturb/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SAPerturb/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SAPerturb/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/SAPerturb/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SAPerturb/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/SAPerturb/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA_PERTURB"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SAPERTURB_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSAPerturbGWOExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SAPerturb/GWO/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SAPerturb/GWO/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SAPerturb/GWO/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/SAPerturb/GWO/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SAPerturb/GWO/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/SAPerturb/GWO/qing.json")],"problems")
    
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
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SAPERTURBGWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSAPerturbMultiOperatorExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SAPerturb/MultiOperators/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SAPerturb/MultiOperators/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SAPerturb/MultiOperators/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/SAPerturb/MultiOperators/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SAPerturb/MultiOperators/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/SAPerturb/MultiOperators/qing.json")],"problems")
    
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
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SAPERTURBMULTIOPERATORS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSAPerturbExperimentsBig():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SAPerturb/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SAPerturb/rastrigin.json"),
            #   ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SAPerturb/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/SAPerturb/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SAPerturb/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/SAPerturb/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([1000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([10],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA_PERTURB"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SAPERTURB_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runGAExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/GA/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/GA/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/GA/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/GA/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/GA/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/GA/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["GA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),GA_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runGAExperimentsBig():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/GA/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/GA/rastrigin.json"),
              ("PROBLEM_TRID","hhanalysis/logs/GA/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/GA/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/GA/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([1000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([10],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["GA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),GA_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runDEExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/DE/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/DE/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/DE/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/DE/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/DE/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/DE/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["DE"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),DE_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runDEExperimentsBig():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/DE/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/DE/rastrigin.json"),
              ("PROBLEM_TRID","hhanalysis/logs/DE/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/DE/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/DE/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([1000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([10],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["DE"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),DE_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runRandomGAExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/RANDOM-GA/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/RANDOM-GA/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/RANDOM-GA/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/RANDOM-GA/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/RANDOM-GA/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/RANDOM-GA/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM-GA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),RANDOM_GA_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runRandomGAExperimentsBig():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/RANDOM-GA/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/RANDOM-GA/rastrigin.json"),
            #   ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/RANDOM-GA/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/RANDOM-GA/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/RANDOM-GA/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/RANDOM-GA/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([1000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([10],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM-GA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),RANDOM_GA_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runRandomDEExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/RANDOM-DE/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/RANDOM-DE/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/RANDOM-DE/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/RANDOM-DE/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/RANDOM-DE/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/RANDOM-DE/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM-DE"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),RANDOM_DE_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runRandomDEExperimentsBig():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/RANDOM-DE/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/RANDOM-DE/rastrigin.json"),
            #   ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/RANDOM-DE/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/RANDOM-DE/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/RANDOM-DE/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/RANDOM-DE/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([1000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([10],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM-DE"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),RANDOM_DE_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSARefineExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SARefine/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SARefine/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SARefine/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/SARefine/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SARefine/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/SARefine/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA_REFINE"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SAREFINE_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runGDExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/GD/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/GD/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/GD/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/GD/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/GD/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/GD/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["GD"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),GD_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runLBFGSExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/LBFGS/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/LBFGS/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/LBFGS/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/LBFGS/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/LBFGS/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/LBFGS/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["LBFGS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),LBFGS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runRandomGDExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/RANDOM-GD/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/RANDOM-GD/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/RANDOM-GD/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/RANDOM-GD/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/RANDOM-GD/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/RANDOM-GD/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM-GD"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),RANDOM_GD_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runRandomLBFGSExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/RANDOM-LBFGS/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/RANDOM-LBFGS/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/RANDOM-LBFGS/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/RANDOM-LBFGS/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/RANDOM-LBFGS/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/RANDOM-LBFGS/qing.json")],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["RANDOM-LBFGS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),RANDOM_LBFGS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runCMAESExperiments():
    logspath="hhanalysis/logs/CMA-ES"
    params={}
    params["problems"]=zipWithProperty([
            ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
            ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
            ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
            ("PROBLEM_TRID",f"{logspath}/trid.json"),
            ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
            ("PROBLEM_QING",f"{logspath}/qing.json"),
               ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),CMA_ES_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSA_CMAESExperiments():
    logspath="hhanalysis/logs/SA-CMA-ES-NMHH/GA_DE_GD_LBFGS"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
                 ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SA_CMA_ES_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runMADSExperiments():
    logspath="hhanalysis/logs/MADS-NMHH/GA_DE_GD_LBFGS"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
            ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),MADS_NMHH_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSANMHH_ManyHyperheuristicSteps():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SA-NMHH/manyHyperSteps/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SA-NMHH/manyHyperSteps/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SA-NMHH/manyHyperSteps/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/SA-NMHH/manyHyperSteps/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SA-NMHH/manyHyperSteps/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/SA-NMHH/manyHyperSteps/qing.json")
              ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([500],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SANMHH_MANY_HYPERSTEPS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSASNLPExperiment():
    n=27
    params={}
    params["problems"]=zipWithProperty([("PROBLEM_SNLP","hhanalysis/logs/SNLP/SA/snlp.json")],"problems")
    params["baselevelIterations"]=zipWithProperty([50000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([20],"populationSize")
    params["modelSize"]=zipWithProperty([n*2],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    constants1=len(pd.read_csv("../data/snlp/exported.snlp",header=None,delimiter=' ')[0])
    constants2=len(pd.read_csv("../data/snlp/exported.snlpa",header=None,delimiter=' ')[0])
    pathFlags=[f"-DPROBLEM_PATH={backslash}{dquote}hhanalysis/data/snlp/exported.snlp{backslash}{dquote} \
                 -DPROBLEM_ANCHOR_PATH={backslash}{dquote}hhanalysis/data/snlp/exported.snlpa{backslash}{dquote} \
                 -DPROBLEM_INPUT_POPULATION_PATH={backslash}{dquote}hhanalysis/data/snlp/random227-400-20.pop{backslash}{dquote} \
                 -DRESIDUAL_CONSTANTS_COUNT_1={constants1} \
                 -DRESIDUAL_CONSTANTS_COUNT_2={constants2}"]
    params["additionalFlags"]=zipWithProperty(pathFlags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SASNLP_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSA_NMHH_GA_DE_GD_LBFGS_GWO():
    params={}
    logspath="hhanalysis/logs/SA-NMHH/GWO"
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
              ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
              ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SANMHH_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runMADS_NMHH_GA_DE_GD_LBFGS_GWO():
    logspath="hhanalysis/logs/MADS-NMHH/GA_DE_GD_LBFGS_GWO"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
               ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO():
    logspath="hhanalysis/logs/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO"
    params={}
    params["problems"]=zipWithProperty([
            ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
            ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
            ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
            ("PROBLEM_TRID",f"{logspath}/trid.json"),
            ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
            ("PROBLEM_QING",f"{logspath}/qing.json"),
            ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSA_MADS_NMHH_GA_DE_GD_LBFGS():
    logspath="hhanalysis/logs/SA-MADS-NMHH/GA_DE_GD_LBFGS"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
            ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)
    
def runbigSA_MADS_NMHH_GA_DE_GD_LBFGS():
    logspath="hhanalysis/logs/SA-MADS-NMHH/GA_DE_GD_LBFGS/bigSA"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
              ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runbigSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO():
    logspath="hhanalysis/logs/SA-MADS-NMHH/GA_DE_GD_LBFGS_GWO/bigSA"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
                 ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66 -DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-MADS"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),BIGSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runCMAES_GA_DE_GD_LBFGS_GWOExperiments():
    logspath="hhanalysis/logs/CMA-ES/GWO"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
                 ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSA_CMAES_ES_GA_DE_GD_LBFGS_GWO_Experiments():
    logspath="hhanalysis/logs/SA-CMA-ES-NMHH/GWO"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
                 ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments():
    logspath="hhanalysis/logs/SA-CMA-ES-NMHH/GA_DE_GD_LBFGS/bigSA"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
              ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
              ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
              ("PROBLEM_LEVY",f"{logspath}/levy.json"),
              ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
              ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
              ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),BIGSA_CMA_ES_GA_DE_GD_LBFGS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments_GWO():
    logspath="hhanalysis/logs/SA-CMA-ES-NMHH/GWO/bigSA"
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
              ("PROBLEM_TRID",f"{logspath}/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
              ("PROBLEM_QING",f"{logspath}/qing.json"),
                ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
            ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
            ("PROBLEM_LEVY",f"{logspath}/levy.json"),
            ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
            ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
            ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([1,2,3,4,5,6,7,8,9,10,15,30,50,100,500,750],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66 -DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),BIGSA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runClusterinProblems():
    logspath="hhanalysis/logs/SA-CMA-ES-NMHH/GWO/bigSA"
    datapath="hhanalysis/experiment/dataset/clustering/CMC/cmc.txt"
    # datapath="hhanalysis/experiment/dataset/clustering/glass/glass.txt"
    # datapath="hhanalysis/experiment/dataset/clustering/iris/iris.txt"
    # datapath="hhanalysis/experiment/dataset/clustering/wine/wine.txt"
    # datapath="hhanalysis/experiment/dataset/test/test.txt"
    with open(f"{ROOT}/{datapath}", 'r') as dataset:
            numbers = dataset.read().split()
            samples=int(numbers[0])
            dimension=int(numbers[1])
            clusters=int(numbers[2])

    params={}
    params["problems"]=zipWithProperty([("PROBLEM_CLUSTERING",f"{logspath}/clustering.json")],"problems")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([clusters*dimension],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([2],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([100],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    # flags=[f"-DHH_SA_HYBRID_PERCENTAGE=0.66 -DBASE_PERTURB_EXTRA_OPERATORS={backslash}{dquote}GWO{backslash}{dquote} -DPROBLEM_PATH={backslash}{dquote}{datapath}{backslash}{dquote}"]
    flags=[f"-DPROBLEM_PATH={backslash}{dquote}{datapath}{backslash}{dquote}"]
    params["additionalFlags"]=zipWithProperty(flags,"additionalFlags")
    # params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),BIGSA_CMA_ES_GA_DE_GD_LBFGS_GWO_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)
    
# nmhh2,saGWOGroup,madsGWOGroup,saMadsGWOGroup,sacmaesGWOGroup,cmaesGWOGroup
# runAllExperiments()
# testScalability()
# runTemperatureAnalysis()
# runSASNLPExperiment()
# runRandomHHControlGroupExperiments()
# runSAPerturbExperiments()
# runSAPerturbGWOExperiments()
# runSAPerturbMultiOperatorExperiments()
# runSAPerturbExperimentsBig()
# runGAExperiments()
# runGAExperimentsBig()
# runDEExperiments()
# runDEExperimentsBig()
# runRandomGAExperiments()
# runRandomGAExperimentsBig()
# runRandomDEExperiments()
# runRandomDEExperimentsBig()
# runSARefineExperiments()
# runGDExperiments()
# runLBFGSExperiments()
# runRandomGDExperiments()
# runRandomLBFGSExperiments()
# runSANMHH_ManyHyperheuristicSteps()

runNMHH2()
runSA_NMHH_GA_DE_GD_LBFGS_GWO()
runMADS_NMHH_GA_DE_GD_LBFGS_GWO()
runMADSExperiments()
runbigSA_MADS_NMHH_GA_DE_GD_LBFGS()
runbigSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO()
runSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO()
runSA_MADS_NMHH_GA_DE_GD_LBFGS()
runCMAESExperiments()
runCMAES_GA_DE_GD_LBFGS_GWOExperiments()
runSA_CMAESExperiments()
runSA_CMAES_ES_GA_DE_GD_LBFGS_GWO_Experiments()
runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments()
runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments_GWO()

# runClusterinProblems()