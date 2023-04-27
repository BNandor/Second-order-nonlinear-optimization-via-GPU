import numpy as np
import json
import subprocess
import itertools
import os
import copy
from timeit import default_timer as timer
from analysis.common import *
import pandas as pd


backslash="\\"
dquote='"'
EXPERIMENT_RECORDS_PATH="../logs/records.json"
SANMHH_MANY_HYPERSTEPS_EXPERIMENT_RECORDS_PATH="../logs/SA-NMHH/manyHyperSteps/records.json"
SANMHH_FULL_PARAMETERS_MANY_HYPERSTEPS_EXPERIMENT_RECORDS_PATH="../logs/SA-NMHH/manyHyperSteps/fullparameterbounds/records.json"
RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH="../logs/randomHH/records.json"
SAPERTURB_EXPERIMENT_RECORDS_PATH="../logs/SAPerturb/records.json"
GA_EXPERIMENT_RECORDS_PATH="../logs/GA/records.json"
DE_EXPERIMENT_RECORDS_PATH="../logs/DE/records.json"
RANDOM_GA_EXPERIMENT_RECORDS_PATH="../logs/RANDOM-GA/records.json"
RANDOM_DE_EXPERIMENT_RECORDS_PATH="../logs/RANDOM-DE/records.json"
SAREFINE_EXPERIMENT_RECORDS_PATH="../logs/SARefine/records.json"
GD_EXPERIMENT_RECORDS_PATH="../logs/GD/records.json"
LBFGS_EXPERIMENT_RECORDS_PATH="../logs/LBFGS/records.json"
RANDOM_GD_EXPERIMENT_RECORDS_PATH="../logs/RANDOM-GD/records.json"
RANDOM_LBFGS_EXPERIMENT_RECORDS_PATH="../logs/RANDOM-LBFGS/records.json"
CMA_ES_EXPERIMENT_RECORDS_PATH="../logs/CMA-ES/records.json"
SA_CMA_ES_EXPERIMENT_RECORDS_PATH="../logs/SA-CMA-ES-NMHH/records.json"
SA_CMA_ES_FULL_PARAMETERS_EXPERIMENT_RECORDS_PATH="../logs/SA-CMA-ES-NMHH/fullparameterbounds/records.json"
SASNLP_EXPERIMENT_RECORDS_PATH="../logs/SNLP/SA/records.json"
SCALABILITY_EXPERIMENT_RECORDS_PATH="../logs/scalabilityTests/records.json"

DEFAULT_THREAD_COUNT=128


def runOptimizerWith(flags):
    process = subprocess.Popen(['make', 'buildAndRunExperimentWith', f'NVCCFLAGS={flags}'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    return process.stdout

def consumeOutput(stdout,consumeline):
    logs=[]
    for linebuffer in stdout:
        line=linebuffer.decode("utf-8")
        logs.append(line)
        consumeline(line)
    return logs

def writeJsonToFile(path,content):

    outfile = open(path, "w")
    [outfile.write(content)]
    outfile.close()

def createIfNotExists(path,content):
    if not os.path.exists(path):
        print(f"Creating {path} with {content}")
        writeJsonToFile(path=path,content=content)

def emptyExperimentRecords():
    return {"experiments":{}}

def experimented(experimentId,experimentRecordsPath):
    createIfNotExists(experimentRecordsPath,json.dumps(emptyExperimentRecords(), indent = 4))
    currentRecords=json.load(open(experimentRecordsPath,'r'))
    return experimentId in currentRecords["experiments"].keys() 

def recordExperiment(experiment,experimentId,experimentRecordsPath,metadata):
    createIfNotExists(experimentRecordsPath,json.dumps(emptyExperimentRecords(), indent = 4))
    currentRecords=json.load(open(experimentRecordsPath,'r'))
    currentRecords["experiments"][experimentId]={"experiment":experiment,"metadata":metadata}
    writeJsonToFile(experimentRecordsPath,json.dumps(currentRecords, indent = 4))
    print(f"Saved {experiment} to {experimentRecordsPath}")

def setOrDefault(experiment,flag,default):
    if flag in experiment:
        return experiment[flag]
    else:
        return default

def experimentWith(experiment,recordsPath,experimentId,threads=128):
            problemname=experiment['problems'][0]
            problemLogPath=experiment['problems'][1]
            hyperLevelMethod=setOrDefault(experiment,'hyperLevelMethod',"SA")
            additionalFlags=setOrDefault(experiment,'additionalFlags',"")
            experimentFlags=f"-D{problemname} \
                            -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE={experiment['trialSampleSizes']} \
                            -DITERATION_COUNT={experiment['baselevelIterations']} \
                            -DPOPULATION_SIZE={experiment['populationSize']} \
                            -DX_DIM={experiment['modelSize']} \
                            -DHH_TRIALS={experiment['trialStepCount']} \
                            -DLOGS_PATH={backslash}{dquote}{problemLogPath}{backslash}{dquote} \
                            -DHH_SA_TEMP={experiment['HH-SA-temp']} \
                            -DHH_SA_ALPHA={experiment['HH-SA-alpha']} \
                            -DEXPERIMENT_HASH_SHA256={backslash}{dquote}{experimentId}{backslash}{dquote},\
                            -DTHREADS_PER_BLOCK={threads},\
                            -DHH_METHOD={backslash}{dquote}{hyperLevelMethod}{backslash}{dquote} \
                            {additionalFlags} \
                            "

            print(f"Running experiment {experimentFlags}")
            start = timer()
            consumeOutput(runOptimizerWith(experimentFlags),lambda line:print(line))
            end = timer()
            return {"elapsedTimeSec":end-start,"threads":threads}

def runExperimentVariations(experimentVariations,experimentIdMapper,recordsPath,threads):
    remainingExperimentsToRun=sum([not experimented(experimentIdMapper(mapExperimentListToDict(experiment)),recordsPath) for experiment in experimentVariations])
    print(f"Total experiments: {len(experimentVariations)}")
    print(f"Remaining experiments: {remainingExperimentsToRun}")
    runningId=0
    for experiment in experimentVariations:
        experimentDict=mapExperimentListToDict(experiment=experiment)
        experimentId=experimentIdMapper(experimentDict)
        if not experimented(experimentId,recordsPath):
            print(f"Running {runningId}/{remainingExperimentsToRun}: {experiment}")
            experimentMetadata=experimentWith(experiment=experimentDict,recordsPath=recordsPath,experimentId=experimentId,threads=threads)
            recordExperiment(experiment=experimentDict,experimentId=experimentId,experimentRecordsPath=recordsPath,metadata=experimentMetadata)
            runningId+=1
        else:
            print(f"Skipping {experiment}")

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
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/CMA-ES/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/CMA-ES/rastrigin.json"),
            #   ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/CMA-ES/styblinskitang.json"),
            #   ("PROBLEM_TRID","hhanalysis/logs/CMA-ES/trid.json"),
            #   ("PROBLEM_SCHWEFEL223","hhanalysis/logs/CMA-ES/schwefel223.json"),
            #   ("PROBLEM_QING","hhanalysis/logs/CMA-ES/qing.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([500],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),CMA_ES_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSA_CMAESExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SA-CMA-ES-NMHH/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SA-CMA-ES-NMHH/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SA-CMA-ES-NMHH/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/SA-CMA-ES-NMHH/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SA-CMA-ES-NMHH/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/SA-CMA-ES-NMHH/qing.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([500],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SA_CMA_ES_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

def runSA_CMAESFullParametersExperiments():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SA-CMA-ES-NMHH/fullparameterbounds/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SA-CMA-ES-NMHH/fullparameterbounds/rastrigin.json"),
            #   ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SA-CMA-ES-NMHH/fullparameterbounds/styblinskitang.json"),
            #   ("PROBLEM_TRID","hhanalysis/logs/SA-CMA-ES-NMHH/fullparameterbounds/trid.json"),
            #   ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SA-CMA-ES-NMHH/fullparameterbounds/schwefel223.json"),
            #   ("PROBLEM_QING","hhanalysis/logs/SA-CMA-ES-NMHH/fullparameterbounds/qing.json")
            ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([500],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA-CMA-ES"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SA_CMA_ES_FULL_PARAMETERS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)


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

def runSANMHHFulParameters_ManyHyperheuristicSteps():
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/SA-NMHH/manyHyperSteps/fullparameterbounds/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/SA-NMHH/manyHyperSteps/fullparameterbounds/rastrigin.json"),
            #   ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/SA-NMHH/manyHyperSteps/fullparameterbounds/styblinskitang.json"),
            #   ("PROBLEM_TRID","hhanalysis/logs/SA-NMHH/manyHyperSteps/fullparameterbounds/trid.json"),
            #   ("PROBLEM_SCHWEFEL223","hhanalysis/logs/SA-NMHH/manyHyperSteps/fullparameterbounds/schwefel223.json"),
            #   ("PROBLEM_QING","hhanalysis/logs/SA-NMHH/manyHyperSteps/fullparameterbounds/qing.json")
              ],"problems")
    
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["trialStepCount"]=zipWithProperty([500],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),SANMHH_FULL_PARAMETERS_MANY_HYPERSTEPS_EXPERIMENT_RECORDS_PATH,DEFAULT_THREAD_COUNT)

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


# runAllExperiments()
# testScalability()
# runTemperatureAnalysis()
# runSASNLPExperiment()
# runRandomHHControlGroupExperiments()
# runSAPerturbExperiments()
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
# runCMAESExperiments()
# runSANMHH_ManyHyperheuristicSteps()
# runSA_CMAESExperiments()
# runSA_CMAESFullParametersExperiments()
runSANMHHFulParameters_ManyHyperheuristicSteps()