import numpy as np
import json
import subprocess
import itertools
import os
import copy
from timeit import default_timer as timer
from analysis.common import *

backslash="\\"
dquote='"'
EXPERIMENT_RECORDS_PATH="../logs/records.json"
RANDOM_CONTROL_GROUP_EXPERIMENT_RECORDS_PATH="../logs/randomHH/records.json"
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
            baselevelParameterOverride=setOrDefault(experiment,'baselevelParameterOverride',"NO")
            baselevelParameterOverridePath=setOrDefault(experiment,'baselevelParameterOverridePath',"")
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
                            -DHH_METHOD={backslash}{dquote}{hyperLevelMethod}{backslash}{dquote}, \
                            -DBASE_PARAM_OVERLOAD={backslash}{dquote}{baselevelParameterOverride}{backslash}{dquote}, \
                            -DBASE_PARAM_OVERLOAD_PATH={backslash}{dquote}{baselevelParameterOverridePath}{backslash}{dquote} \
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

{
# INITIALIZER_EXPERIMENT_PATH="../logs/paramInitializerTests/records.json"
# def runInitAnalysis():
#     params={}
#     # BASE_PARAM_OVERLOAD
#     # BASE_PARAM_OVERLOAD_PATH
#     params["problems"]=zipWithProperty([
#               ("PROBLEM_ROSENBROCK","hhanalysis/logs/rosenbrock.json")],"problems")
    
#     params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
#     params["populationSize"]=zipWithProperty([30],"populationSize")
#     params["modelSize"]=zipWithProperty([500],"modelSize")
#     params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
#     params["trialStepCount"]=zipWithProperty([1],"trialStepCount")
#     params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
#     params["HH-SA-alpha"]=zipWithProperty([5],"HH-SA-alpha")
#     params["baselevelParameterOverride"]=zipWithProperty(["YES","NO"],"baselevelParameterOverride")
#     params["baselevelParameterOverridePath"]=zipWithProperty(["baseLevelParamOverride.json"],"baselevelParameterOverridePath")
    
#     variations=list(itertools.product(*list(params.values())))
#     runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),INITIALIZER_EXPERIMENT_PATH,DEFAULT_THREAD_COUNT)
}

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
runAllExperiments()
# testScalability()
# runTemperatureAnalysis()
# runRandomHHControlGroupExperiments()