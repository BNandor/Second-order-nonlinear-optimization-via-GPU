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

def runOptimizerWith(flags):
    process = subprocess.Popen(['make', 'buildAndRunExperimentWith', f'NVCCFLAGS={flags}', f'ROOT={os.getcwd()}/../..'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.STDOUT)
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Creating {path} with {content}")
        writeJsonToFile(path=path,content=content)

def emptyExperimentRecords():
    return {"experiments":{}}

def experimented(experimentId,experimentRecordsPath):
    createIfNotExists(experimentRecordsPath,json.dumps(emptyExperimentRecords(), indent = 4))
    currentRecords=json.load(open(experimentRecordsPath,'r'))
    return experimentId in currentRecords["experiments"].keys() 

def trimExtraSamples(path):
    if os.path.isfile(path):
        currentRecords=json.load(open(path,'r'))
        for experiment in currentRecords["experiments"]:
            experiment['trials']=experiment['trials'][:experiment['trialCount']]
        writeJsonToFile(path,json.dumps(currentRecords, indent = 4))

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
def enforceFunctionEvaluationLimit(recordsPath):
    problems=['rosenbrock.json',
              'rastrigin.json',
              'styblinskitang.json',
              'trid.json',
              'schwefel223.json',
              'qing.json',
              "michalewicz.json",
              "dixonprice.json",
              "levy.json",
              "schwefel.json",
              "sumsquares.json",
              "sphere.json"]
    for problem in problems:
        trimExtraSamples(f"{recordsPath.replace('/records.json','')}/{problem}")
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

def runExperimentVariations(experimentVariations,experimentIdMapper,recordsPath,threads,enforceEvalLimit=True):
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
    if enforceEvalLimit:
        print('Enforcing function evaluation limits')
        enforceFunctionEvaluationLimit(recordsPath)
