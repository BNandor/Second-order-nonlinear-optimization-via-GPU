import numpy as np
import json
import subprocess
import itertools
import os
from dict_hash import sha256
from timeit import default_timer as timer

backslash="\\"
dquote='"'
EXPERIMENT_RECORDS_PATH="../logs/records.json"

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

def hashOfExperiment(experiment):
    return sha256(experiment)

def experimented(experiment,experimentRecordsPath):
    createIfNotExists(experimentRecordsPath,json.dumps(emptyExperimentRecords(), indent = 4))
    currentRecords=json.load(open(experimentRecordsPath,'r'))
    return hashOfExperiment(experiment=experiment) in currentRecords["experiments"].keys()

def mapExperimentListToDict(experiment):
    paramsDict={}
    for param in experiment:
        paramsDict[param[0]]=param[1]
    return json.loads(json.dumps(paramsDict))


def recordExperiment(experiment,experimentRecordsPath,metadata):
    createIfNotExists(experimentRecordsPath,json.dumps(emptyExperimentRecords(), indent = 4))
    currentRecords=json.load(open(experimentRecordsPath,'r'))
    currentRecords["experiments"][hashOfExperiment(experiment)]={"experiment":experiment,"metadata":metadata}
    writeJsonToFile(experimentRecordsPath,json.dumps(currentRecords, indent = 4))
    print(f"Saved {experiment} to {experimentRecordsPath}")

def zipWithProperty(list,property):
    print([property]*len(list))
    return zip([property]*len(list),list)

def experimentWith(experiment,recordsPath):
            problemname=experiment['problems'][0]
            problemLogPath=experiment['problems'][1]
            experimentFlags=f"-D{problemname} \
                            -DHYPER_LEVEL_TRIAL_SAMPLE_SIZE={experiment['trialSampleSizes']} \
                            -DITERATION_COUNT={experiment['baselevelIterations']} \
                            -DPOPULATION_SIZE={experiment['populationSize']} \
                            -DX_DIM={experiment['modelSize']} \
                            -DHH_TRIALS={experiment['trialStepCount']} \
                            -DLOGS_PATH={backslash}{dquote}{problemLogPath}{backslash}{dquote} \
                            -DHH_SA_TEMP={experiment['HH-SA-temp']} \
                            -DHH_SA_ALPHA={experiment['HH-SA-alpha']} \
                            -DEXPERIMENT_HASH_SHA256={backslash}{dquote}{hashOfExperiment(experiment=experiment)}{backslash}{dquote}"
            print(f"Running experiment {experimentFlags}")
            start = timer()
            consumeOutput(runOptimizerWith(experimentFlags),lambda line:print(line))
            end = timer()
            return {"elapsedTimeSec":end-start}

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

    for experiment in list(itertools.product(*list(params.values()))):
        experimentDict=mapExperimentListToDict(experiment=experiment)
        if not experimented(experimentDict,EXPERIMENT_RECORDS_PATH):
            experimentMetadata=experimentWith(experiment=experimentDict,recordsPath=EXPERIMENT_RECORDS_PATH)
            recordExperiment(experiment=experimentDict,experimentRecordsPath=EXPERIMENT_RECORDS_PATH,metadata=experimentMetadata)
        else:
            print(f"Skipping {experiment}")

runAllExperiments()