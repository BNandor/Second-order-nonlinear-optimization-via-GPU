import os
import json
import itertools
import numpy as np
import scipy
import json
from timeit import default_timer as timer
from dict_hash import sha256
from customhys.experiment import *

backslash="\\"
dquote='"'
def zipWithProperty(list,property):
    print([property]*len(list))
    return zip([property]*len(list),list)

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

def mapExperimentListToDict(experiment):
    paramsDict={}
    for param in experiment:
        if param[1] != None:
            paramsDict[param[0]]=param[1]
    return json.loads(json.dumps(paramsDict))

def setOrDefault(experiment,flag,default):
    if flag in experiment:
        return experiment[flag]
    else:
        return default
    
def recordExperiment(experiment,experimentId,experimentRecordsPath,metadata):
    createIfNotExists(experimentRecordsPath,json.dumps(emptyExperimentRecords(), indent = 4))
    currentRecords=json.load(open(experimentRecordsPath,'r'))
    currentRecords["experiments"][experimentId]={"experiment":experiment,"metadata":metadata}
    writeJsonToFile(experimentRecordsPath,json.dumps(currentRecords, indent = 4))
    print(f"Saved {experiment} to {experimentRecordsPath}")

def hashOfExperiment(experiment):
    return sha256(experiment)

def experimentWith(experiment,path):
            problemname=experiment['problems'][0]
            start = timer()
            exp = Experiment(exp_config={
                                "experiment_name": f"{problemname}-{experiment['modelSize']}", 
                                "experiment_type": "default",
                                "recordspath":path},
                            hh_config={"cardinality": 3, 
                                           "num_replicas": experiment['trialSampleSizes'],
                                           'num_iterations':experiment['baselevelIterations'],
                                           'num_agents':experiment['populationSize'],
                                           "num_steps": experiment['hhsteps']},
                            prob_config= {"dimensions": [experiment['modelSize']],
                                                          "functions": [problemname]})
            
            runDemo(exp)
            end = timer()
            return {"elapsedTimeSec":end-start}

def runExperimentVariations(experimentVariations,experimentIdMapper,recordsPath):
    remainingExperimentsToRun=sum([not experimented(experimentIdMapper(mapExperimentListToDict(experiment)),recordsPath) for experiment in experimentVariations])
    print(f"Total experiments: {len(experimentVariations)}")
    print(f"Remaining experiments: {remainingExperimentsToRun}")
    runningId=0
    for experiment in experimentVariations:
        experimentDict=mapExperimentListToDict(experiment=experiment)
        experimentId=experimentIdMapper(experimentDict)
        if not experimented(experimentId,recordsPath):
            print(f"Running {runningId}/{remainingExperimentsToRun}: {experiment}")
            experimentMetadata=experimentWith(experiment=experimentDict,path=os.path.dirname(recordsPath))
            recordExperiment(experiment=experimentDict,experimentId=experimentId,experimentRecordsPath=recordsPath,metadata=experimentMetadata)
            runningId+=1
        else:
            print(f"Skipping {experiment}")


def runExperiments(recordsRootPath,config):
    recordsPath=f"{recordsRootPath}/{config['name']}/records.json"
    params={}
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["problems"]=zipWithProperty(config['problems'],"problems")
    params["hhsteps"]=zipWithProperty([100],"hhsteps")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordsPath)

