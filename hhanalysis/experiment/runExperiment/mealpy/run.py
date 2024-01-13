#!/usr/bin/env python
# Created by "Thieu" at 15:49, 10/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
from mealpy.bio_based import BBO, EOA, IWO, SBO, SMA, TPO, VCS, WHO
from mealpy.evolutionary_based import CRO, DE, EP, ES, FPA, GA, MA
from mealpy.human_based import BRO, BSO, CA, CHIO, FBIO, GSKA, ICA, LCO, QSA, SARO, SSDO, TLO
from mealpy.math_based import AOA, CEM, CGO, GBO, HC, PSS, SCA
from mealpy.music_based import HS
from mealpy.physics_based import ArchOA, ASO, EFO, EO, HGSO, MVO, NRO, SA, TWO, WDO
from mealpy.system_based import AEO, GCO, WCA
from mealpy.swarm_based import ABC, ACOR, ALO, AO, BA, BeesA, BES, BFO, BSA, COA, CSA, CSO, DO, EHO, FA, FFA, FOA, GOA, GWO, HGS
from mealpy.swarm_based import HHO, JA, MFO, MRFO, MSA, NMRA, PFA, PSO, SFO, SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, WOA
import itertools
import os
import numpy as np
import scipy
import json
import math
from timeit import default_timer as timer
from dict_hash import sha256
import multiprocessing

backslash="\\"
dquote='"'

def rosenbrock(variables):
        return np.sum(100. * np.square(variables[1:] - np.square(
                variables[:-1])) + np.square(variables[:-1] - 1))
def qing(variables):
    variables_num=len(variables)
    return np.sum(np.square(np.square(variables) - (np.arange(variables_num) + 1.)))      
def rastrigin(variables):
    variables_num=len(variables)
    return 10. * variables_num + np.sum(
            np.square(variables) - 10. * np.cos(2. * np.pi * variables))
def schwefel223(variables):
    return np.sum(np.power(variables, 10.))
def styblinskitang(variables):
    return 0.5 * np.sum(np.power(variables, 4) - 16. * np.square(variables) + 5. * variables)
def trid(variables):
    return np.sum(np.square(variables - 1)) - np.sum(variables[1:] * variables[:-1])
def michalewicz(x):
    D = len(x)
    i = np.arange(1, D + 1)
    return-np.sum(np.sin(x) * np.sin(((i * x**2) / np.pi))**(20))
def dixon_price(x):
    return (x[0] - 1) ** 2 + np.sum(np.arange(2, len(x) + 1)*(2 * (x[1:]**2) - x[:-1]) ** 2)
def levy(x):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1]) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3
def schwefel(x):
    n = len(x)
    abs_x = np.abs(x)
    return n* 418.9829 - np.sum(x * np.sin(np.sqrt(abs_x)))


class ProblemSet:
    def __init__(self, dim):
        self.dim=dim
    def fitness_function(solution):
        return np.sum(solution**2)
    def getProblem(self,name):
       return {"PROBLEM_ROSENBROCK":self.getRosenbrock(),
              "PROBLEM_RASTRIGIN":self.getRastrigin(),
              "PROBLEM_STYBLINSKITANG":self.getStyblinskiTang(),
              "PROBLEM_TRID":self.getTrid(),
              "PROBLEM_SCHWEFEL223":self.getSchwefel223(),
              "PROBLEM_QING":self.getQing(),
              "PROBLEM_MICHALEWICZ":self.getMichalewicz(),
              "PROBLEM_DIXONPRICE":self.getDixonPrice(),
              "PROBLEM_LEVY":self.getLevy(),
              "PROBLEM_SCHWEFEL":self.getSchwefel(),
              "PROBLEM_SUMSQUARES": self.getSumSquares(),
              "PROBLEM_SPHERE": self.getSphere()}[name]

    def getRosenbrock(self):
        return {
            "fit_func": rosenbrock,
            "lb": [-30, ] * self.dim,
            "ub": [30, ] * self.dim,
            "minmax": "min",
               "log_to": None
        }

    def getQing(self):
        return {
        "fit_func": qing,
        "lb": [-500, ] * self.dim,
        "ub": [500, ] * self.dim,
        "minmax": "min",
           "log_to": None
    }

    def getRastrigin(self):
        return {
            "fit_func": rastrigin,
            "lb": [-5.12, ] * self.dim,
            "ub": [5.12, ] * self.dim,
            "minmax": "min",
               "log_to": None
        }

    def getSchwefel223(self):
        return {
            "fit_func": schwefel223,
            "lb": [-10, ] * self.dim,
            "ub": [10, ] * self.dim,
            "minmax": "min",
               "log_to": None
        }

    def getStyblinskiTang(self):
        return {
            "fit_func": styblinskitang,
            "lb": [-5, ] * self.dim,
            "ub": [5, ] * self.dim,
            "minmax": "min",
               "log_to": None
        }

    def getTrid(self):
        return {
            "fit_func": trid,
            "lb": [-np.square(self.dim), ] * self.dim,
            "ub": [np.square(self.dim), ] * self.dim,
            "minmax": "min",
               "log_to": None
        }
    def getMichalewicz(self):
        return {
            "fit_func": michalewicz,
            "lb": [0, ] * self.dim,
            "ub": [math.pi, ] * self.dim,
            "minmax": "min",
               "log_to": None
        }
    def getDixonPrice(self):
        return {
            "fit_func": dixon_price,
            "lb": [-10, ] * self.dim,
            "ub": [10, ] * self.dim,
            "minmax": "min",
               "log_to": None
        }
    def getLevy(self):
        return {
            "fit_func": levy,
            "lb": [-10, ] * self.dim,
            "ub": [10, ] * self.dim,
            "minmax": "min",
               "log_to": None
        }
    def getSchwefel(self):
        return {
            "fit_func": schwefel,
            "lb": [-500, ] * self.dim,
            "ub": [500, ] * self.dim,
            "minmax": "min",
            "log_to": None
        }
    def getSumSquares(self):
        return {
            "fit_func": lambda x: np.sum(np.arange(1, self.dim + 1) * np.square(x)),
            "lb": [-5.12, ] * self.dim,
            "ub": [5.12, ] * self.dim,
            "minmax": "min",
            "log_to": None
        }
    def getSphere(self):
        return {
            "fit_func": lambda x: np.sum(np.square(x)),
            "lb": [-5.12, ] * self.dim,
            "ub": [5.12, ] * self.dim,
            "minmax": "min",
            "log_to": None
        }

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
def createMealpyOptimizer(name,epoch,pop_size):
    optimizers={'CRO': CRO.OriginalCRO(epoch=epoch, pop_size=pop_size,po= 0.4,Fb= 0.9,Fa= 0.1, Fd= 0.1,Pd= 0.5,GCR= 0.1,gamma_min= 0.02,gamma_max= 0.2,n_trials= 5),
    'BRO': BRO.BaseBRO(epoch=epoch, pop_size=pop_size, threshold=3),
    'ArchOA':ArchOA.OriginalArchOA(epoch=epoch, pop_size=pop_size,c1= 2,
                                                        c2= 5,
                                                        c3= 2,
                                                        c4= 0.5,
                                                        acc_max= 0.9,
                                                        acc_min= 0.1,),
    'AEO': AEO.OriginalAEO(epoch=epoch, pop_size=pop_size),
    'SMA': SMA.BaseSMA(epoch=epoch, pop_size=pop_size, pr=0.03),
    'PSO':PSO.OriginalPSO(epoch=epoch, pop_size=pop_size,c1=2.05,c2=2.05,w_min=0.4,w_max=0.9)
    }
    return optimizers[name]

def experimentWith(experiment):
            problemname=experiment['problems'][0]
            problemSet=ProblemSet(experiment['modelSize'])
            epochs=experiment['baselevelIterations']
            start = timer()
            trials=[]
            min_med_iqr=float("inf")
            for step in range(experiment['hhsteps']):
                results=[]
                for sample in range(experiment['trialSampleSizes']):
                    print(f"Running experiment {experiment} {step}/{experiment['hhsteps']} - {sample}/{experiment['trialSampleSizes']}")
                    optimizer=createMealpyOptimizer(experiment['optimizer'],epochs,experiment['populationSize'])
                    best_position, best_fitness = optimizer.solve(problemSet.getProblem(problemname),mode='swarm')
                    print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
                    results.append(best_fitness)
                med_iqr=np.median(results)+scipy.stats.iqr(results)
                trial={"med_+_iqr": med_iqr,
                    "performanceSamples":results}
                if min_med_iqr > med_iqr:
                    min_med_iqr=med_iqr
                trials.append(trial)
            end = timer()
            return {"elapsedTimeSec":end-start,'med_iqr':min_med_iqr,'trials':trials}

def currentExperimentsAndRecordPath(recordsPathPrefix,threads=1,threadId=0):
    allExperiments=set()
    if threads>1:
        for experiments in [json.load(open(experimentRecordsPath,'r'))["experiments"].keys() for experimentRecordsPath in [f"{recordsPathPrefix}_{thread}" for thread in range(threads)] ]:
            for experiment in experiments:
                allExperiments.add(experiment)
        recordsPath=f"{recordsPathPrefix}_{threadId}"
    else:
        allExperiments=json.load(open(recordsPathPrefix,'r'))["experiments"].keys()
        recordsPath=recordsPathPrefix
        
    return allExperiments,recordsPath

def runExperimentVariations(experimentVariations,experimentIdMapper,recordsPathPrefix,threadId=0,threads=1):
    allExperiments,recordsPath=currentExperimentsAndRecordPath(recordsPathPrefix,threads,threadId)
    print(f"Total experiments: {len(experimentVariations)}")
    remainingExperiments=len(experimentVariations)-len(allExperiments)
    print(f"Remaining experiments: {remainingExperiments}")
    runningId=0
    for experiment in experimentVariations:
        experimentDict=mapExperimentListToDict(experiment=experiment)
        experimentId=experimentIdMapper(experimentDict)
        if experimentId not in allExperiments:
            if runningId%threads == threadId:
                print(f"Running {runningId}/{remainingExperiments}: {experiment}")
                experimentMetadata=experimentWith(experiment=experimentDict)
                recordExperiment(experiment=experimentDict,experimentId=experimentId,experimentRecordsPath=recordsPath,metadata=experimentMetadata)
            runningId+=1
        else:
            print(f"Skipping {experiment}")

def mealpyOptimizers():
    return [ 'AEO','CRO','BRO','ArchOA','SMA','PSO']

def runGeccoExperiments(recordsPath,config):
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/qing.json")],"problems")
    params["hhsteps"]=zipWithProperty([100],"hhsteps")
    params["baselevelIterations"]=zipWithProperty([5000],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["optimizer"]=zipWithProperty(mealpyOptimizers(),"optimizer")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordsPath)

def runLODExperiments(recordsPath,config):
    params={}
    params["problems"]=zipWithProperty([
              ("PROBLEM_ROSENBROCK","hhanalysis/logs/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN","hhanalysis/logs/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG","hhanalysis/logs/styblinskitang.json"),
              ("PROBLEM_TRID","hhanalysis/logs/trid.json"),
              ("PROBLEM_SCHWEFEL223","hhanalysis/logs/schwefel223.json"),
              ("PROBLEM_QING","hhanalysis/logs/qing.json")],"problems")
    params["hhsteps"]=zipWithProperty([100],"hhsteps")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty([30],"populationSize")
    params["modelSize"]=zipWithProperty([5,50,100,500],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["optimizer"]=zipWithProperty(['CRO'],"optimizer")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordsPath)

def runExtraBenchMarks(recordsRootPath,config):
    recordsPath=f"{recordsRootPath}/{config['name']}/records.json"
    params={}
    params["problems"]=zipWithProperty(config['problems'],"problems")
    params["hhsteps"]=zipWithProperty([100],"hhsteps")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    params["optimizer"]=zipWithProperty(config['optimizers'],"optimizer")
    variations=list(itertools.product(*list(params.values()))) 
    processes=[]
    # Start processes
    cores=multiprocessing.cpu_count()
    for i in range(cores):
        process = multiprocessing.Process(target=runExperimentVariations, args=(variations,lambda exp:hashOfExperiment(exp),recordsPath,i,cores))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
