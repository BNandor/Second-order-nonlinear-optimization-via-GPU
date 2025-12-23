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


def runAblationSAPerturb(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SAPerturb/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"

    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    steps= config['trialStepCount'] if 'trialStepCount' in config else 100
    params["trialStepCount"]=zipWithProperty([steps],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA_PERTURB"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)


def runAblationSARefine(logsPathFromRoot,root,config):
    logspath=f"{logsPathFromRoot}/SARefine/{config['name']}"
    recordspath=f"{root}/{logspath}/records.json"

    params={}
    params["problems"]=zipWithProperty(config['problems'](logspath),"problems")
    params["baselevelIterations"]=zipWithProperty([100],"baselevelIterations")
    params["populationSize"]=zipWithProperty(config['populationSize'],"populationSize")
    params["modelSize"]=zipWithProperty(config['dimensions'],"modelSize")
    params["trialSampleSizes"]=zipWithProperty([30],"trialSampleSizes")
    steps= config['trialStepCount'] if 'trialStepCount' in config else 100
    params["trialStepCount"]=zipWithProperty([steps],"trialStepCount")
    params["HH-SA-temp"]=zipWithProperty([10000],"HH-SA-temp")
    params["HH-SA-alpha"]=zipWithProperty([50],"HH-SA-alpha")
    params["hyperLevelMethod"]=zipWithProperty(["SA_REFINE"],"hyperLevelMethod")
    variations=list(itertools.product(*list(params.values())))
    runExperimentVariations(variations,lambda exp:hashOfExperiment(exp),recordspath,DEFAULT_THREAD_COUNT)
