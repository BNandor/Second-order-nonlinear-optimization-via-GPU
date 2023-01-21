import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
import scipy.stats as stats
import json
from common import *
import re

CustomHYSPath="./results/CustomHYS/results.json"
problemNameMapping={
    "Qing":"PROBLEM_QING",
    "Rastrigin":"PROBLEM_RASTRIGIN",
    "Rosenbrock":"PROBLEM_ROSENBROCK",
    "Schwefel223": "PROBLEM_SCHWEFEL223",
    "StyblinskiTang": "PROBLEM_STYBLINSKITANG",
    "Trid":"PROBLEM_TRID"
}

def loadResultsSteps(path):
    return loadJsonFrom(path,ignoreTrailingCommas=True)

def stepsToResults(steps):
    results=[]
    for problem in steps.keys():
        problemParts=problem.split("-")
        results.append({
            "problemName":problemNameMapping[problemParts[0]],
            "modelSize":int(problemParts[1].replace("D","")),
            "baselevelIterations":int(re.sub("iterations.*","",problemParts[3])),
            "minMedIQR":steps[problem][-1]["med_iqr"],
        })
    return results

def onlySteps(steps):
    results=[]
    for problem in steps.keys():
        problemParts=problem.split("-")
        results.append({
            "problemName":problemNameMapping[problemParts[0]],
            "modelSize":int(problemParts[1].replace("D","")),
            "baselevelIterations":int(re.sub("iterations.*","",problemParts[3])),
            "steps":json.dumps(steps[problem]),
        })
    return results