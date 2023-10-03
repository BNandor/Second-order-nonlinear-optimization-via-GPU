import numpy as np
import json
import subprocess
import itertools
import os
import copy
from timeit import default_timer as timer
from analysis.common import *
from runExperiment.ecai_peerj.run import *
import runExperiment.mealpy.run

import pandas as pd


backslash="\\"
dquote='"'
ROOT="../../"
LOGS_PATH_FROM_ROOT="hhanalysis/logs"

def runEcaiSuite():
    problems=lambda logspath: [
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
              ]
    dimensions=[100]
    populationSize=[100]
    config={'name':'bigDim/100/bigPop/100',
            'problems':problems,
            'dimensions':dimensions,
            'populationSize':populationSize
            }
    runNMHH2(LOGS_PATH_FROM_ROOT,ROOT,config)
    runSA_NMHH_GA_DE_GD_LBFGS_GWO(LOGS_PATH_FROM_ROOT,ROOT,config)
    runMADS_NMHH_GA_DE_GD_LBFGS_GWO(LOGS_PATH_FROM_ROOT,ROOT,config)
    runMADSExperiments(LOGS_PATH_FROM_ROOT,ROOT,config)
    runbigSA_MADS_NMHH_GA_DE_GD_LBFGS(LOGS_PATH_FROM_ROOT,ROOT,config)
    runbigSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO(LOGS_PATH_FROM_ROOT,ROOT,config)
    runSA_MADS_NMHH_GA_DE_GD_LBFGS_GWO(LOGS_PATH_FROM_ROOT,ROOT,config)
    runSA_MADS_NMHH_GA_DE_GD_LBFGS(LOGS_PATH_FROM_ROOT,ROOT,config)
    runCMAESExperiments(LOGS_PATH_FROM_ROOT,ROOT,config)
    runCMAES_GA_DE_GD_LBFGS_GWOExperiments(LOGS_PATH_FROM_ROOT,ROOT,config)
    runSA_CMAESExperiments(LOGS_PATH_FROM_ROOT,ROOT,config)
    runSA_CMAES_ES_GA_DE_GD_LBFGS_GWO_Experiments(LOGS_PATH_FROM_ROOT,ROOT,config)
    runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments(LOGS_PATH_FROM_ROOT,ROOT,config)
    runbigSA_CMAES_ES_GA_DE_GD_LBFGS_Experiments_GWO(LOGS_PATH_FROM_ROOT,ROOT,config)

def runMealpySuite():
    EXPERIMENT_RECORDS_PATH=f"{ROOT}/{LOGS_PATH_FROM_ROOT}/mealpyPerf/benchmarks/"
    dimensions=[2,3,4,5,6,7,8,9,10,15,30,50,100]
    optimizers=[ 'AEO','CRO','BRO','ArchOA','SMA','PSO']
    problems=[("PROBLEM_MICHALEWICZ","hhanalysis/logs/michalewicz.json"),
            ("PROBLEM_DIXONPRICE","hhanalysis/logs/dixonprice.json"),
            ("PROBLEM_LEVY","hhanalysis/logs/levy.json"),
            ("PROBLEM_SCHWEFEL", "hhanalysis/logs/schwefel.json"),
            ("PROBLEM_SUMSQUARES", "hhanalysis/logs/sumsquares.json"),
            ("PROBLEM_SPHERE", "hhanalysis/logs/sphere.json"),
            ]
    populationSize=[30]
    config={'name':'dim/2_100/pop/30',
            'problems':problems,
            'dimensions':dimensions,
            'populationSize':populationSize,
            'optimizers':optimizers,
            }
    runExperiment.mealpy.run.runExtraBenchMarks(EXPERIMENT_RECORDS_PATH,config)

# runEcaiSuite()
runMealpySuite()

