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
import runExperiment.customhys.customhys.batchexperiments 

import pandas as pd


backslash="\\"
dquote='"'
ROOT="../../"
LOGS_PATH_FROM_ROOT="hhanalysis/logs"

def runRandomHHSuite():
        problems=lambda logspath: [("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json")]
        dimensions=[5,50,100,500]
        populationSize=[30]
        config={'name':'/',
                'problems':problems,
                'dimensions':dimensions,
                'populationSize':populationSize
                }
        runRandomHHControlGroupExperiments(LOGS_PATH_FROM_ROOT,ROOT,config)

def runNMHHSuite():
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
    populationSize=[30]
    config={'name':'newExperiment',
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

def runCUSTOMHySSuite():
    EXPERIMENT_RECORDS_PATH=f"{ROOT}/{LOGS_PATH_FROM_ROOT}/CustomHYSPerf/michalewiczDixonPriceLecy750/"
    #dimensions=[2,3,4,5,6,7,8,9,10,15,30,50,100]
    dimensions=[750]
 
    problems=[
#              ("Rosenbrock",f"hhanalysis/logs/rosenbrock.json"),
#              ("Rastrigin",f"hhanalysis/logs/rastrigin.json"),
#          ("StyblinskiTang",f"/styblinskitang.json"),
#              ("Trid",f"hhanalysis/logs/trid.json"),
#              ("Schwefel223",f"hhanalysis/logs/schwefel223.json"),
#              ("Qing",f"hhanalysis/logs/qing.json"),
#            ("PROBLEM_MICHALEWICZ","hhanalysis/logs/michalewicz.json"),
#            ("PROBLEM_DIXONPRICE","hhanalysis/logs/dixonprice.json"),
            ("PROBLEM_LEVY","hhanalysis/logs/levy.json"),
#            ("PROBLEM_SCHWEFEL", "hhanalysis/logs/schwefel.json"),
#            ("PROBLEM_SUMSQUARES", "hhanalysis/logs/sumsquares.json"),
#            ("PROBLEM_SPHERE", "hhanalysis/logs/sphere.json"),
            ]
    populationSize=[30]
    config={'name':'/',
            'problems':problems,
            'dimensions':dimensions,
            'populationSize':populationSize,
            }
    runExperiment.customhys.customhys.batchexperiments.runExperiments(EXPERIMENT_RECORDS_PATH,config)

def runMealpySuite():
    EXPERIMENT_RECORDS_PATH=f"{ROOT}/{LOGS_PATH_FROM_ROOT}/mealpyPerf/benchmarks/"
    #dimensions=[2,3,4,5,6,7,8,9,10,15,30,50,100,500]
    dimensions=[750]
    optimizers=[ 'AEO','CRO','BRO','ArchOA','SMA','PSO']
    problems=[
              ("PROBLEM_ROSENBROCK",f"hhanalysis/logs/rosenbrock.json"),
              ("PROBLEM_RASTRIGIN",f"hhanalysis/logs/rastrigin.json"),
              ("PROBLEM_STYBLINSKITANG",f"/styblinskitang.json"),
              ("PROBLEM_TRID",f"hhanalysis/logs/trid.json"),
              ("PROBLEM_SCHWEFEL223",f"hhanalysis/logs/schwefel223.json"),
              ("PROBLEM_QING",f"hhanalysis/logs/qing.json"),
            ("PROBLEM_MICHALEWICZ","hhanalysis/logs/michalewicz.json"),
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

def runNMHHComputationalTimeExperiments():
        problems=lambda logspath: [("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json")]
        dimensions=[5,100]
        populationSize=[30]
        for i in range(10):
                config={'name':f'comptime/{i}',
                        'problems':problems,
                        'dimensions':dimensions,
                        'populationSize':populationSize
                        }
                runNMHH2(LOGS_PATH_FROM_ROOT,ROOT,config)

def runCUSTOMHySComputationalTimeExperiments():
        EXPERIMENT_RECORDS_PATH=f"{ROOT}/{LOGS_PATH_FROM_ROOT}/CustomHYSPerf/newExperiment/"
        problems= [("Rosenbrock",f"hhanalysis/logs/rosenbrock.json")]
        dimensions=[5,100]
        populationSize=[30]
        for i in range(10):
                config={'name':f'comptime/{i}',
                        'problems':problems,
                        'dimensions':dimensions,
                        'populationSize':populationSize
                        }
  #              runExperiment.customhys.customhys.batchexperiments.runExperiments(EXPERIMENT_RECORDS_PATH,config)

def runClusteringSuite():
    populationSize=[40]
    config={
                'name':'/clustering/iris/40k',
                'populationSize':populationSize
            }
    runClusterinProblems(LOGS_PATH_FROM_ROOT,ROOT,config)

def runSPRTTTestNMHHSuite():
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
    dimensions=[5,50,100]
    populationSize=[30]
    config={'name':'sprt-seq-t',
            'problems':problems,
            'dimensions':dimensions,
            'populationSize':populationSize}
    runSPRTTestNMHH(LOGS_PATH_FROM_ROOT,ROOT,config)
# runNMHHComputationalTimeExperiments()
# runCUSTOMHySComputationalTimeExperiments()
# runRandomHHSuite()
# runNMHHSuite()
# runCUSTOMHySSuite()
# runMealpySuite()
# runClusteringSuite()
runSPRTTTestNMHHSuite()