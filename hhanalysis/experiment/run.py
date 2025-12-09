import numpy as np
import json
import subprocess
import itertools
import os
import copy
import pandas as pd
from timeit import default_timer as timer
from analysis.common import *
from runExperiment.ecai_peerj.run import *
from runExperiment.classification.commonClassificationRun import *
import runExperiment.mealpy.run
import runExperiment.customhys.customhys.batchexperiments 


backslash="\\"
dquote='"'
ROOT="../../"
LOGS_PATH_FROM_ROOT="hhanalysis/logs"


def runSNLPSuite():
    problems=lambda logspath: [
              ("PROBLEM_SNLP",f"{logspath}/snlp.json"),
              ]
    config={'name':'boxsize_100_comm_r_50_total_nodes_n_50_anchors_n_10_gpsErr_3',
            'problems':problems,
            'solver':'SA',
            'n':47,
            'populationSize':100,
            'baselevelIterations':200,
            'trialStepCount':100
            }
    runSASNLPExperiment(LOGS_PATH_FROM_ROOT,ROOT,config)

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

def runShiftedNMHHSuite():
    problems=lambda logspath: [
              ("PROBLEM_SHIFTED_RASTRIGIN",f"{logspath}/shifted_rastrigin.json"),
              ("PROBLEM_SHIFTED_SCHWEFEL223",f"{logspath}/shifted_schwefel223.json")
              ]
    dimensions=[5,50,100,500]
    populationSize=[30]
    config={'name':'shifted',
            'problems':problems,
            'dimensions':dimensions,
            'populationSize':populationSize
            }
    runNMHH2(LOGS_PATH_FROM_ROOT,ROOT,config)

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
 #   runExperiment.customhys.customhys.batchexperiments.runExperiments(EXPERIMENT_RECORDS_PATH,config)

def runShiftedCUSTOMHySSuite():
    EXPERIMENT_RECORDS_PATH=f"{ROOT}/{LOGS_PATH_FROM_ROOT}/CustomHYSPerf"
    dimensions=[5,50,100,500]
 
    problems=[
             ("PROBLEM_SHIFTED_RASTRIGIN","hhanalysis/logs/CustomHYSPerf/shifted_rastrigin.json"),
              ("PROBLEM_SHIFTED_SCHWEFEL223","hhanalysis/logs/CustomHYSPerf/shifted_schwefel223.json")
            ]
    populationSize=[30]
    config={'name':'shifted',
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


def runShiftedMealpySuite():
    EXPERIMENT_RECORDS_PATH=f"{ROOT}/{LOGS_PATH_FROM_ROOT}/mealpyPerf/benchmarks/"
    dimensions=[5,50,100,500]
    optimizers=[ 'AEO','CRO','BRO','ArchOA','SMA','PSO']
#     dimensions=[500]
#     optimizers=['BRO']
    problems=[
              ("PROBLEM_SHIFTED_RASTRIGIN",f"hhanalysis/logs/shifted_rastrigin.json"),
              ("PROBLEM_SHIFTED_SCHWEFEL223",f"hhanalysis/logs/schwefel223.json"),
            ]
    populationSize=[30]
    config={'name':'shifted/pop/30',
            'problems':problems,
            'dimensions':dimensions,
            'populationSize':populationSize,
            'optimizers':optimizers,
            }
    runExperiment.mealpy.run.runExtraBenchMarks(EXPERIMENT_RECORDS_PATH,config)


def runNMHHComputationalTimeExperiments():
        problems=lambda logspath: [("PROBLEM_ROSENBROCK",f"{logspath}/rosenbrock.json"),
                                   ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
                                        ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
                                        ("PROBLEM_TRID",f"{logspath}/trid.json"),
                                        ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
                                        ("PROBLEM_QING",f"{logspath}/qing.json"),
                                ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
                                ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
                                ("PROBLEM_LEVY",f"{logspath}/levy.json"),
                                ("PROBLEM_SCHWEFEL", f"{logspath}/schwefel.json"),
                                ("PROBLEM_SUMSQUARES", f"{logspath}/sumsquares.json"),
                                ("PROBLEM_SPHERE", f"{logspath}/sphere.json")]
        dimensions=[2,100,750]
        populationSize=[30]
        iterations=10
        for i in range(iterations):
                config={'name':f'comptime_extended/{i}',
                        'problems':problems,
                        'dimensions':dimensions,
                        'populationSize':populationSize
                        }
                start_time = timer()
                runNMHH2(LOGS_PATH_FROM_ROOT,ROOT,config)
                elapsed = timer() - start_time
                remaining_iterations = iterations - (i + 1)
                estimated_remaining = elapsed * remaining_iterations
                print(f"Iteration {i+1}/iterations completed in {elapsed:.2f}s. Estimated remaining time: {estimated_remaining:.2f}s")


def runCUSTOMHySComputationalTimeExperiments():
        EXPERIMENT_RECORDS_PATH=f"{ROOT}/{LOGS_PATH_FROM_ROOT}/CustomHYSPerf/newExperiment/"
        problems= [("Rosenbrock",f"hhanalysis/logs/rosenbrock.json"),
                   ("Rastrigin",f"hhanalysis/logs/rastrigin.json"),
                   ("StyblinskiTang",f"/styblinskitang.json"),
                   ("Trid",f"hhanalysis/logs/trid.json"),
                   ("Schwefel223",f"hhanalysis/logs/schwefel223.json"),
                   ("Qing",f"hhanalysis/logs/qing.json")]
        dimensions=[5,100]
        populationSize=[30]
        iterations=10
        for i in range(iterations):
                config={'name':f'comptime/{i}',
                        'problems':problems,
                        'dimensions':dimensions,
                        'populationSize':populationSize
                        }
                start_time = timer()
                runExperiment.customhys.customhys.batchexperiments.runExperiments(EXPERIMENT_RECORDS_PATH,config)
                elapsed = timer() - start_time
                remaining_iterations = iterations - (i + 1)
                estimated_remaining = elapsed * remaining_iterations
                print(f"Iteration {i+1}/iterations completed in {elapsed:.2f}s. Estimated remaining time: {estimated_remaining:.2f}s")

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
        #       ("PROBLEM_RASTRIGIN",f"{logspath}/rastrigin.json"),
        #       ("PROBLEM_STYBLINSKITANG",f"{logspath}/styblinskitang.json"),
        #       ("PROBLEM_TRID",f"{logspath}/trid.json"),
        #       ("PROBLEM_SCHWEFEL223",f"{logspath}/schwefel223.json"),
        #       ("PROBLEM_QING",f"{logspath}/qing.json"),
        #       ("PROBLEM_MICHALEWICZ",f"{logspath}/michalewicz.json"),
        #       ("PROBLEM_DIXONPRICE",f"{logspath}/dixonprice.json"),
        #       ("PROBLEM_LEVY",f"{logspath}/levy.json"),
        #       ("PROBLEM_SCHWEFEL",f"{logspath}/schwefel.json"),
        #       ("PROBLEM_SUMSQUARES",f"{logspath}/sumsquares.json"),
        #       ("PROBLEM_SPHERE",f"{logspath}/sphere.json")
]
#     dimensions=[2,3,4,5,6,7,8,9,10,15,30,50,100,500,750]
    dimensions=[15]
    populationSize=[30]
    config={'name':'sprtGraphExample',
            'problems':problems,
            'dimensions':dimensions,
            'populationSize':populationSize}
    runSPRTTestNMHH(LOGS_PATH_FROM_ROOT,ROOT,config)

def runSPRTTComputationalTimeExperiments():
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
              ("PROBLEM_SPHERE",f"{logspath}/sphere.json")]
    #dimensions=[2,3,4,5,6,7,8,9,10,15,30,50,100,500,750]
    dimensions=[2,100,750]
    populationSize=[30]
    iterations=10
    for i in range(iterations):
                config={'name':f'comptime_extended/{i}',
                        'problems':problems,
                        'dimensions':dimensions,
                        'populationSize':populationSize
                        }
                start_time = timer()
                runSPRTTestNMHH(LOGS_PATH_FROM_ROOT,ROOT,config)
                elapsed = timer() - start_time
                remaining_iterations = iterations - (i + 1)
                estimated_remaining = elapsed * remaining_iterations
                print(f"Iteration {i+1}/iterations completed in {elapsed:.2f}s. Estimated remaining time: {estimated_remaining:.2f}s")

def runSPRTClusteringSuite():
    populationSize=[120]
    config={
                'name':'/clustering/wine/constrained/sprt',
                'populationSize':populationSize
            }
    runSPRTClusterinProblems(LOGS_PATH_FROM_ROOT,ROOT,config)

def runpyNMHHClassificationSuite():
    classifiers=[
                {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                },
                 {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                ]
    problems=lambda logspath: [
                        #        {'name':"Digits"},
                        #        {'name':'Wine'},
                        #        {'name':'AuditRisk'},
                        #        {'name':'CervicalCancer'},
                        #        {'name':'GallStone'},
                        #        {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                               ]
    solutionConfigs=[{'populationSize':15,'baselevelIterations':500,'pyNMHHSteps':5}]
#     baseLevelConfigs=[classify.initialClassificationBaseLevelConfig()]
#     baseLevelConfigs=[classify.initialClassificationBaseLevelConfigBayes()]
    baseLevelConfigs=[classify.initialClassificationBaseLevelConfigEmphBayes()]
    
    config={
                'name':'/smallDatasets/biggerIter/smallerPop/bayesinit',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':baseLevelConfigs,
                'solver':'pyNMHH'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,pyNMHHClassificationExperiment)


def runpyNMHHClassificationSuite_HalfSA():
    classifiers=[
                {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                },
                 {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                ]
    problems=lambda logspath: [
                               {'name':"Digits"},
                               {'name':'Wine'},
                               {'name':'AuditRisk'},
                               {'name':'CervicalCancer'},
                               {'name':'GallStone'},
                               {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                               ]
    solutionConfigs=[{'populationSize':15,'baselevelIterations':250,'pyNMHHSteps':10,'hyperLevel':'HALF_SA'}]
#     baseLevelConfigs=[classify.initialClassificationBaseLevelConfig()]
#     baseLevelConfigs=[classify.initialClassificationBaseLevelConfigBayes()]
    baseLevelConfigs=[classify.initialClassificationBaseLevelConfigEmphBayes()]
    
    config={
                'name':'/smallDatasets/halfSA/smallerPop/bayesinit',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':baseLevelConfigs,
                'solver':'pyNMHH_HALF_SA'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,pyNMHHClassificationExperiment)


def runbayesGPClassificationSuite():
    classifiers=[
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                 {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                 {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                 {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
    problems=lambda logspath: [{'name':"Digits"},{'name':"Wine"},
                               {'name':'AuditRisk'},
                               {'name':'Iris'},
                               {'name':'CervicalCancer'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'iterations':150}]
    config={
                'name':'/smallDatasets/smallIter',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':10,
                'baseLevelConfigs':[None],
                'solver':'bayesGP'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,bayesGPClassificationExperiment)

def runBigbayesGPClassificationSuite():
    classifiers=[
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                 {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                 {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                 {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
    problems=lambda logspath: [{'name':"Digits"},{'name':"Wine"},
                               {'name':'AuditRisk'},
                        #        {'name':'Iris'},
                               {'name':'CervicalCancer'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'iterations':2500,'bayesCap':15}]
    config={
                'name':'/smallDatasets/bigIterEstimatorFix',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':[None],
                'solver':'bayesGP'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,bayesGPClassificationExperiment)

def runBigbayesGPPyNMHHClassificationSuite():
    classifiers=[
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                 {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                 {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                 {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
   
    problems=lambda logspath: [{'name':"Digits"},{'name':'Wine'},
                               {'name':'AuditRisk'},
                        #        {'name':'Iris'},
                               {'name':'CervicalCancer'},
                               {'name':'GallStone'},
                               {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'populationSize':15,'baselevelIterations':2500,'pyNMHHSteps':1}]
#     baseLevelConfigs=[classify.initialClassificationBaseLevelConfig()]
    baseLevelConfigs=[classify.initialClassificationBaseLevelConfigBayesGP()]
    
    config={
                'name':'/smallDatasets/pyNMHHBased',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':baseLevelConfigs,
                'solver':'bayesGP'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,pyNMHHClassificationExperiment)

def runBigbayesTPEClassificationSuite():
    classifiers=[
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                 {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                 {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                 {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
   
    problems=lambda logspath: [{'name':"Digits"},{'name':'Wine'},
                               {'name':'AuditRisk'},
                        #        {'name':'Iris'},
                               {'name':'CervicalCancer'},
                               {'name':'GallStone'},
                               {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'populationSize':15,'baselevelIterations':2500,'pyNMHHSteps':1}]
#     baseLevelConfigs=[classify.initialClassificationBaseLevelConfig()]
    baseLevelConfigs=[classify.initialClassificationBaseLevelConfigBayesTPE()]
    
    config={
                'name':'/smallDatasets/biggerIter',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':baseLevelConfigs,
                'solver':'bayesTPE'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,pyNMHHClassificationExperiment)

def runRandomSearchClassificationSuite():
    classifiers=[
                {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
    problems=lambda logspath: [{'name':"Digits"},{'name':"Wine"},
                               {'name':'AuditRisk'},
                        #        {'name':'Iris'},
                               {'name':'CervicalCancer'},
                               {'name':'GallStone'},
                               {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'iterations':500}]
    config={
                'name':'/smallDatasets/smallIter',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':5,
                'baseLevelConfigs':[None],
                'solver':'randomSearch'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,randomSearchClassificationExperiment)

def runGeneticSearchClassificationSuite():
    classifiers=[
                {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
    problems=lambda logspath: [{'name':"Digits"},{'name':"Wine"},
                               {'name':'AuditRisk'},
                        #        {'name':'Iris'},
                               {'name':'CervicalCancer'},
                               {'name':'GallStone'},
                               {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'iterations':50,'populationSize':50}]
    config={
                'name':'/smallDatasets/biggerIter',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':[None],
                'solver':'geneticSearch'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,geneticSearchClassificationExperiment)

def runGridSearchClassificationSuite():
    classifiers=[
                {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
    problems=lambda logspath: [{'name':"Digits"},{'name':"Wine"},
                               {'name':'AuditRisk'},
                        #        {'name':'Iris'},
                               {'name':'CervicalCancer'},
                               {'name':'GallStone'},
                               {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'iterations':2500}]
    config={
                'name':'/smallDatasets/biggerIter',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':[None],
                'solver':'gridSearch'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,gridSearchClassificationExperiment)

def runDefaultClassificationSuite():
    classifiers=[
                {
                  'name':'SVM',
                  'model':'SVM',
                  'hyperparameters': {
                        'C': [1.0,50.0],
                        "kernel":['linear','poly','rbf','sigmoid'],
                        'degree': [2, 5],
                        'gamma': ['scale', 'auto', 0.1, 1.0, 10.0],
                        'coef0': [0.0, 1.0],
                        'shrinking': [True, False],
                        'tol': [1e-5, 1e-3],
                        'class_weight': [None, 'balanced']
                        },
                },
                {
                  'name':'RandomForest',
                  'model':'RF',
                  'hyperparameters':  {
                        'n_estimators': [10,100],
                        "max_features":[1,64],
                        'max_depth': [5,50],
                        "min_samples_split":[2,11],
                        "min_samples_leaf":[1,11],
                        "criterion":['gini','entropy']
                   }
                },
                {
                 'name':'GradientBoost',
                 'model':'GBoost',
                 'hyperparameters': {
                        'n_estimators': [100, 1000],
                        'learning_rate': [0.01, 0.2],
                        'max_depth': [3, 8],
                        'min_samples_split': [2, 20],
                        'min_samples_leaf': [1, 8],
                        'subsample': [0.8, 1.0],
                        'max_features': ['sqrt', 'log2', None],
                        'criterion': ['friedman_mse', 'squared_error'],
                        'warm_start': [True, False],
                        'validation_fraction': [0.1, 0.2],
                        'n_iter_no_change': [5,20],
                        'tol': [1e-6, 1e-4]
                }
                },
                # {
                #  'name':'KNN',
                #  'model':'KNN',
                #  'hyperparameters': {
                #         'n_neighbors': [2, 5],
                #         'weights': ['uniform', 'distance'],
                #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                #         'leaf_size': [10, 100],
                #         'p': [1, 15],
                #         'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                # }
                # },
                {
                 'name':'DecisionTree',
                 'model':'DecisionTree',
                 'hyperparameters': {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [5, 50],
                        'min_samples_split': [2, 11],
                        'min_samples_leaf': [1, 11],
                        'max_features': [1,64],
                        'max_leaf_nodes': [2, 50]
                }
                }
                ]
    problems=lambda logspath: [{'name':"Digits"},{'name':"Wine"},
                               {'name':'AuditRisk'},
                        #        {'name':'Iris'},
                               {'name':'CervicalCancer'},
                               {'name':'GallStone'},
                               {'name':'HigherEducation'},
                               {'name':'HeartFailure'}
                        #        {'name':'BreastCancer'}
                               ]
    solutionConfigs=[{'iterations':2500}]
    config={
                'name':'/smallDatasets/defaultParams',
                'classifiers':classifiers,
                'problems': problems,
                'solutionConfigs':solutionConfigs,
                'classificationTuningCount':1,
                'baseLevelConfigs':[None],
                'solver':'defaultParameters'
    }
    runClassificationExperiments(LOGS_PATH_FROM_ROOT,ROOT,config,defaultClassificationExperiment)

# runSNLPSuite()
if __name__ == '__main__':
        runNMHHComputationalTimeExperiments()
# runCUSTOMHySComputationalTimeExperiments()
# # runRandomHHSuite()
# # runNMHHSuite()
# runShiftedNMHHSuite()
# # runCUSTOMHySSuite()
# # runMealpySuite()
        # runShiftedMealpySuite() 
#   runShiftedCUSTOMHySSuite()
# runShiftedMealpySuite()
# # runClusteringSuite()
# # runSPRTTTestNMHHSuite()
        runSPRTTComputationalTimeExperiments()
# # runSPRTClusteringSuite()
# # runSPRTTTestNMHHSuite()

# # runpyNMHHClassificationSuite()
# # runpyNMHHClassificationSuite_HalfSA()

# # runbayesGPClassificationSuite()
# # runBigbayesGPClassificationSuite()

# runBigbayesGPPyNMHHClassificationSuite()
# runBigbayesTPEClassificationSuite()

# runRandomSearchClassificationSuite()
# runGridSearchClassificationSuite()
# runDefaultClassificationSuite()

# runGeneticSearchClassificationSuite()