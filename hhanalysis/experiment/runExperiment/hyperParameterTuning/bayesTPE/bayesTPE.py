# import sys
# import os

# sys.path.insert(0, '../')
# sys.path.insert(0, '../..')

# import numpy as np
# import json
# import subprocess
# import itertools
# import os
# import copy
# from timeit import default_timer as timer
# import pandas as pd

# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# from skopt.space import Real, Integer,Categorical
# from skopt.utils import use_named_args
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# from sklearn.model_selection import cross_val_score
# from skopt import gp_minimize

# from runExperiment.commonRun import *
# from runExperiment.classification.classifiers import getClassifiers
# from runExperiment.classification.datasets import getDatasets
# import ipynb.fs.full.runExperiment.hyperParameterTuning.pyNMHH.classification.classify as classify


# def add_pre_evaluated_point(trials,tid, params, loss):
#     trial = {
#         'tid': tid,
#         'result': {'status': STATUS_OK, 'loss': loss},
#         'misc': {
#             'tid': tid,
#             'cmd': ('domain_attachment', 'FMinIter_Domain'),
#             'workdir': None,
#             'idxs': {k: [tid] for k in params},
#             'vals': {k: [v] for k, v in params.items()}
#         },
#         'spec': None,
#         'state': 2,
#         'owner': None,
#         'book_time': None,
#         'refresh_time': None,
#         'exp_key': None
#     }
#     trials.insert_trial_doc(trial)
#     return trials

# def generate_trials_to_calculate(points,results):
#     """
#     Function that generates trials to be evaluated from list of points

#     :param points: List of points to be inserted in trials object in form of
#         dictionary with variable names as keys and variable values as dict
#         values. Example value:
#         [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 1.0}]

#     :return: object of class base.Trials() with points which will be calculated
#         before optimisation start if passed to fmin().
#     """
#     trials = Trials()
#     [add_pre_evaluated_point(trials,tid,x,y) for tid, (x,y) in enumerate(zip(points,results))]
#     return trials

# def snapToType(params,func):
#     listparams=[]
#     for (i,(lower,upper,type)) in enumerate(zip(func.lowerbounds,func.upperbounds,func.xtypes)):
#           if type == 'discrete':
#                listparams.append(int(np.round(params[i])))
#           else:
#                listparams.append(params[i])
#     return listparams

# def toTPEParams(func):
#     tpeParamconfig={}
#     for (i,(lower,upper,type)) in enumerate(zip(func.lowerbounds,func.upperbounds,func.xtypes)):
#             if type == 'continuous':
#                 tpeParamconfig[str(i)]=(hp.uniform(str(i),float(lower),float(upper)))
#             elif type == 'discrete':
#                 tpeParamconfig[str(i)]=(hp.quniform(str(i),int(lower),int(upper),1))
#             else:
#                 print(f'Invalid func config {func}')
#                 exit 
#     return tpeParamconfig


# def pyNMHHHyperParametersToTPE(paramConfig):
#     gpParamconfig=[]
#     for (key,value) in paramConfig.items():
#             if isinstance(value[0], int) and not isinstance(value[0], bool):
#                 gpParamconfig.append(Integer(value[0],value[1],name=key))
#             elif isinstance(value[0], float) and not isinstance(value[0], bool):
#                 gpParamconfig.append(Real(value[0],value[1],name=key))
#             else:
#                 converted=[]
#                 for category in value:
#                     if isinstance(category,np.str_):
#                           converted.append(str(category))
#                     else:
#                         converted.append(category)
#                 gpParamconfig.append(Categorical(converted,name=key))
#     return gpParamconfig

# def unflatten(flatParams):
#     unflattened={}
#     for i,val in enumerate(flatParams):
#         unflattened[str(i)]=val
#     return unflattened

# def bayesTPETuning(config):      
#     gpParams=pyNMHHHyperParametersToTPE((config['hyperParameters']))
#     @use_named_args(gpParams)
#     def objective(**params):
#         clf = config['classifier'](**castToRightString(params))
#         scores = cross_val_score(clf, config['X'], config['Y'],cv=config['crossValidations'],scoring='accuracy',n_jobs=-1)
#         return -np.mean(scores)
#     if 'bayesCap'in config:
#         print(f'BayesCap of {config["bayesCap"]} enabled')
#         estimator=None
#         res_gp = gp_minimize(objective, gpParams, base_estimator=estimator,n_calls=config['bayesCap'], random_state=0,verbose=True,n_jobs=-1,n_restarts_optimizer=1, model_queue_size=config['bayesCap'])
#         solx=res_gp.x_iters[-(config['bayesCap']):]
#         soly=res_gp.func_vals[-(config['bayesCap']):]
#         solx=[flattenFromString(sol,config['hyperParameters']) for sol in solx]
#         for i in range(config['bayesTPEIterations']//config['bayesCap']-1):
#             print(f'\ntotal func evals={(i+1)*config["bayesCap"]}\n')
#             # start_time = time.time()
#             # callback = partial(callback_with_timer, start_time=start_time)

#             res_gp = gp_minimize(objective, gpParams,base_estimator=estimator, x0=solx,y0=soly,n_calls=config['bayesCap'],n_initial_points=0, random_state=0,verbose=True,n_jobs=-1,n_restarts_optimizer=1, model_queue_size=config['bayesCap'])
#             bestIndex=np.where(res_gp.func_vals == res_gp.fun)[-1][0]
#             if bestIndex>len(solx):
#                 estimator=res_gp.models[bestIndex-len(solx)]
#             newsolx=res_gp.x_iters[-(config['bayesCap']):]
#             newsolx=[flattenFromString(sol,config['hyperParameters']) for sol in newsolx]
#             newsoly=res_gp.func_vals[-(config['bayesCap']):]
#             for i in range(config['bayesCap']):
#                  if newsoly[i]<soly[i]:
#                       solx[i]=newsolx[i]
#                       soly[i]=newsoly[i]
#             print(solx)
#     else:
#         res_gp = gp_minimize(objective, gpParams, n_calls=config['bayesTPEIterations'], random_state=0,verbose=True,n_jobs=-1)
        
#     print("Accuracy:%.4f" % -res_gp.fun)
#     print(res_gp.x)
#     return {
#                 "bestAccuracy":-res_gp.fun,
#                 'solution':json.loads(json.dumps(unflatten(res_gp.x,config['hyperParameters']))),
#             }

# def bayesTPETunings(config):
#     solutions=[]
#     experimentTimes=[]
#     for i in range(config['classificationTuningCount']):
#         print(f'            >>>Running  iteration {i+1}/{config["classificationTuningCount"]}\n')
#         start = timer()
#         solutions.append(bayesTPETuning(config))
#         end = timer()
#         elapsed=end-start
#         experimentTimes.append(elapsed)
#         etaSeconds=(config['classificationTuningCount']-(i+1))*np.mean(experimentTimes)
#         print(f'            >>>Ran  {i+1}/{config["classificationTuningCount"]} iterations elapsed seconds {elapsed} eta: {etaSeconds} seconds')
#     return solutions

# def toPersist(config,experiment,solutions):
#         persistedData=copy.copy(config)
#         persistedData['hyperParameters']=experiment['classifier']['hyperparameters']
#         persistedData['classifierModel']=experiment['classifier']['model'] 
#         persistedData['solutions']=json.loads(json.dumps(solutions))
#         persistedData['totalFunctionEvaluations']=config['bayesTPEIterations']
#         persistedData['solver']=experiment['solver']
#         persistedData.pop('X')
#         persistedData.pop('classifier')
#         persistedData.pop('Y')
#         return persistedData

# def bayesTPEClassificationExperiment(experiment,recordsPath,experimentId):
#             print(f"        >>>Running experiment {experimentId}")
#             start = timer()
#             dataset=getDatasets()[experiment['problems']['name']]()
#             config={
#                         'X':dataset.data,
#                         'Y':dataset.target,
#                         'hyperParameters':experiment['classifier']['hyperparameters'],
#                         'crossValidations':3,
#                         'bayesTPEIterations': experiment['solutionConfigs']['iterations'],
#                         'classifier':getClassifiers()[experiment['classifier']['model']](),
#                         'classificationTuningCount':experiment['classificationTuningCount'],
#                         'classifierName':experiment['classifier']['name'],
#                         'datasetName':experiment['problems']['name']
#                     }
#             if 'bayesCap' in experiment['solutionConfigs']:
#                 config['bayesCap']=experiment['solutionConfigs']['bayesCap']

#             solutions=bayesTPETunings(config)
#             end = timer()
#             metadata={"elapsedTimeSec":end-start}            

#             recordExperiment(toPersist(config,experiment,solutions),experimentId,recordsPath,metadata)
#             return metadata
